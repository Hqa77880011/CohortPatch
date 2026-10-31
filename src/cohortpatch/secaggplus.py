"""Real Flower 1.23 SecAgg+ with a durable gate immediately before unmasking.

The local Grid executes unchanged Flower client crypto and protobuf serialization.
It measures application messages, not network captures. Every roster change needs
a new invocation/session and fresh client keys; stale authorization never unmasks.
"""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import secrets
import time
from types import SimpleNamespace
import uuid

import numpy as np

from .protocol import RosterLedger


def secure_sum(vectors, ledger, session, *, threshold, clipping_range=8.0,
               quantization_range=2**22, drop=None):
    """Execute all four real SecAgg+ stages for one already-reserved roster."""
    if version("flwr") != "1.23.0":
        raise RuntimeError("This checked adapter requires flwr==1.23.0")
    import flwr.common.recorddict_compat as compat
    from flwr.client import ClientApp, NumPyClient
    from flwr.client.mod import secaggplus_mod
    from flwr.common import ConfigRecord, Context, FitIns, Message, Metadata, RecordDict, ndarrays_to_parameters
    from flwr.common.serde import message_from_proto, message_to_proto
    from flwr.common.secure_aggregation.secaggplus_constants import Key, RECORD_KEY_CONFIGS
    from flwr.common.typing import Run
    from flwr.server import Grid, LegacyContext
    from flwr.server.strategy import FedAvg
    from flwr.server.workflow import SecAggPlusWorkflow
    from flwr.server.workflow.constant import MAIN_CONFIGS_RECORD, MAIN_PARAMS_RECORD, Key as WorkflowKey

    roster = sorted(vectors)
    if (len(roster) < 2 or not 2 <= threshold <= len(roster)
            or len({np.shape(vector) for vector in vectors.values()}) != 1
            or any(not np.isfinite(vector).all() or np.max(np.abs(vector), initial=0) > clipping_range
                   for vector in vectors.values())):
        raise ValueError("Aligned bounded client vectors and 2 <= threshold <= roster required")
    row = ledger.db.execute('SELECT roster,status,answer FROM sessions WHERE id=?', (session,)).fetchone()
    if row is None or json.loads(row[0]) != roster:
        raise ValueError("Native session must match an already durable authorized roster")
    if row[1] == "released":
        return np.asarray(json.loads(row[2])), dict(replayed=True, source="stored native sum")
    ledger.db.execute('CREATE TABLE IF NOT EXISTS native_runs (session TEXT PRIMARY KEY)')
    if ledger.db.execute('SELECT 1 FROM native_runs WHERE session=?', (session,)).fetchone():
        raise RuntimeError("Uncertain native attempt retained; use a new safety-checked session")
    with ledger.db:
        ledger.db.execute('INSERT INTO native_runs VALUES (?)', (session,))
    nodes = {1000 + index: client for index, client in enumerate(roster)}

    class VectorClient(NumPyClient):
        def __init__(self, client):
            self.client = client

        def fit(self, parameters, config):
            return [np.asarray(vectors[self.client], np.float64)], 1, {}

    app = ClientApp(client_fn=lambda context: VectorClient(context.node_config["client"]).to_client(),
                    mods=[secaggplus_mod])

    class SerializedGrid(Grid):
        def __init__(self):
            self._run = Run.create_empty(secrets.randbits(63))
            self.contexts = {node: Context(self._run.run_id, node, {"client": client}, RecordDict(), {})
                             for node, client in nodes.items()}
            self.replies = {}
            self.bytes = {"server_to_client": 0, "client_to_server": 0}
            self.stages = []

        def set_run(self, run_id):
            self._run = Run.create_empty(run_id)

        @property
        def run(self):
            return self._run

        def get_node_ids(self):
            return list(nodes)

        def create_message(self, content, message_type, dst_node_id, group_id, ttl=None):
            metadata = Metadata(self.run.run_id, uuid.uuid4().hex, 0, dst_node_id, "", group_id,
                                time.time(), ttl or 3600, message_type)
            return Message(metadata=metadata, content=content)

        def push_messages(self, messages):
            identifiers = []
            for message in messages:
                proto = message_to_proto(message)
                proto.metadata.run_id = self.run.run_id
                proto.metadata.src_node_id = 0
                proto.metadata.message_id = uuid.uuid4().hex
                encoded = proto.SerializeToString()
                self.bytes["server_to_client"] += len(encoded)
                incoming = message_from_proto(type(proto).FromString(encoded))
                stage = str(incoming.content.config_records[RECORD_KEY_CONFIGS][Key.STAGE])
                if stage not in self.stages:
                    self.stages.append(stage)
                node = incoming.metadata.dst_node_id
                identifier = incoming.metadata.message_id
                identifiers.append(identifier)
                if drop and nodes[node] == drop[0] and stage == drop[1]:
                    continue
                response = app(incoming, self.contexts[node])
                reply = message_to_proto(response)
                encoded_reply = reply.SerializeToString()
                self.bytes["client_to_server"] += len(encoded_reply)
                self.replies[identifier] = message_from_proto(type(reply).FromString(encoded_reply))
            return identifiers

        def pull_messages(self, message_ids):
            return [self.replies.pop(identifier) for identifier in message_ids if identifier in self.replies]

        def send_and_receive(self, messages, *, timeout=None):
            return self.pull_messages(self.push_messages(messages))

    class ExactStrategy(FedAvg):
        captured = None

        def configure_fit(self, server_round, parameters, client_manager):
            return [(SimpleNamespace(node_id=node, cid=str(node)), FitIns(parameters, {})) for node in nodes]

        def aggregate_fit(self, server_round, results, failures):
            if {proxy.node_id for proxy, _ in results} != set(nodes) or any(result.num_examples != 1 for _, result in results):
                raise ValueError("Secure aggregate must use the authorized equal-weight roster")
            answer = super().aggregate_fit(server_round, results, failures)
            self.captured = answer[0]
            return answer

    class GatedWorkflow(SecAggPlusWorkflow):
        def unmask_stage(self, grid, context, state):
            if {nodes[node] for node in state.active_node_ids} != set(roster):
                raise ValueError("Roster drift: no unmask; fresh setup and safety authorization required")
            ledger.freeze(session, roster)
            return super().unmask_stage(grid, context, state)

    grid = SerializedGrid()
    strategy = ExactStrategy(min_fit_clients=len(roster), min_available_clients=len(roster),
                            accept_failures=False, initial_parameters=ndarrays_to_parameters([np.zeros_like(next(iter(vectors.values())))]))
    context = LegacyContext(Context(grid.run.run_id, 0, {}, RecordDict(), {}), strategy=strategy)
    context.state.config_records[MAIN_CONFIGS_RECORD] = ConfigRecord({WorkflowKey.CURRENT_ROUND: 1})
    initial = ndarrays_to_parameters([np.zeros_like(next(iter(vectors.values())))])
    context.state.array_records[MAIN_PARAMS_RECORD] = compat.parameters_to_arrayrecord(initial, keep_input=True)
    workflow = GatedWorkflow(num_shares=1.0, reconstruction_threshold=threshold / len(roster), max_weight=1.0,
                            clipping_range=clipping_range, quantization_range=quantization_range, modulus_range=2**32)
    workflow(grid, context)
    if strategy.captured is None:
        raise RuntimeError("SecAgg+ aborted without a released sum; reservation retained")
    from flwr.common import parameters_to_ndarrays
    answer = parameters_to_ndarrays(strategy.captured)[0] * len(roster)
    ledger.commit_aggregate(session, answer)
    return answer, dict(backend="Flower SecAgg+", version=version("flwr"),
                        transport="local serialized protobuf", bytes=grid.bytes, stages=grid.stages,
                        threshold=threshold, quantization_range=quantization_range, equal_client_weights=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Trusted client-side NPZ client_ids[N], vectors[N,D]")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--policy", required=True, help="JSON initial_classes,k,capacity,threshold")
    parser.add_argument("--session", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--drop-client", type=int)
    parser.add_argument("--drop-stage", default="collect_masked_vectors")
    parser.add_argument('--clipping-range',type=float,default=8.0)
    args = parser.parse_args()
    policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    arrays = dict(np.load(args.data, allow_pickle=False))
    ledger = RosterLedger(args.ledger, policy["initial_classes"], policy["k"], policy["capacity"], policy["threshold"])
    vectors = dict(zip(arrays["client_ids"].tolist(), arrays["vectors"]))
    answer, report = secure_sum(vectors, ledger, args.session, threshold=policy["threshold"],
                               clipping_range=args.clipping_range,
                               drop=None if args.drop_client is None else (args.drop_client, args.drop_stage))
    ledger.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(answer=answer.tolist(), **report), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
