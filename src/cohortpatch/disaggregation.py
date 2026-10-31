"""Lam et al. participant-nullspace attack, independently implemented from paper.

The author artifact has no license file; it is not vendored or imported. Binary
quadratic nullspace minimization is solved exactly as a linearized MILP with
HiGHS. Public known-matrix adaptation bypasses participant inference and solves
the original fixed-update least-squares system. FedAvg drift is model mismatch,
not exact dynamic-gradient recovery; the separate trajectory module tests it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import lil_matrix,vstack,csr_matrix


def infer_participation(sums,summary_matrix,counts,clients,*,seconds=60):
    """Minimize each binary participant column's residual in the left nullspace."""
    sums=np.asarray(sums,float);summary=np.asarray(summary_matrix,float);counts=np.asarray(counts,float)
    rounds=len(sums)
    if (sums.ndim!=2 or summary.ndim!=2 or summary.shape[1]!=rounds or counts.shape!=(len(summary),clients)
            or clients<1 or not all(np.isfinite(a).all() for a in (sums,summary,counts))
            or not np.isin(summary,[0,1]).all() or not np.equal(counts,np.floor(counts)).all()):
        raise ValueError('Finite aggregate sums and exact public binary-window participation summaries required')
    u,singular,_=np.linalg.svd(sums,full_matrices=True)
    null=u[:,min(clients,len(u)):]
    quadratic=null@null.T
    pairs=[(i,j) for i in range(rounds) for j in range(i+1,rounds)]
    objective=np.r_[np.diag(quadratic),[2*quadratic[i,j] for i,j in pairs]]
    rows=len(summary)+3*len(pairs)
    matrix=lil_matrix((rows,rounds+len(pairs)))
    matrix[:len(summary),:rounds]=summary
    lower=np.full(rows,-np.inf);upper=np.zeros(rows)
    for index,(i,j) in enumerate(pairs):
        variable=rounds+index;row=len(summary)+3*index
        matrix[row,variable]=1;matrix[row,i]=-1
        matrix[row+1,variable]=1;matrix[row+1,j]=-1
        matrix[row+2,variable]=-1;matrix[row+2,i]=1;matrix[row+2,j]=1;upper[row+2]=1
    result=[];reports=[]
    for client in range(clients):
        lo,hi=lower.copy(),upper.copy();lo[:len(summary)]=counts[:,client];hi[:len(summary)]=counts[:,client]
        solved=milp(objective,integrality=np.r_[np.ones(rounds),np.zeros(len(pairs))],
                    bounds=Bounds(0,1),constraints=LinearConstraint(matrix.tocsr(),lo,hi),options=dict(time_limit=seconds))
        if not solved.success:
            raise RuntimeError('Participant inference did not prove an optimal solution: '+solved.message)
        column=np.rint(solved.x[:rounds]).astype(int)
        exclude=np.r_[1-2*column,np.zeros(len(pairs))]
        ambiguity_matrix=vstack([matrix.tocsr(),csr_matrix(objective[None,:]),csr_matrix(exclude[None,:])])
        alternative=milp(np.zeros(len(objective)),integrality=np.r_[np.ones(rounds),np.zeros(len(pairs))],
                         bounds=Bounds(0,1),constraints=LinearConstraint(ambiguity_matrix,
                         np.r_[lo,-np.inf,1-column.sum()],np.r_[hi,solved.fun+1e-8,np.inf]),options=dict(time_limit=seconds))
        unique=True if alternative.status==2 else (False if alternative.success else None)
        result.append(column)
        reports.append(dict(client=client,nullspace_residual=float(np.square(null.T@column).sum()),optimal=True,
                            unique_optimal_participation=unique))
    participation=np.stack(result,axis=1)
    return participation,dict(clients=reports,aggregate_singular_values=singular.tolist(),
                              assumed_fixed_update_rank=clients,public_summary_windows=len(summary),
                              interpretation='a column is identifiable only when its excluded-solution MILP proves no alternative optimum')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True,help='NPZ sums[R,D], optional known_roster or summary_matrix/counts')
    parser.add_argument('--clients',type=int,required=True)
    parser.add_argument('--seconds-per-client',type=float,default=60)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();data=dict(np.load(args.data,allow_pickle=False))
    if 'known_roster' in data:
        participation=np.asarray(data['known_roster'])
        if participation.shape!=(len(data['sums']),args.clients) or not np.isin(participation,[0,1]).all():
            raise ValueError('Known-matrix adaptation needs the actual aligned public binary roster')
        report=dict(mode='known_matrix_adaptation',participant_inference='not required by this public-roster threat model')
    else:
        participation,report=infer_participation(data['sums'],data['summary_matrix'],data['counts'],args.clients,seconds=args.seconds_per_client)
        report['mode']='original_nullspace_participant_inference'
    updates=np.linalg.pinv(participation)@data['sums']
    report.update(implementation='independent paper algorithm; no unlicensed author code imported',
                  fixed_model_residual=float(np.square(participation@updates-data['sums']).sum()),
                  dynamic_interpretation='FedAvg drift is noise/model mismatch; these are fitted fixed update vectors')
    if 'scorer_roster' in data:
        report['participant_bit_error']=float(np.mean(participation!=data['scorer_roster']))
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'disaggregation.npz',roster=participation,updates=updates)
    (output/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
