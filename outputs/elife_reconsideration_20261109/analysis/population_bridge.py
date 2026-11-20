"""Finite, correlation-aware reanalysis of archived synchronous SNN responses."""
from itertools import product
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
from scipy.special import xlogy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'outputs/snn_budget_frontier_20261016'
sys.path.insert(0, str(OLD))
from frontier import gibbs

LN2 = np.log(2.0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def entropy(p, axis=None):
    p = np.asarray(p, dtype=float)
    return -np.sum(xlogy(p, p), axis=axis) / LN2


def kl(p, q):
    p, q = np.broadcast_arrays(p, q)
    mask = p > 0
    if np.any(q[mask] <= 0):
        raise ValueError('Reference has zero mass on observed support')
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def count_decomposition(spikes, expected):
    # template,repeat,cell,nonoverlapping-window,bin -> cell,samples
    counts = spikes.reshape(10, 8, 75, 2, 1000).sum(axis=-1)
    rows = np.moveaxis(counts, 2, 0).reshape(75, -1).astype(int)
    maximum = int(np.ceil(1000 / 1.1))
    p_i = np.array([np.bincount(row, minlength=maximum+1)/row.size for row in rows])
    p_pool = p_i.mean(axis=0)
    mean = float(np.dot(p_pool, np.arange(len(p_pool))))
    q, beta = gibbs(mean, maximum)
    logq = (-beta*np.arange(maximum+1) - np.log(np.exp(-beta*np.arange(maximum+1)).sum())) / LN2
    cross_entropy = -float(np.dot(p_pool, logq))
    pool_h = float(entropy(p_pool))
    marginal_h = float(entropy(p_i, axis=1).mean())
    pooled_kl = cross_entropy - pool_h
    mean_marginal_kl = cross_entropy - marginal_h
    heterogeneity = pool_h - marginal_h
    np.testing.assert_allclose(pooled_kl, expected['deficit_nats']/LN2, atol=3e-12, rtol=0)
    np.testing.assert_allclose(mean, expected['mean_count'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(mean_marginal_kl, pooled_kl+heterogeneity, atol=1e-12, rtol=0)
    return {'mean_count':mean,'pooled_kl_bits':pooled_kl,'pooled_G_bits_per_spike':pooled_kl/mean,
            'cell_heterogeneity_bits':heterogeneity,'mean_marginal_kl_bits':mean_marginal_kl,
            'mean_marginal_gap_per_spike':mean_marginal_kl/mean,
            'heterogeneity_per_spike':heterogeneity/mean,
            'samples_per_cell':rows.shape[1]}


def select_groups(seed, cells, group_count):
    rng = np.random.default_rng(seed+110900+cells)
    groups = set()
    while len(groups) < group_count:
        groups.add(tuple(sorted(rng.choice(75, cells, replace=False))))
    return np.array(sorted(groups), dtype=int)


def total_correlation(binary, groups):
    # binary shape: stimulus/window, repeat, cell. Preserve matching sample order.
    flat = binary.reshape(-1, binary.shape[-1])
    marginal = entropy(np.stack([flat.mean(axis=0), 1-flat.mean(axis=0)]), axis=0)
    chosen = flat[:, groups]
    weights = (1 << np.arange(groups.shape[1], dtype=np.uint16))
    words = np.sum(chosen*weights, axis=-1, dtype=np.uint16)
    joint = np.array([entropy(np.bincount(words[:, g], minlength=1 << groups.shape[1])/len(flat))
                      for g in range(len(groups))])
    tc = marginal[groups].sum(axis=1)-joint
    if not np.isfinite(tc).all() or tc.min() < -1e-10:
        raise ValueError('Invalid total-correlation estimate')
    return tc


def conditional_shuffle(binary, rng):
    # An independent permutation across repeats for every condition and neuron.
    permutation = np.argsort(rng.random(binary.shape), axis=1)
    shuffled = np.take_along_axis(binary, permutation, axis=1)
    np.testing.assert_array_equal(shuffled.sum(axis=1), binary.sum(axis=1))
    return shuffled


def diagnostic(spikes, config, seed, shuffle_seed):
    window, cells = config['window_ms'],config['cells_per_group']
    repeats = config.get('repeat_subset', list(range(8)))
    data = spikes[:, repeats]
    # Sample conditions are fixed input-template and relative window index.
    occupancy = data.reshape(10,len(repeats),75,2000//window,window).any(axis=-1)
    binary = occupancy.transpose(0,3,1,2).reshape(-1,len(repeats),75).astype(np.uint8)
    groups = select_groups(seed,cells,config['groups'])
    observed = total_correlation(binary,groups)
    rng = np.random.default_rng(shuffle_seed+seed+1000*window+100*cells+len(repeats))
    surrogate = np.array([total_correlation(conditional_shuffle(binary,rng),groups)
                          for _ in range(config['shuffles'])])
    result = {'window_ms':window,'cells':cells,'repeats':len(repeats),'groups':len(groups),
              'shuffles':len(surrogate),'response_samples':int(binary.shape[0]*binary.shape[1]),
              'observed_TC_bits':float(observed.mean()),'conditional_shuffle_TC_bits':float(surrogate.mean()),
              'signed_excess_bits':float(observed.mean()-surrogate.mean()),
              'shuffle_mean_MC_SD_bits':float(surrogate.mean(axis=1).std(ddof=1)/np.sqrt(len(surrogate))),
              'group_neurons':groups.tolist()}
    if not all(np.isfinite(v) for v in result.values() if isinstance(v,float)):
        raise ValueError('Non-finite diagnostic')
    return result


def sign_flip(values):
    x = np.asarray(values,float)
    signs = np.array(list(product((-1,1),repeat=len(x))))
    null = np.sum(signs*x[None,:],axis=1)
    return float(np.mean(np.abs(null) >= abs(x.sum())-1e-12))


def holm(pvalues):
    p = np.asarray(pvalues)
    order = np.argsort(p)
    out = np.zeros_like(p)
    out[order] = np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    return out.tolist()


def self_test():
    rng = np.random.default_rng(719)
    errors=[]
    for _ in range(30):
        shape=(3,3,3)
        p=rng.dirichlet(np.ones(np.prod(shape))).reshape(shape)
        marg=[p.sum(axis=tuple(j for j in range(3) if j!=i)) for i in range(3)]
        pool=np.mean(marg,axis=0)
        # Shift mass toward zero to stay on the active upper-budget branch.
        p=p*.4
        p[0,0,0]+=.6
        marg=[p.sum(axis=tuple(j for j in range(3) if j!=i)) for i in range(3)]
        pool=np.mean(marg,axis=0)
        mean=pool@np.arange(3)
        q,_=gibbs(mean,2)
        qq=q[:,None,None]*q[None,:,None]*q[None,None,:]
        joint_gap=float(entropy(qq)-entropy(p))
        tc=sum(float(entropy(m)) for m in marg)-float(entropy(p))
        heterogeneity=sum(kl(m,pool) for m in marg)
        decomposition=3*kl(pool,q)+heterogeneity+tc
        errors.append(abs(joint_gap-decomposition))
        np.testing.assert_allclose(joint_gap,kl(p,qq),atol=1e-11)
        np.testing.assert_allclose(joint_gap,decomposition,atol=1e-11)
    binary=rng.integers(0,2,(200,8,75),dtype=np.uint8)
    shuffled=conditional_shuffle(binary,rng)
    np.testing.assert_array_equal(binary.mean(axis=1),shuffled.mean(axis=1))
    independent=np.array(list(product((0,1),repeat=4)),dtype=np.uint8).reshape(2,8,4)
    groups=np.array([[0,1,2,3]])
    np.testing.assert_allclose(total_correlation(independent,groups),0,atol=1e-12)
    copied=np.repeat(np.array([0,1],dtype=np.uint8)[:,None,None],4,axis=2)
    np.testing.assert_allclose(total_correlation(copied,groups),3,atol=1e-12)
    assert sign_flip(np.ones(8))==2/256
    np.testing.assert_allclose(holm([.01,.04,.02]),[.03,.04,.04])
    return {'random_joint_distributions':30,'max_decomposition_error_bits':max(errors),
            'conditional_marginals_preserved':True,'independence_and_copy_controls':True,
            'exact_seed_inference_checked':True}


def run():
    protocol=json.loads((HERE/'protocol.json').read_text())
    configs=[protocol['primary_dependence_diagnostic']]+protocol['sensitivity_diagnostics']
    checks=self_test()
    source_manifest={'protocol_sha256':sha(HERE/'protocol.json'),'files':{}}
    rows=[]
    for seed in protocol['training_seeds']:
        for budget in protocol['total_activity_targets']:
            tag=f'seed_{seed}_budget_{budget}'
            dest=HERE/(tag+'.json')
            paths=[OLD/'replication'/(tag+s) for s in ('.json','_responses.npz')]
            inputs={str(p.relative_to(ROOT)):sha(p) for p in paths}
            source_manifest['files'].update(inputs)
            if dest.exists():
                cached=json.loads(dest.read_text())
                assert cached['input_hashes']==inputs and cached['protocol_sha256']==source_manifest['protocol_sha256']
                assert cached['analysis_code_sha256']==sha(__file__)
                rows.extend(cached['endpoints'])
                print('Loaded',tag,flush=True)
                continue
            saved=json.loads(paths[0].read_text())
            archive=np.load(paths[1],allow_pickle=False)
            assert archive['packed_spikes'].shape==(5,10,8,75,250)
            assert int(archive['n_bins'])==2000
            endpoints=[]
            for stage in protocol['stages']:
                spikes=np.unpackbits(archive['packed_spikes'][stage],axis=-1,count=2000,bitorder='little')
                expected=next(x for x in saved['checkpoints'][stage]['distributions'] if x['group']=='all' and x['window_ms']==1000)
                decomposition=count_decomposition(spikes,expected)
                row={'seed':seed,'budget':budget,'stage':stage,'reconstruction_error':saved['checkpoints'][stage]['waveform_re'],
                     'count_decomposition':decomposition,
                     'diagnostics':[diagnostic(spikes,c,seed,protocol['shuffle_seed']) for c in configs]}
                endpoints.append(row)
                print(tag,'stage',stage,'G',decomposition['pooled_G_bits_per_spike'],
                      'heterogeneity',decomposition['cell_heterogeneity_bits'],
                      'TCexcess',row['diagnostics'][0]['signed_excess_bits'],flush=True)
            payload={'input_hashes':inputs,'protocol_sha256':source_manifest['protocol_sha256'],
                     'analysis_code_sha256':sha(__file__),'endpoints':endpoints}
            dest.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
            rows.extend(endpoints)
    (HERE/'source_manifest.json').write_text(json.dumps(source_manifest,indent=2)+'\n')
    results={'checks':checks,'analysis_protocol':protocol,'count_results':[],'dependence_results':[]}
    pvalues=[]
    for budget in protocol['total_activity_targets']:
        initial=sorted([r for r in rows if r['budget']==budget and r['stage']==0],key=lambda r:r['seed'])
        learned=sorted([r for r in rows if r['budget']==budget and r['stage']==4],key=lambda r:r['seed'])
        assert len(initial)==len(learned)==8
        summary={'budget':budget,'per_cell_target':budget/75,'metrics':{}}
        for key in initial[0]['count_decomposition']:
            a=np.array([r['count_decomposition'][key] for r in initial])
            b=np.array([r['count_decomposition'][key] for r in learned])
            summary['metrics'][key]={'initial':float(a.mean()),'learned':float(b.mean()),'delta_mean':float((b-a).mean()),
                                     'delta_SD':float((b-a).std(ddof=1)),'decreases':int((b<a).sum()),'seed_deltas':(b-a).tolist()}
        results['count_results'].append(summary)
        for config_index,config in enumerate(configs):
            item={'budget':budget,'config_index':config_index,'config':config,'metrics':{}}
            for key in ['observed_TC_bits','conditional_shuffle_TC_bits','signed_excess_bits']:
                a=np.array([r['diagnostics'][config_index][key] for r in initial])
                b=np.array([r['diagnostics'][config_index][key] for r in learned])
                delta=b-a
                item['metrics'][key]={'initial':float(a.mean()),'learned':float(b.mean()),'delta_mean':float(delta.mean()),
                                      'delta_SD':float(delta.std(ddof=1)),'decreases':int((delta<0).sum()),'seed_deltas':delta.tolist()}
            if config_index==0:
                pvalue=sign_flip(item['metrics']['signed_excess_bits']['seed_deltas'])
                item['primary_signed_excess_change_p']=pvalue
                pvalues.append(pvalue)
            results['dependence_results'].append(item)
    adjusted=holm(pvalues)
    for index,item in enumerate([r for r in results['dependence_results'] if r['config_index']==0]):
        item['primary_signed_excess_change_p_Holm']=adjusted[index]
    # Store every endpoint/setting, not only the summary selected for the figure.
    flat=[]
    for row in rows:
        for index,diag in enumerate(row['diagnostics']):
            flat.append({k:row[k] for k in ['seed','budget','stage','reconstruction_error']}|{'config_index':index}|
                        {k:v for k,v in diag.items() if k!='group_neurons'})
    with (HERE/'dependence_measurements.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]))
        writer.writeheader();writer.writerows(flat)
    results['endpoints']=rows
    (HERE/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in results.items() if k not in ['endpoints','analysis_protocol']},indent=2),flush=True)


if __name__=='__main__':
    if '--self-test' in sys.argv:
        print(json.dumps(self_test(),indent=2))
    else:
        run()
