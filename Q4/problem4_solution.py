#!/usr/bin/env python3
"""F题 Q4：可审计的 Loss 桥接、条件贡献分解与能力前沿情景预测。
Python >=3.10；依赖 numpy pandas scipy matplotlib。Mac/Windows/Linux 通用。
继承 Q2 实际输出，直接调用 Q3 求解器；不在 Q4 重读 A/B 原始数据。
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, re, sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.special import expit, logit

BASE = Path(__file__).resolve().parent
TASKS = ['IFEval', 'BBH', 'MATH Lvl 5', 'GPQA', 'MUSR', 'MMLU-PRO']
BTASKS = ['LB_IFEval', 'LB_BBH', 'LB_MATH', 'LB_GPQA', 'LB_MUSR', 'LB_MMLU_PRO']
LICENSES = {'apache-2.0','mit','bsd-2-clause','bsd-3-clause','cc-by-4.0','cc-by-sa-4.0','cc0-1.0','gpl-3.0','gpl-2.0','unlicense','wtfpl'}

def clean(x):
    if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)): return [clean(v) for v in x]
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.bool_,)): return bool(x)
    if isinstance(x,(float,np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x,(Path,pd.Timestamp)): return str(x)
    return x

def js(path,obj): path.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csv(path,rows): pd.DataFrame(rows).to_csv(path,index=False,encoding='utf-8-sig')
def zscore(s): return logit(np.clip(np.asarray(s,float)/100,.001,.999))
def key(s): return re.sub('[^a-z0-9]','',str(s).split('/')[-1].lower().replace('meta-','').replace('-hf',''))
def family(s):
    s=str(s).lower()
    for f in ['pythia','qwen','llama','gemma','mistral','mixtral','phi','falcon','bloom','opt','gpt','yi','smollm','stablelm','mpt']:
        if f in s:return f
    return str(s).split('/')[0].lower()
def kind(s):
    s=str(s).lower()
    if 'pretrained' in s:return 'pretrained'
    if any(w in s for w in ['chat','fine-tuned']):return 'chat_finetuned'
    return 'other'
def load_module(path):
    spec=importlib.util.spec_from_file_location('q3_reused',path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def design(df, levels=None):
    if levels is None:levels=sorted(df.family.unique())
    X=np.column_stack([np.ones(len(df)),df.t,(df.kind=='chat_finetuned').astype(float)]+[(df.family==v).astype(float) for v in levels[1:]])
    return X,levels

def ridge(X,y,w=None,penalty=.2,positive_index=None):
    w=np.ones(len(y)) if w is None else np.asarray(w,float)
    reg=np.eye(X.shape[1])*np.sqrt(penalty);reg[0,0]=0
    lo=np.full(X.shape[1],-np.inf)
    if positive_index is not None:lo[positive_index]=0
    fit=lsq_linear(np.vstack([X*np.sqrt(w[:,None]),reg]),np.r_[np.asarray(y)*np.sqrt(w),np.zeros(X.shape[1])],bounds=(lo,np.full(X.shape[1],np.inf)),tol=1e-10)
    if not fit.success:raise RuntimeError('回归求解失败：'+fit.message)
    return fit.x

def group_weights(df):
    # 每个模型族总权重相同，降低一个模型族派生版本过多的影响。
    w=1/df.groupby('family').family.transform('size').to_numpy();return w/w.mean()
def loss_ref(df,params):
    return params.E+params.A*np.asarray(df.N_B)**(-params.alpha)+params.B*(np.asarray(df.D_B)/100)**(-params.beta)

def load_tables(root,out,asof):
    lb=pd.read_csv(root/'leaderboard_cleaned.csv');ep=pd.read_csv(root/'epoch_all_ai_models.csv')
    lb['date']=pd.to_datetime(lb['Submission Date'],errors='coerce')
    cutoff=pd.Timestamp(asof) if asof else lb.date.max()
    lb['S']=lb[TASKS].mean(axis=1);lb['key']=lb.Model.map(key);lb['family']=lb.Model.map(family);lb['kind']=lb.Type.map(kind)
    lb['N_B']=pd.to_numeric(lb['#Params (B)'],errors='coerce')
    ep['key']=ep.Model.map(key);ep['publication']=pd.to_datetime(ep['Publication date'],errors='coerce')
    for c in ['Training compute (FLOP)','Training dataset size (total)','Parameters']:
        ep[c]=pd.to_numeric(ep[c],errors='coerce')
    # 限制历史信息截止日期；元数据表后来的模型不进入本轮预测基线。
    ep=ep[ep.publication.notna() & (ep.publication<=cutoff)].copy()
    ep['completeness']=ep[['Training compute (FLOP)','Training dataset size (total)','Parameters']].notna().sum(axis=1)
    # 同一规范名多条记录选字段最完整者；不按得分择优。
    csv(out/'C4_duplicate_keys.csv',ep[ep.duplicated('key',keep=False)])
    ep=ep.sort_values(['completeness','publication'],ascending=[False,True]).drop_duplicates('key')
    cols=['key','Model','Publication date','Training compute (FLOP)','Training dataset size (total)','Parameters','Open model weights?','Confidence','Parameters notes']
    lb=lb.merge(ep[cols].rename(columns={'Model':'C4_Model'}),on='key',how='left',validate='many_to_one')
    lb['license_allowed']=lb['Hub License'].fillna('').str.lower().isin(LICENSES)
    lb['open_allowed']=lb.license_allowed | lb['Open model weights?'].eq('Yes')
    lb.loc[lb['Open model weights?'].eq('No'),'open_allowed']=False
    lb['D_B']=lb['Training dataset size (total)']/1e9
    lb['C_FLOPs']=lb['Training compute (FLOP)']
    lb['reason']='included'
    checks=[(~lb.open_allowed,'open_or_license_unverified'),(lb.kind.eq('other'),'merge_multimodal_or_other'),(lb.N_B.isna()|(lb.N_B<=0),'invalid_parameters'),(lb.date.isna()|(lb.date>cutoff),'missing_or_future_submission'),(~lb[TASKS].notna().all(axis=1),'missing_task'),(~lb[TASKS].ge(0).all(axis=1)|~lb[TASKS].le(100).all(axis=1),'score_outside_0_100')]
    for mask,reason in checks:lb.loc[mask,'reason']=reason
    csv(out/'C1_screening_audit.csv',lb)
    good=lb[lb.reason.eq('included')].sort_values('date').drop_duplicates('Model').copy()
    good['t']=(good.date-cutoff).dt.total_seconds()/(365.25*86400)
    # 明确标注 MoE 及总/激活参数歧义；不代入密集模型 Q2 标度式。
    moe=good.Model.str.contains(r'mixtral|moe|\d+x\d+|\d+B-A\d+',case=False,regex=True)|good['Parameters notes'].fillna('').str.contains('active|activated|MoE',case=False)
    ratio=good['Parameters']/1e9/good.N_B
    size_ok=ratio.between(.75,1.25)|ratio.isna()
    match=good[(good.D_B>0)&(good.C_FLOPs>0)&good['Open model weights?'].eq('Yes')&~moe&size_ok].copy()
    # 一个 C4 模型可能被多个仓库镜像；固定选最早提交，再按名称排序。
    match=match.sort_values(['date','Model']).drop_duplicates('key')
    match['submission_date']=match['date']
    match['date']=pd.to_datetime(match['Publication date'],errors='coerce')
    match['t']=(match.date-cutoff).dt.total_seconds()/(365.25*86400)
    match=match[match.date.notna()].copy()
    match['C_over_6ND']=match.C_FLOPs/(6*match.N_B*match.D_B*1e18)
    csv(out/'C4_matched_models.csv',match)
    historical=pd.read_csv(root/'leaderboard_extended_timeseries.csv')
    historical['six_task_mean']=historical[['IFEval','BBH','MATH_Lvl5','GPQA','MUSR','MMLU_PRO']].mean(axis=1)
    historical['average_gap']=historical.Average-historical.six_task_mean
    historical['already_in_C1']=historical.Model.isin(lb.Model)
    csv(out/'C3_source_year_summary.csv',historical.groupby(['Source','Year']).agg(n=('Model','size'),score_median=('Average','median'),score_q95=('Average',lambda x:x.quantile(.95)),max_abs_average_gap=('average_gap',lambda x:x.abs().max()),duplicate_C1=('already_in_C1','sum')).reset_index())
    csv(out/'C3_historical_audit.csv',historical[~historical.Source.eq('Open LLM Leaderboard')])
    # C3 来源内趋势仅作口径敏感性诊断，不把旧年份缺测的 0 分等同真实失败。
    trends=[]
    for source,g in historical.groupby('Source'):
        g=g[(g.Params_B>0)&g.Average.between(0,100)].dropna(subset=['Year'])
        X=np.column_stack([np.ones(len(g)),np.log(g.Params_B),g.Year-g.Year.min()])
        b=ridge(X,zscore(g.Average))
        trends.append({'source':source,'n':len(g),'year_logit_slope':b[2],'interpretation':'within-source descriptive sensitivity; not pooled with primary model'})
    csv(out/'C3_descriptive_trends.csv',trends)
    return good,match,ep,cutoff,{'C1_total':len(lb),'C1_included':len(good),'C4_matched_complete_dense':len(match),'C3_rows':len(historical),'C3_overlap_C1':int(historical.already_in_C1.sum()),'average_max_difference':float((lb.S-lb['Average ⬆️']).abs().max())}

def c8_analysis(root,out,cutoff,lb):
    leaves=[];audits=[];models=[];sources=[]
    for folder in sorted((root/'detailed_results').iterdir()):
        if not folder.is_dir():continue
        used=None
        for f in sorted(folder.glob('*.json'),reverse=True):
            d=pd.to_datetime(f.name[8:18],errors='coerce')
            if pd.notna(d) and d>cutoff:continue
            try:
                obj=json.loads(f.read_text(encoding='utf-8'));res=obj['results']
                if not isinstance(res,dict):raise ValueError('results 非字典')
                used=f;break
            except (ValueError,KeyError,UnicodeError) as ex:audits.append({'file':str(f),'reason':str(ex)[:100]})
        if used is None:continue
        sources.append({'file':str(used.relative_to(root)),'sha256':hashlib.sha256(used.read_bytes()).hexdigest()})
        name=obj.get('model_name',folder.name)
        if not isinstance(name,str):name=folder.name
        rec=[]
        for task,values in res.items():
            if not task.startswith('leaderboard_bbh_'):continue
            val=values.get('acc_norm,none')
            if not isinstance(val,(int,float)) or not 0<=val<=1:continue
            ns=obj.get('n-samples',{}).get(task,{})
            n=ns.get('effective',ns.get('original',1)) if isinstance(ns,dict) else 1
            n=float(n) if isinstance(n,(int,float)) and n>0 else 1.
            row={'Model':name,'directory':folder.name,'task':task,'raw_accuracy':val,'n':n,'json':used.name};rec.append(row);leaves.append(row)
        if rec:
            models.append({'Model':name,'directory':folder.name,'n_subtasks':len(rec),'BBH_raw_macro':100*np.mean([r['raw_accuracy'] for r in rec]),'BBH_raw_micro':100*np.average([r['raw_accuracy'] for r in rec],weights=[r['n'] for r in rec])})
    if not leaves:raise ValueError('C8 未找到任何 BBH 子任务；不可用汇总表代替')
    leaf=pd.DataFrame(leaves);mod=pd.DataFrame(models)
    csv(out/'C8_BBH_subtasks.csv',leaf);csv(out/'C8_model_aggregation.csv',mod);csv(out/'C8_skipped_json.csv',audits)
    csv(out/'C8_task_summary.csv',leaf.groupby('task').agg(models=('Model','size'),mean_raw_accuracy=('raw_accuracy','mean'),std_raw_accuracy=('raw_accuracy','std')).reset_index())
    lb=lb.copy();lb['directory']=lb.Model.str.replace('/','_',regex=False)
    check=mod.merge(lb[['directory','BBH','kind']],on='directory',how='inner')
    csv(out/'C8_C1_BBH_comparison.csv',check)
    csv(out/'C8_source_manifest.csv',sources)
    return {'C8_models':len(mod),'C8_leaf_rows':len(leaf),'C8_bad_json_attempts':len(audits),'C8_C1_matched':len(check),'C8_raw_vs_C1_spearman':check.BBH_raw_macro.corr(check.BBH,method='spearman'),'C8_note':'原始 acc_norm 子任务汇总与榜单变换后分数不直接相减；比较覆盖与排序'}

def bridge_frame(root,lb):
    b=pd.read_csv(root/'loss_benchmark_bridge_expanded.csv')
    # C6 已包含 C5，不叠加重复样本；必须尊重模型身份和截止日期。
    b=b.merge(lb[['Model','kind','family','date']],on='Model',how='inner',validate='one_to_one')
    b=b.dropna(subset=['Val_Loss']+BTASKS).drop_duplicates('Model')
    b['high']=b.Loss_Comparability.str.startswith('High');return b

def fit_bridge(b,mode):
    d=b[b.high].copy() if mode=='high' else b.copy()
    if len(d)<4:raise ValueError('可比桥接样本不足 4 条')
    families=sorted(d.family.unique(),key=lambda v:(v!='pythia',v)) if mode=='stratified' else []
    X=np.column_stack([np.ones(len(d)),-d.Val_Loss,(d.kind=='chat_finetuned').astype(float)]+[(d.family==f).astype(float) for f in families[1:]])
    w=np.where(d.high,1,.25);coef=[]
    for c in BTASKS:coef.append(ridge(X,zscore(d[c]),w,penalty=.2,positive_index=1))
    return {'coef':np.asarray(coef),'families':families,'mode':mode,'n':len(d),'range':[float(d.Val_Loss.min()),float(d.Val_Loss.max())]}

def bridge_score(m,loss):
    # 在 Pythia/基础模型参考截距上返回尺度一致的六维宏平均；族截距只用于校准。
    loss=np.atleast_1d(loss);b=m['coef'];return 100*expit(b[:,0,None]-b[:,1,None]*loss).mean(axis=0)

def bridge_logit(m,loss):return zscore(bridge_score(m,loss))

def bridge_validation(b,out):
    rows=[]
    for mode in ['high','stratified']:
        d=b[b.high] if mode=='high' else b
        groups=d.Model.unique() if mode=='high' else d.family.unique()
        for g in groups:
            is_test=d.Model.eq(g) if mode=='high' else d.family.eq(g)
            train=d[~is_test];test=d[is_test]
            if len(train)<4:continue
            m=fit_bridge(train,mode)
            X=np.column_stack([np.ones(len(test)),-test.Val_Loss,(test.kind=='chat_finetuned').astype(float)]+[(test.family==f).astype(float) for f in m['families'][1:]])
            pred=100*expit(X@m['coef'].T)
            for i,(_,r) in enumerate(test.iterrows()):
                rows.append({'mode':mode,'held_out':g,'Model':r.Model,'observed':r[BTASKS].mean(),'predicted':pred[i].mean(),'naive_train_mean':train[BTASKS].mean(axis=1).mean(),'high':r.high})
    rows=pd.DataFrame(rows);csv(out/'bridge_heldout_predictions.csv',rows)
    metric=[]
    for mode,d in rows.groupby('mode'):
        metric.append({'mode':mode,'n':len(d),'MAE_points':np.mean(abs(d.predicted-d.observed)),'RMSE_points':np.sqrt(np.mean((d.predicted-d.observed)**2)),'naive_MAE_points':np.mean(abs(d.naive_train_mean-d.observed))})
    csv(out/'bridge_validation.csv',metric);return metric

def fit_dynamics(df,bm):
    X,levels=design(df);y=zscore(df.S)-bridge_logit(bm,df.L_ref)
    beta=ridge(X,y,group_weights(df));return {'beta':beta,'levels':levels,'condition_number':np.linalg.cond(X)}

def residual_predict(df,m):return design(df,m['levels'])[0]@m['beta']

def decompose(df,bm,dm):
    lo,hi=df.t.quantile([.25,.75]);early=df[df.t<=lo];late=df[df.t>=hi]
    w=group_weights(df);L0=float(early.L_ref.mean());L1=float(late.L_ref.mean());t0=float(early.t.mean());t1=float(late.t.mean())
    tmp=df.copy();tmp['t']=t0;base=residual_predict(tmp,dm)
    s0=bridge_logit(bm,[L0])[0];s1=bridge_logit(bm,[L1])[0];dt=dm['beta'][1]*(t1-t0)
    f=lambda s,t:float(np.average(100*expit(base+s+t),weights=w))
    scale=.5*((f(s1,0)-f(s0,0))+(f(s1,dt)-f(s0,dt)))
    tech=.5*((f(s0,dt)-f(s0,0))+(f(s1,dt)-f(s1,0)));total=scale+tech
    return {'scale_points':scale,'nonscale_time_points':tech,'total_modeled_points':total,'scale_share':scale/total if total>1e-6 else np.nan,'nonscale_share':tech/total if total>1e-6 else np.nan,'positive_total_change':total>1e-6,'start_t':t0,'end_t':t1,'start_loss':L0,'end_loss':L1,'interpretation':'fixed family/type composition, conditional Shapley decomposition; not identified causal effects'}

def backtest(df,bm,out):
    rows=[];preds=[]
    for q in [.6,.75,.85]:
        cut=df.date.quantile(q);tr=df[df.date<=cut];te=df[df.date>cut]
        if len(tr)<20 or len(te)<4:continue
        m=fit_dynamics(tr,bm);pred=100*expit(bridge_logit(bm,te.L_ref)+residual_predict(te,m))
        naive=np.repeat(tr.S.mean(),len(te))
        rows.append({'train_end':str(cut),'test_n':len(te),'MAE':np.mean(abs(pred-te.S)),'RMSE':np.sqrt(np.mean((pred-te.S)**2)),'naive_MAE':np.mean(abs(naive-te.S)),'scope':'rolling dynamics validation; bridge fixed from cross-sectional C6, not full vintage backtest'})
        preds.extend({'Model':n,'predicted':p,'observed':s,'cut':str(cut)} for n,p,s in zip(te.Model,pred,te.S))
    csv(out/'temporal_validation.csv',rows);csv(out/'temporal_predictions.csv',preds)
    return rows

def bootstrap_sample(df,rng):
    groups=df.family.unique();picked=rng.choice(groups,len(groups),replace=True)
    parts=[]
    for i,g in enumerate(picked):
        d=df[df.family==g].copy();d['family']=f'{g}__draw{i}';parts.append(d)
    return pd.concat(parts,ignore_index=True)

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    candidates=[BASE.parent/'real_attachments'/'C_efficiency_evolution',Path.home()/'数模建模方法/中文题目/F题/real_attachments/C_efficiency_evolution']
    p.add_argument('--data-root',type=Path,default=next((x for x in candidates if x.is_dir()),candidates[0]))
    p.add_argument('--problem2-output',type=Path,default=BASE/'upstream_runs/Q2')
    p.add_argument('--problem3-code',type=Path,default=BASE/'upstream/Q3/problem3_solution.py')
    p.add_argument('--output-dir',type=Path,default=BASE/'problem4_outputs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    p.add_argument('--as-of',default=None,help='默认 C1 最大提交日期，不使用计算机今日日期')
    p.add_argument('--bootstrap',type=int,default=200)
    p.add_argument('--seed',type=int,default=20260926)
    p.add_argument('--annual-growth',nargs='+',type=float,default=[0,.1,.25])
    p.add_argument('--no-plots',action='store_true')
    return p.parse_args()

def main(a):
    if a.bootstrap<0 or any(g<0 for g in a.annual_growth):raise ValueError('bootstrap 和算力增长率不得为负')
    required=[a.problem3_code,a.problem2_output/'generalized_model.json']+[a.data_root/f for f in ['leaderboard_cleaned.csv','leaderboard_extended_timeseries.csv','epoch_all_ai_models.csv','loss_benchmark_bridge_expanded.csv','model_architecture_metadata.csv','detailed_results']]
    missing=[str(f) for f in required if not f.exists()]
    if missing:raise FileNotFoundError('缺少输入，请检查 --data-root / --problem2-output / --problem3-code：\n'+'\n'.join(missing))
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    root=a.data_root.resolve();q3=load_module(a.problem3_code.resolve());model,params,bounds=q3.load_model(a.problem2_output.resolve())
    print('[1/6] C1/C3/C4 数据审计与严格匹配',flush=True)
    lb,matched,ep,cutoff,audit=load_tables(root,out,a.as_of)
    if len(matched)<20:raise ValueError(f'C4 完整可匹配记录只有 {len(matched)}，不足以支持本配置的贡献分解')
    matched['L_ref']=loss_ref(matched,params)
    matched['Q2_extrapolation']=~matched.N_B.between(bounds.b1_n_min/1e9,bounds.b1_n_max/1e9)|~matched.D_B.between(bounds.b1_d_min/1e9,bounds.b1_d_max/1e9)
    csv(out/'analysis_panel.csv',matched)
    print('[2/6] C8 逐文件读取与 BBH 子任务聚合',flush=True)
    audit.update(c8_analysis(root,out,cutoff,lb))
    print('[3/6] C6 可比性分层桥接与留出检验',flush=True)
    b=bridge_frame(root,lb);csv(out/'bridge_used.csv',b);bm=fit_bridge(b,'high');sm=fit_bridge(b,'stratified')
    vm=bridge_validation(b,out);js(out/'bridge_models.json',{'primary':bm,'sensitivity':sm})
    print('[4/6] 条件动力学、贡献分解与时间留出',flush=True)
    dm=fit_dynamics(matched,bm);dep=decompose(matched,bm,dm);valid=backtest(matched,bm,out)
    # C1 全样本参数量代理模型为补充，不能宣称已经控制训练数据量。
    X,lev=design(lb);X=np.column_stack([X,np.log(lb.N_B)])
    nb=ridge(X,zscore(lb.S),group_weights(lb),positive_index=X.shape[1]-1)
    csv(out/'C1_parameter_proxy_sensitivity.csv',[{'n':len(lb),'time_logit_slope':nb[1],'logN_coefficient':nb[-1],'note':'D,Q,p unknown; descriptive parameter-only sensitivity'}])
    print('[5/6] 调用原 Q3 求解器，建立未来算力情景',flush=True)
    lang=ep[ep.Domain.fillna('').str.contains('Language') & ep['Open model weights?'].eq('Yes') & (ep['Training compute (FLOP)']>0)]
    recent=lang[lang.publication>=cutoff-pd.DateOffset(months=12)]
    if len(recent)<5:raise ValueError('截止日期前一年 C4 开放语言模型算力不足 5 条')
    C0=float(recent['Training compute (FLOP)'].quantile(.95));csv(out/'C4_budget_reference.csv',recent)
    hist=lang[lang.publication>=cutoff-pd.DateOffset(years=3)].copy()
    hist['quarter']=hist.publication.dt.to_period('Q')
    quarters=hist.groupby('quarter')['Training compute (FLOP)'].agg(n='size',q95=lambda s:s.quantile(.95)).reset_index()
    quarters['date']=quarters.quarter.dt.to_timestamp()
    usable=quarters[quarters.n>=3].copy()
    historic_growth=None
    if len(usable)>=4:
        t=(usable.date-usable.date.min()).dt.days.to_numpy()/365.25
        slope=np.linalg.lstsq(np.column_stack([np.ones(len(t)),t]),np.log(usable.q95),rcond=None)[0][1]
        historic_growth=float(np.expm1(slope))
    csv(out/'C4_historical_compute_quarters.csv',quarters)
    _,contexts=q3.load_context_lengths(root/'model_architecture_metadata.csv')
    q0=float(model['mixture_bridge']['Q_reference'])
    library=pd.DataFrame([{'q_base':q0,'mixture_r':0.,'allocation_policy':'reference','mix_id':'reference_p0','dataset':'reference','index':-1}])
    alloc=[]
    for cost in q3.COST_MODELS:
        for ctx in contexts:
            for growth in a.annual_growth:
                for h in [0,12,24]:
                    r=q3.optimize_scenario(library,'reference',C0*(1+growth)**(h/12),ctx,cost,params,bounds)
                    r.update(annual_growth=growth,horizon_months=h);alloc.append(r)
    alloc=pd.DataFrame(alloc);csv(out/'Q3_future_allocations.csv',alloc)
    anchor=lb[lb.date>=cutoff-pd.DateOffset(months=3)].groupby('kind').S.quantile(.95).to_dict()
    if not anchor:raise ValueError('最近三个月无可用于前沿锚点的模型')
    def forecast_rows(bridge,dynamic,allocations,anchor_values):
        rows=[]
        for (cost,ctx,g),part in allocations.groupby(['cost_model','context_length','annual_growth']):
            l0=float(part.loc[part.horizon_months.eq(0),'predicted_loss'].iloc[0])
            for _,r in part[part.horizon_months>0].iterrows():
                dz=bridge_logit(bridge,[r.predicted_loss])[0]-bridge_logit(bridge,[l0])[0]
                for typ,s0 in anchor_values.items():
                    for trend in [0.,.5,1.]:
                        pred=float(100*expit(zscore([s0])[0]+dz+trend*dynamic['beta'][1]*r.horizon_months/12))
                        rows.append({'cost_model':cost,'context_length':ctx,'annual_growth':g,'horizon_months':int(r.horizon_months),'kind':typ,'technology_retention':trend,'anchor_score_q95':s0,'prediction':pred,'scale_logit_change':dz,'tech_logit_change':trend*dynamic['beta'][1]*r.horizon_months/12,'predicted_loss':r.predicted_loss,'bridge_outside_high_support':not bridge['range'][0]<=r.predicted_loss<=bridge['range'][1],'Q3_evidence_flag':r.evidence_flag,'forecast_date':str((cutoff+pd.DateOffset(months=int(r.horizon_months))).date())})
        return pd.DataFrame(rows)
    forecast=forecast_rows(bm,dm,alloc,anchor)
    sensitivity=forecast_rows(sm,fit_dynamics(matched,sm),alloc,anchor);csv(out/'medium_bridge_sensitivity.csv',sensitivity)
    print('[6/6] 模型族重采样、桥接误差与结果输出',flush=True)
    rng=np.random.default_rng(a.seed);boot=[];draws=[];failures=[]
    high=b[b.high].copy()
    # 上游 Bootstrap 可用时传播参数抽样；否则明确标记条件于点估计。
    upstream_path=a.problem2_output/'bootstrap_draws.csv';up=pd.read_csv(upstream_path) if upstream_path.exists() else pd.DataFrame()
    for k in range(a.bootstrap):
        try:
            bs=high.iloc[rng.integers(0,len(high),len(high))].copy();mb=fit_bridge(bs,'high')
            ds=bootstrap_sample(matched,rng)
            pp=params
            if len(up) and all(c in up for c in ['E','A','B','alpha','beta','gamma']):
                u=up.iloc[int(rng.integers(len(up)))];pp=replace(params,**{c:float(u[c]) for c in ['E','A','B','alpha','beta']},gamma_quality=float(u.gamma))
            ds['L_ref']=loss_ref(ds,pp)
            md=fit_dynamics(ds,mb);dd=decompose(ds,mb,md);dd['draw']=k;draws.append(dd)
            # 不把拟合噪声当作独立的真实模型观测；仅重采样前沿锚点总体。
            al={}
            for typ in anchor:
                v=lb.loc[(lb.kind==typ)&(lb.date>=cutoff-pd.DateOffset(months=3))];al[typ]=float(bootstrap_sample(v,rng).S.quantile(.95))
            ba=alloc.copy()
            if len(up) and all(c in up for c in ['E','A','B','alpha','beta','gamma']):
                # 固定本次 Q3 最优配置，只传播上游预测参数；不是每次重新优化。
                ba['predicted_loss']=q3.generalized_loss(pp,ba.N_parameters,ba.D_tokens,ba.Q_opt,q0,0.)
            f=forecast_rows(mb,md,ba,al);boot.append(f.prediction.to_numpy())
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as ex:failures.append({'draw':k,'error':str(ex)})
        if (k+1)%50==0:print(f'  Bootstrap {k+1}/{a.bootstrap}',flush=True)
    if boot:
        arr=np.array(boot);forecast['conditional_lower_95'],forecast['conditional_upper_95']=np.quantile(arr,[.025,.975],axis=0)
    if a.bootstrap and len(boot)<.8*a.bootstrap:
        csv(out/'bootstrap_failures.csv',failures)
        raise RuntimeError(f'Bootstrap 仅成功 {len(boot)}/{a.bootstrap}，不输出误导性的完整区间')
    csv(out/'frontier_forecasts.csv',forecast);csv(out/'contribution_bootstrap.csv',draws);csv(out/'bootstrap_failures.csv',failures)
    if draws:
        bd=pd.DataFrame(draws)
        dep['bootstrap_positive_total_fraction']=float(bd.positive_total_change.mean())
        dep['intervals']={c:list(bd[c].quantile([.025,.975])) for c in ['scale_points','nonscale_time_points','scale_share','nonscale_share']}
    js(out/'contribution_decomposition.json',dep)
    hashes={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in [a.problem3_code,a.problem2_output/'generalized_model.json',root/'leaderboard_cleaned.csv',root/'leaderboard_extended_timeseries.csv',root/'epoch_all_ai_models.csv',root/'loss_benchmark_bridge_expanded.csv']}
    summary={'as_of':str(cutoff.date()),'forecast_origin':'C1 maximum submission date unless --as-of supplied','audit':audit,'C0_FLOPs':C0,'primary_bridge_n':bm['n'],'bridge_validation':vm,'dynamics_annual_logit_trend':dm['beta'][1],'dynamics_condition_number':dm['condition_number'],'decomposition':dep,'temporal_validation':valid,'bootstrap_requested':a.bootstrap,'bootstrap_success':len(boot),'Q2_bootstrap_available':len(up),'upstream_uncertainty':'Q2 parameter draws at fixed optimal configurations; Q0/mapping/quality mechanism and optimizer re-selection excluded','source_hashes':hashes,'limitations':['non-scale time residual includes omitted factors and selection; not causal identification','C3 historical scores not pooled because measurement differs and zeros may be missing','95th percentile operational frontier, not absolute attainable maximum','12/24-month extrapolation exceeds C1 observation span','high bridge only Pythia; medium bridge stratified sensitivity','pretraining budget proxy excludes unknown post-training compute','primary matched dynamics uses C4 publication dates; C1 frontier anchor and broad sensitivity use submission dates','retrospective benchmark evaluation is not a real-time historical leaderboard']}
    summary.update(historical_annual_compute_growth=historic_growth,compute_growth_scenarios=a.annual_growth,scenario_below_historical_growth={str(g):g<historic_growth if historic_growth is not None else None for g in a.annual_growth},dynamics_coefficients={'names':['intercept','annual_time','chat_finetuned']+['family_'+f for f in dm['levels'][1:]],'values':dm['beta']},Q_reference=q0,Q2_extrapolation_count=int(matched.Q2_extrapolation.sum()))
    js(out/'analysis_summary.json',summary)
    if not a.no_plots:plots(out,lb,b,bm,forecast,cutoff)
    (out/'results_summary.md').write_text('# Q4 本次运行结果\n\n'+f'数据截止：{cutoff.date()}；主分析匹配样本 {len(matched)}；高可比桥接 {bm["n"]}。\n\n'+f'条件分解：规模 {dep["scale_points"]:.4f} 分，非规模时间项 {dep["nonscale_time_points"]:.4f} 分。\n\n'+f'Bootstrap 完成 {len(boot)}/{a.bootstrap} 次。预测详见 frontier_forecasts.csv。\n\n本结果是模型与数据口径条件下的情景分析，不是已识别因果贡献，也不是对真实最高分的保证。\n',encoding='utf-8')
    print(json.dumps(clean({'output':out,'audit':audit,'decomposition':dep,'bootstrap_success':len(boot)}),ensure_ascii=False,indent=2))

def plots(out,lb,b,bm,fore,cutoff):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'figures';folder.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,4.5));x=np.linspace(b.Val_Loss.min(),b.Val_Loss.max(),150)
    for flag,color in [(True,'#28766f'),(False,'#b6904c')]:
        g=b[b.high==flag];ax.scatter(g.Val_Loss,g[BTASKS].mean(axis=1),s=22,c=color,label='High comparability' if flag else 'Medium comparability',alpha=.65)
    ax.plot(x,bridge_score(bm,x),c='#343c46',label='High-only reference mapping');ax.set(xlabel='Validation loss (source-dependent)',ylabel='Six-task mean score');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(folder/'q4_fig01_loss_benchmark_bridge.pdf',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for typ,g in lb.groupby('kind'):
        trend=g.set_index('date').S.resample('MS').quantile(.95);ax.plot(trend.index,trend.values,'o-',label=typ)
    ax.set(ylabel='Monthly 95th-percentile score',xlabel='Submission date');ax.legend();fig.autofmt_xdate();fig.tight_layout();fig.savefig(folder/'q4_fig02_historical_frontier.pdf',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,(typ,g) in zip(axes,fore.groupby('kind')):
        g=g[(g.context_length==8192)&(g.cost_model=='exponential')&(g.technology_retention==.5)]
        for growth,d in g.groupby('annual_growth'):
            d=d.sort_values('horizon_months');ax.plot(d.horizon_months,d.prediction,'o-',label=f'Compute +{growth:.0%}/year')
            if 'conditional_lower_95' in d:ax.fill_between(d.horizon_months,d.conditional_lower_95,d.conditional_upper_95,alpha=.12)
        ax.set(title=typ,xlabel='Months after '+str(cutoff.date()),ylabel='Conditional frontier score',ylim=(0,100));ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(folder/'q4_fig03_frontier_scenarios.pdf',bbox_inches='tight');plt.close(fig)

if __name__=='__main__':
    try:main(parse_args())
    except (FileNotFoundError,ValueError,RuntimeError) as exc:
        print(f'运行终止：{exc}',file=sys.stderr);sys.exit(2)
