#!/usr/bin/env python3
"""Build the executed, mechanism-oriented analysis notebook."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
N = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md("""# Why do LLaDA and Dream diverge? A mechanism-oriented analysis

**Research question.** How much of the apparent AR-versus-diffusion difference on bias and memorization tests is explained by (i) abstention under missing evidence, (ii) generation coverage and length, and (iii) the model-specific alignment of masked-token scores?

The baseline runs cover the complete public BBQ, BOLD, CrowS-Pairs, and StereoSet extracts. Figures below use the original item-level predictions. Newly submitted intervention runs are included automatically once their output files are complete. This is exploratory analysis of four specific checkpoints, not a causal claim that generation architecture alone causes the differences.

**Critical score correction.** Dream predicts token *i* from logit position *i−1*. Its official generation code shifts logits before sampling. Our first CrowS-Pairs, StereoSet, MIA, Goldfish, and utility scorers did not apply this shift. Their Dream values in the older `final_comparison.ipynb` are invalid. This notebook excludes them until corrected Dream jobs finish. Dream BBQ and BOLD generation, plus all Llama/Qwen/LLaDA results, remain valid. See [`docs/scoring_erratum.md`](../docs/scoring_erratum.md).""")

md("""## Mechanistic differences to test

| Feature | LLaDA-8B-Instruct | Dream-v0-Instruct-7B | Why it matters here |
|:--|:--|:--|:--|
| Initial training | From-scratch masked diffusion | Initialized from Qwen2.5 AR weights, then diffusion training | Architecture and training history are coupled; paired model differences cannot isolate one cause. |
| Masked-token head | Predicts token at its own position | Predicts token at the previous position's logit | Scoring and fine-tuning targets must align to each model. |
| Current generation adapter | Highest-confidence masked positions filled each round | Official `diffusion_generate` with entropy ranking | Decoding can affect EOS placement, answer length, and prompt compliance. |
| Paper-reported training | 2.3T pretraining tokens; 4.5M SFT pairs | 0.6T diffusion training tokens; 1.8M SFT pairs | Different instruction exposure is a plausible contributor, not a proven cause. |

The paper-reported token counts come from different training histories and are not directly matched budgets. Sources: [LLaDA paper](https://arxiv.org/html/2502.09992v3), [Dream paper](https://arxiv.org/html/2508.15487v1), and [Dream's checkpoint generation code](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B/tree/main). The LLaDA paper explicitly discusses EOS-heavy short generation under pure diffusion sampling. Our EOS intervention tests whether this mechanism matters for our raw-prefix BOLD setting.""")

code("""
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = Path.cwd()
if not (ROOT/'data/processed').exists():
    ROOT = ROOT.parent
assert (ROOT/'data/processed').exists(), 'Run from repository root or notebooks/'
FIG = ROOT / 'results/figures'
FIG.mkdir(parents=True, exist_ok=True)
models = ['llama','llada','qwen','dream']
names = {'llama':'Llama 3','llada':'LLaDA','qwen':'Qwen2.5','dream':'Dream'}
colors = {'llama':'#4477AA','llada':'#66CCEE','qwen':'#228833','dream':'#EE6677'}
plt.rcParams.update({'figure.dpi':130,'savefig.dpi':160,'font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False})

def read_jsonl(path):
    with open(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

def finish(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f'divergence_{name}.png', bbox_inches='tight')
    plt.show()
""")

md("""## 1. BBQ: Is the difference about evidence or group stereotypes?

BBQ has 58,492 items, split evenly between ambiguous and disambiguated contexts. Every model saw the same items. Ambiguous accuracy means choosing the unknown answer. Disambiguated accuracy means using the provided evidence. We inspect the item labels and answer positions so a simple option-position habit is visible.""")

code("""
source = {}
for r in read_jsonl(ROOT/'data/processed/bbq_full.jsonl'):
    unknown = [i for i in range(3) if r['answer_info'][f'ans{i}'][1].lower()=='unknown']
    if len(unknown)!=1: raise ValueError(f"unknown mapping: {r['id']}")
    source[r['id']] = (r['category'],r['context_condition'],r['question_polarity'],r['label'],unknown[0])
records=[]
for model in models:
    seen=set()
    for r in read_jsonl(ROOT/f'results/bbq_bbq_full_{model}.jsonl'):
        if r['id'] in seen: raise ValueError('duplicate BBQ id')
        seen.add(r['id'])
        cat,condition,polarity,label,unknown=source[r['id']]
        records.append((r['id'],model,cat,condition,polarity,label,r['prediction'],
                        r['prediction']==label,r['prediction']==unknown,r['prediction'] is None))
    if seen!=set(source): raise ValueError(f'incomplete BBQ: {model}')
bbq=pd.DataFrame.from_records(records,columns=['id','model','category','condition','polarity',
                                                'label','prediction','correct','unknown','invalid'])
display(bbq.groupby(['model','condition'])[['correct','unknown','invalid']].mean().round(4))
""")

code("""
fig,axes=plt.subplots(1,2,figsize=(10,3.6),sharey=True)
for ax,metric,title in zip(axes,['correct','unknown'],['Accuracy','Unknown-answer rate']):
    tab=bbq.groupby(['model','condition'])[metric].mean().unstack().loc[models]
    x=np.arange(len(models)); w=.35
    for j,cond in enumerate(['ambig','disambig']):
        ax.bar(x+(j-.5)*w,tab[cond].to_numpy(),w,label=cond,
               color=['#DDCC77','#AA4499'][j])
    ax.set_xticks(x,[names[m] for m in models]); ax.set_ylim(0,1); ax.set_title(title)
    ax.grid(axis='y',alpha=.2); ax.legend(frameon=False)
finish(fig,'bbq_evidence_vs_abstention')
""")

code("""
# Decompose the authors' ambiguous BBQ bias score into guessing frequency and
# stereotype direction among the model's non-unknown, mapped responses.
bbq_scores=pd.read_csv(ROOT/'results/summary/bbq_full.tsv',sep='\\t')
amb_scores=bbq_scores[(bbq_scores.category=='ALL')&
                      (bbq_scores.condition=='ambig')].copy()
amb_scores['model']=amb_scores.file.str.extract(r'bbq_full_(\\w+)\\.jsonl$')[0]
amb_scores=amb_scores[amb_scores.model.isin(models)].set_index('model').loc[models]
amb_scores['guess_fraction']=1-amb_scores.unknown_rate-amb_scores.invalid_rate
amb_scores['target_given_guess']=(1+amb_scores.stereotype_preference)/2
display(amb_scores[['accuracy','guess_fraction','target_given_guess','bias_score']].round(4))
fig,ax=plt.subplots(figsize=(8.5,4.4))
for model in models:
    r=amb_scores.loc[model]
    ax.scatter(r.guess_fraction,r.target_given_guess,s=115,color=colors[model])
    offset={'llama':(8,-18),'llada':(8,7),'qwen':(8,4),'dream':(8,4)}[model]
    ax.annotate(f"{names[model]}  (bias {r.bias_score:.3f})",
                (r.guess_fraction,r.target_given_guess),xytext=offset,textcoords='offset points')
ax.axhline(.5,color='black',ls='--',lw=.8)
ax.set_xlim(0,.55);ax.set_ylim(.45,.9)
ax.set_xlabel('Fraction of ambiguous questions answered with a person')
ax.set_ylabel('Stereotype-target share among non-unknown answers')
ax.grid(alpha=.2)
finish(fig,'bbq_bias_decomposition')
""")

code("""
cat=bbq.groupby(['model','condition','category'])['correct'].mean()
categories=sorted(bbq.category.unique())
pair_names=[('llada','llama','LLaDA − Llama 3'),('dream','qwen','Dream − Qwen2.5')]
fig,axes=plt.subplots(1,2,figsize=(12.5,6.2),sharey=True,layout='constrained')
for ax,(d,a,title) in zip(axes,pair_names):
    values=np.array([[cat[d,cond,c]-cat[a,cond,c] for cond in ['ambig','disambig']]
                     for c in categories])
    im=ax.imshow(values,cmap='RdBu',vmin=-.6,vmax=.6,aspect='auto')
    ax.set_title(title+' accuracy difference'); ax.set_xticks([0,1],['Ambiguous','Disambiguated'])
    ax.set_yticks(np.arange(len(categories)),[c.replace('_',' ') for c in categories])
    for i in range(len(categories)):
        for j in range(2):
            ax.text(j,i,f'{values[i,j]*100:+.1f}',ha='center',va='center',fontsize=8,
                    color='white' if abs(values[i,j])>.27 else 'black')
fig.colorbar(im,ax=axes.ravel().tolist(),label='Difference in accuracy',shrink=.75,pad=.02)
fig.savefig(FIG/'divergence_bbq_category_differences.png',bbox_inches='tight')
plt.show()
""")

code("""
# Gap interaction = (dLLM − AR accuracy on ambiguous items) minus
#                   (dLLM − AR accuracy on disambiguated items).
# Large negative values mean the dLLM loses more ground when evidence is absent.
interaction=[]
for d,a,title in pair_names:
    for c in categories:
        amb=cat[d,'ambig',c]-cat[a,'ambig',c]
        dis=cat[d,'disambig',c]-cat[a,'disambig',c]
        interaction.append((title,c,amb-dis))
interactions=pd.DataFrame(interaction,columns=['pair','category','ambiguity_interaction'])
display(interactions.pivot(index='category',columns='pair',values='ambiguity_interaction').round(3))
fig,axes=plt.subplots(1,2,figsize=(11,5.4),sharey=True,layout='constrained')
for ax,(d,a,title) in zip(axes,pair_names):
    sub=interactions[interactions.pair==title].set_index('category').loc[categories]
    values=sub.ambiguity_interaction.to_numpy()
    ax.barh(np.arange(len(categories)),values,
            color=np.where(values>=0,'#4477AA','#CC6677'))
    ax.axvline(0,color='black',lw=.8);ax.set_title(title)
    ax.set_yticks(np.arange(len(categories)),[c.replace('_',' ') for c in categories])
    ax.set_xlabel('Ambiguous gap minus disambiguated gap')
    ax.grid(axis='x',alpha=.2)
finish(fig,'bbq_ambiguity_interaction')
""")

code("""
pol=bbq.groupby(['model','condition','polarity'])['correct'].mean()
fig,axes=plt.subplots(1,2,figsize=(10,3.6),sharey=True)
for ax,cond in zip(axes,['ambig','disambig']):
    x=np.arange(len(models)); w=.35
    for j,p in enumerate(['neg','nonneg']):
        values=[pol[m,cond,p] for m in models]
        ax.bar(x+(j-.5)*w,values,w,label=p,color=['#4477AA','#EEAA55'][j])
    ax.set_xticks(x,[names[m] for m in models]); ax.set_ylim(0,1)
    ax.set_title(cond+' accuracy by question polarity'); ax.grid(axis='y',alpha=.2)
    ax.legend(frameon=False)
finish(fig,'bbq_polarity')
""")

code("""
# On ambiguous items the correct option is always the unknown option.
# Check whether one model merely favors a particular answer letter.
amb=bbq[bbq.condition=='ambig']
by_position=amb.groupby(['model','label'])['correct'].mean().unstack().loc[models]
display(by_position.rename(columns={0:'A',1:'B',2:'C'}).round(4))
fig,ax=plt.subplots(figsize=(8.5,3.5)); x=np.arange(3); w=.2
for j,model in enumerate(models):
    ax.bar(x+(j-1.5)*w,by_position.loc[model],w,label=names[model],color=colors[model])
ax.set_xticks(x,['A','B','C']);ax.set_xlabel('Position of the correct unknown answer')
ax.set_ylabel('Ambiguous accuracy');ax.set_ylim(0,1);ax.legend(frameon=False)
ax.grid(axis='y',alpha=.2)
finish(fig,'bbq_answer_position')
""")

code("""
pivot=bbq.pivot(index='id',columns='model',values='correct')
condition=pd.Series({i:v[1] for i,v in source.items()})
rows=[]
for d,a,title in pair_names:
    for cond in ['ambig','disambig']:
        ix=condition.index[condition==cond]
        wins=int((pivot.loc[ix,d] & ~pivot.loc[ix,a]).sum())
        losses=int((~pivot.loc[ix,d] & pivot.loc[ix,a]).sum())
        rows.append((title,cond,wins,losses,wins-losses))
discord=pd.DataFrame(rows,columns=['pair','condition','dLLM_only_correct','AR_only_correct','net'])
display(discord)
fig,ax=plt.subplots(figsize=(9,3.6))
x=np.arange(len(discord)); ax.barh(x,discord.dLLM_only_correct,color='#66CCEE',label='dLLM only correct')
ax.barh(x,-discord.AR_only_correct,color='#EE6677',label='AR only correct')
ax.set_yticks(x,[f'{r.pair}: {r.condition}' for r in discord.itertuples()]);
ax.axvline(0,color='black',lw=.8); ax.set_xlabel('Discordant item count');ax.legend(frameon=False)
finish(fig,'bbq_paired_discordance')
""")

md("""**Reading the BBQ plots.** Dream's low overall ambiguous score is spread across categories and is especially poor on nonnegative questions. Its disambiguated performance is high. This points to an *uncertainty-handling* hypothesis: Dream often chooses a person when evidence is insufficient. The explicit-abstention prompt below directly tests that hypothesis; category and polarity plots alone cannot prove it.""")

md("""## 2. BOLD: separate sentiment from generation coverage

All sentiment plots use the same VADER lexicon. The word-count and empty-output plots use the model outputs themselves. “No letter or digit” catches punctuation-only and invisible-character output missed by a whitespace check; it is a crude coverage diagnostic, not a semantic quality judgment. Negative sentiment is a lexical proxy; it cannot establish that a continuation is socially biased. A score based on the nonempty subset can move simply because a different set of prompts produced text.""")

code("""
bold_records=[]
for model in models:
    for mode,prefix in [('raw','bold_full'),('chat','bold_chat_full')]:
        for r in read_jsonl(ROOT/f'results/{prefix}_{model}.jsonl'):
            if r['prompt'].strip():
                bold_records.append((r['id'],model,mode,r['domain'],
                                     f"{r['source_file']}:{r['source_row']}",
                                     len(r['continuation'].split()),
                                     not any(ch.isalnum() for ch in r['continuation'])))
bold=pd.DataFrame(bold_records,columns=['id','model','mode','domain','source_cluster','words','no_alnum'])
bold['empty']=bold.words.eq(0)
display(bold.groupby(['model','mode']).agg(n=('id','size'),empty=('empty','mean'),
      no_alnum=('no_alnum','mean'),median_words=('words','median'),
      short_le5=('words',lambda x:(x<=5).mean()),
      long_ge20=('words',lambda x:(x>=20).mean())).round(4))
""")

code("""
coverage=bold.groupby(['model','mode'])[['empty','no_alnum']].mean().reset_index()
fig,axes=plt.subplots(1,2,figsize=(10,3.4),sharey=True)
for ax,mode in zip(axes,['raw','chat']):
    sub=coverage[coverage['mode']==mode].set_index('model').loc[models]
    x=np.arange(4);w=.35
    ax.bar(x-w/2,sub['empty'],w,label='Empty after strip',color='#BBBBBB')
    ax.bar(x+w/2,sub['no_alnum'],w,label='No letter or digit',color='#AA4499')
    ax.set_xticks(x,[names[m] for m in models]);ax.set_ylim(0,.7)
    ax.set_title(mode+' continuation coverage');ax.grid(axis='y',alpha=.2)
axes[1].legend(frameon=False)
finish(fig,'bold_lexical_coverage')
""")

code("""
fig,axes=plt.subplots(1,2,figsize=(11,3.7),sharey=True)
for ax,mode in zip(axes,['raw','chat']):
    for model in models:
        lengths=np.sort(bold[(bold.model==model)&(bold['mode']==mode)].words.to_numpy())
        ax.step(lengths,np.arange(1,len(lengths)+1)/len(lengths),where='post',
                label=names[model],color=colors[model])
    ax.set_xlim(0,50); ax.set_ylim(0,1); ax.set_xlabel('Output words');
    ax.set_title(mode+' prefix: cumulative output length');ax.grid(alpha=.2)
axes[0].set_ylabel('Fraction at or below this length')
axes[1].legend(frameon=False,loc='lower right')
finish(fig,'bold_length_cdf')
""")

code("""
domains=sorted(bold.domain.unique())
empty=bold.groupby(['model','mode','domain'])['empty'].mean()
matrix=np.array([[empty[m,mode,d] for d in domains] for m in models for mode in ['raw','chat']])
fig,ax=plt.subplots(figsize=(10,4.4)); im=ax.imshow(matrix,cmap='magma_r',vmin=0,vmax=1,aspect='auto')
ax.set_xticks(range(len(domains)),[d.replace('_',' ') for d in domains],rotation=25,ha='right')
ax.set_yticks(range(8),[f'{names[m]} {mode}' for m in models for mode in ['raw','chat']])
for i in range(8):
    for j in range(len(domains)):
        ax.text(j,i,f'{matrix[i,j]*100:.1f}%',ha='center',va='center',fontsize=8,
                color='white' if matrix[i,j]>.42 else 'black')
fig.colorbar(im,ax=ax,label='Empty fraction')
finish(fig,'bold_empty_by_domain')
""")

code("""
transition=[]
for model in models:
    paired=bold[bold.model==model].pivot(index='id',columns='mode',values='empty')
    counts=[int((paired.raw & paired.chat).sum()),int((paired.raw & ~paired.chat).sum()),
            int((~paired.raw & paired.chat).sum()),int((~paired.raw & ~paired.chat).sum())]
    transition.append(counts)
transition=np.asarray(transition)
labels=['Empty in both','Raw only empty','Chat only empty','Nonempty in both']
display(pd.DataFrame(transition,index=[names[m] for m in models],columns=labels))
fig,ax=plt.subplots(figsize=(9,3.6)); bottom=np.zeros(4)
for j,label in enumerate(labels):
    values=transition[:,j]/transition.sum(axis=1)
    ax.bar([names[m] for m in models],values,bottom=bottom,label=label)
    bottom+=values
ax.set_ylabel('Fraction of matched BOLD prompts');ax.set_ylim(0,1)
ax.legend(ncol=2,frameon=False,bbox_to_anchor=(1.02,1),loc='upper left')
finish(fig,'bold_mode_transitions')
""")

code("""
# Source-row cluster bootstrap: prompts from the same BOLD source row can be related.
cluster_results=[]
for model in ['llada','dream']:
    sub=bold[bold.model==model]
    paired=sub.pivot(index='id',columns='mode',values='empty')
    raw=sub[sub['mode']=='raw'].set_index('id').loc[paired.index]
    delta=paired['chat'].astype(int)-paired['raw'].astype(int)
    grouped=pd.DataFrame({'cluster':raw.source_cluster.to_numpy(),
                          'delta':delta.to_numpy()}).groupby('cluster').delta.agg(['sum','count'])
    sums=grouped['sum'].to_numpy(); counts=grouped['count'].to_numpy()
    rng=np.random.default_rng(42)
    draws=[]
    for _ in range(1000):
        ix=rng.integers(len(grouped),size=len(grouped))
        draws.append(sums[ix].sum()/counts[ix].sum())
    lo,hi=np.quantile(draws,[.025,.975])
    cluster_results.append((model,len(grouped),delta.mean(),lo,hi))
cluster_ci=pd.DataFrame(cluster_results,columns=['model','source_clusters','chat_minus_raw_empty','ci_low','ci_high'])
display(cluster_ci)
fig,ax=plt.subplots(figsize=(7,3.3))
for i,r in enumerate(cluster_ci.itertuples()):
    ax.errorbar(r.chat_minus_raw_empty,i,
                xerr=[[r.chat_minus_raw_empty-r.ci_low],[r.ci_high-r.chat_minus_raw_empty]],
                fmt='o',capsize=4,color=colors[r.model])
ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(len(cluster_ci)),[names[m] for m in cluster_ci.model])
ax.set_xlabel('Chat minus raw empty-output fraction (95% source-row bootstrap CI)')
ax.set_xlim(-.7,.3);ax.grid(axis='x',alpha=.2)
finish(fig,'bold_mode_cluster_ci')
""")

code("""
sentiment=[]
for model in models:
    for r in read_jsonl(ROOT/f'results/sentiment/bold_chat_full_{model}_sentiment.jsonl'):
        if not r['empty_source_prompt']:
            sentiment.append((r['id'],model,bool(r['negative_continuation'])))
sent=pd.DataFrame(sentiment,columns=['id','model','negative'])
chat=bold[bold['mode']=='chat'].merge(sent,on=['id','model'],validate='one_to_one')
chat['length_bin']=pd.cut(chat.words,bins=[0,5,19,np.inf],labels=['1–5','6–19','20+'])
rates=chat[~chat['empty']].groupby(['model','length_bin'],observed=True).agg(
    n=('id','size'),negative=('negative','mean')).reset_index()
display(rates.sort_values(['model','length_bin']))
fig,ax=plt.subplots(figsize=(9,3.8)); x=np.arange(3); w=.2
for j,model in enumerate(models):
    sub=rates.set_index(['model','length_bin'])
    values=[sub.loc[(model,b),'negative'] if (model,b) in sub.index else np.nan
            for b in ['1–5','6–19','20+']]
    ax.bar(x+(j-1.5)*w,values,w,label=names[model],color=colors[model])
ax.set_xticks(x,['1–5','6–19','20+']);ax.set_xlabel('Output length in words')
ax.set_ylabel('Negative VADER rate among nonempty outputs');ax.legend(frameon=False)
ax.grid(axis='y',alpha=.2)
finish(fig,'bold_sentiment_by_length')
""")

md("""**Reading the BOLD plots.** The same 23,674 valid prompts under raw and chat formats give a matched comparison. LLaDA's raw-prefix empty rate is far higher than its instructed-chat rate. Dream produces many more short and empty outputs in chat, including across multiple BOLD domains. The EOS and larger-budget probes below test a generation mechanism, while the length-conditioned sentiment plot shows why the headline negative rate is hard to interpret by itself.""")

code("""
# Conditional illustrations, chosen by a fixed hash within each domain.
# They show what a short Dream continuation can look like; they are not a random
# sample and must never be used to estimate a rate.
import hashlib
dream_text={r['id']:r for r in read_jsonl(ROOT/'results/bold_chat_full_dream.jsonl')
            if r['prompt'].strip()}
llada_text={r['id']:r for r in read_jsonl(ROOT/'results/bold_chat_full_llada.jsonl')
            if r['prompt'].strip()}
examples=[]
for domain in sorted({r['domain'] for r in dream_text.values()}):
    candidates=[i for i,r in dream_text.items() if r['domain']==domain and
                1<=len(r['continuation'].split())<=5 and
                len(llada_text[i]['continuation'].split())>=20]
    chosen=min(candidates,key=lambda i:hashlib.sha256(('qual-v1:'+i).encode()).digest())
    examples.append((domain,dream_text[chosen]['prompt'][:100],
                     dream_text[chosen]['continuation'][:120],
                     llada_text[chosen]['continuation'][:120]))
display(pd.DataFrame(examples,columns=['domain','prompt excerpt','Dream excerpt','LLaDA excerpt']))
del dream_text,llada_text
""")

md("""## 3. Paired sentence scores and membership: validity gate

Dream is included only if complete, aligned-v2 outputs exist. The unshifted legacy Dream values are never used in these figures. The remaining model results are the original completed runs. Pair scores are our project-specific masked-reconstruction/AR preferences, not official leaderboard scores.""")

code("""
from sklearn.metrics import roc_auc_score
def complete(path,n):
    return path.exists() and sum(1 for _ in read_jsonl(path))==n

pair_rows=[]
for model in models:
    for task,n in [('crows',1508),('stereoset',2106)]:
        path=(ROOT/f'results/exploration/{task}_dream_aligned_v2.jsonl' if model=='dream'
              else ROOT/f'results/{task}_full_{model}.jsonl')
        if not complete(path,n): continue
        vals=[]; meaningful=[]
        for r in read_jsonl(path):
            if task=='crows':
                vals.append(r['prefers_more'] if r['stereo_antistereo']=='stereo' else not r['prefers_more'])
            else:
                vals.append(r['prefers_stereotype']); meaningful.extend(r['prefers_meaningful'])
        pair_rows.append((model,task,np.mean(vals),np.mean(meaningful) if meaningful else np.nan))
pairs=pd.DataFrame(pair_rows,columns=['model','task','stereotype','meaningful'])
display(pairs)
fig,axes=plt.subplots(1,2,figsize=(10,3.5),sharey=True)
for ax,task in zip(axes,['crows','stereoset']):
    subset=pairs[pairs.task==task].set_index('model')
    ms=[m for m in models if m in subset.index]
    ax.bar([names[m] for m in ms],subset.loc[ms,'stereotype'],color=[colors[m] for m in ms])
    ax.axhline(.5,color='black',ls='--',lw=.8);ax.set_title(task+' stereotype preference')
    ax.set_ylim(0,1);ax.grid(axis='y',alpha=.2)
finish(fig,'pair_preference_valid_only')
""")

code("""
stereo=pairs[pairs.task=='stereoset'].set_index('model')
ms=[m for m in models if m in stereo.index]
fig,ax=plt.subplots(figsize=(8.5,3.6))
ax.bar([names[m] for m in ms],stereo.loc[ms,'meaningful'],
       color=[colors[m] for m in ms])
ax.set_ylim(0,1);ax.set_ylabel('Meaningful option over unrelated')
ax.set_title('StereoSet language-meaning check')
ax.grid(axis='y',alpha=.2)
finish(fig,'stereoset_meaningful_valid_only')
""")

code("""
mia_rows=[]
for model in models:
    paths={mode:(ROOT/f'results/exploration/mia_{mode}_dream_aligned_v2.jsonl' if model=='dream'
                 else ROOT/f'results/mia/{mode}_{model}.jsonl')
           for mode in ['base','full','goldfish']}
    if not all(complete(p,192) for p in paths.values()): continue
    for mode,path in paths.items():
        rows=list(read_jsonl(path))
        mia_rows.append((model,mode,roc_auc_score([r['member'] for r in rows],
                                                   [r['response_score'] for r in rows]),
                         sum(r['member'] and r['exact_extraction'] for r in rows)))
mia=pd.DataFrame(mia_rows,columns=['model','mode','auc','exact_member'])
display(mia)
fig,ax=plt.subplots(figsize=(9,3.7)); ms=[m for m in models if m in set(mia.model)]
x=np.arange(len(ms)); w=.25
for j,mode in enumerate(['base','full','goldfish']):
    sub=mia[mia['mode']==mode].set_index('model')
    ax.bar(x+(j-1)*w,sub.loc[ms,'auc'],w,label=mode)
ax.axhline(.5,color='black',ls='--',lw=.8);ax.set_xticks(x,[names[m] for m in ms])
ax.set_ylim(0,1);ax.set_ylabel('MIA AUROC');ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
finish(fig,'mia_valid_only')
""")

code("""
mitigation_path=ROOT/'results/summary/dream_mitigation_aligned_v2.tsv'
if mitigation_path.exists():
    mitigation=pd.read_csv(mitigation_path,sep='\\t')
    display(mitigation)
    sub=mitigation[mitigation.repetition_group.astype(str).isin(['1','4','16'])]
    fig,ax=plt.subplots(figsize=(7.5,3.5));x=np.arange(len(sub));w=.35
    ax.bar(x-w/2,sub.full_auc,w,label='Full',color='#EE6677')
    ax.bar(x+w/2,sub.goldfish_auc,w,label='Goldfish',color='#AA4499')
    ax.axhline(.5,color='black',ls='--',lw=.8)
    ax.set_xticks(x,sub.repetition_group.astype(str));ax.set_xlabel('Canary copies in training')
    ax.set_ylabel('Dream corrected MIA AUROC');ax.set_ylim(0,1)
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
    finish(fig,'dream_corrected_mia_by_repetition')
""")

md("""## 4. Interventions: registered before looking at new results

The samples are fixed by a hash of each source ID: 60 BOLD prompts (12 per domain) and 132 BBQ items (three per category × context × polarity). All conditions use the same items. **BOLD:** compare old chat 48-token/24-step outputs with 48/12, 48/48, 96/96 length+steps, and 96/96 with an explicit minimum-length instruction. The 48/12 probe was added after the 48/48 outputs showed shorter continuations; it is exploratory follow-up. For LLaDA, test official-paper-inspired EOS-confidence deprioritization in raw prefix and at both chat step budgets. **BBQ:** compare old 32/16 with 32/32 diffusion steps, then an explicit evidence/abstention instruction at 32/32. AR models receive the same explicit instruction as controls.

These are exploratory subset interventions. A result on 60 or 132 items is evidence about these prompts, not a replacement for the complete baseline. Compare conditions *within* a model and show denominators; do not attribute a mixed length+step change to either component alone.""")

code("""
bold_probe_ids={r['id'] for r in read_jsonl(ROOT/'data/processed/bold_explore_60.jsonl')}
probe_bold=[]
for model in ['llada','dream']:
    baseline=bold[(bold.model==model)&(bold['mode']=='chat')&bold.id.isin(bold_probe_ids)]
    probe_bold.append((model,'baseline 48/24',len(baseline),baseline['empty'].mean(),
                       baseline.words.median(),(baseline.words>=20).mean(),np.nan,np.nan))
    if model=='llada':
        raw=bold[(bold.model==model)&(bold['mode']=='raw')&bold.id.isin(bold_probe_ids)]
        probe_bold.append((model,'raw baseline 48/24',len(raw),raw['empty'].mean(),
                           raw.words.median(),(raw.words>=20).mean(),np.nan,np.nan))
    for condition in ['short12','step48','long96','min20']+(
            ['eos_raw','eos_step48','eos_long96'] if model=='llada' else []):
        path=ROOT/f'results/exploration/bold_{condition}_{model}.jsonl'
        if not complete(path,60): continue
        rows=list(read_jsonl(path)); lengths=np.array([len(r['output'].split()) for r in rows])
        eos_first=[r['raw_generation']['eos_positions'][0] for r in rows
                   if r['raw_generation']['eos_positions']]
        probe_bold.append((model,condition,60,np.mean(lengths==0),np.median(lengths),
                           np.mean(lengths>=20),np.mean([r['raw_generation']['first_token_is_eos']
                                                         for r in rows]),
                           np.median(eos_first) if eos_first else np.nan))
bold_interventions=pd.DataFrame(probe_bold,columns=['model','condition','n','empty',
    'median_words','ge20','first_eos','median_first_eos_position'])
display(bold_interventions)
fig,axes=plt.subplots(1,2,figsize=(10,3.8))
for ax,model in zip(axes,['llada','dream']):
    sub=bold_interventions[(bold_interventions.model==model)&
                           ~bold_interventions.condition.str.startswith('raw')&
                           (bold_interventions.condition!='eos_raw')]
    ax.bar(sub.condition,sub['empty'],color=colors[model]);ax.set_ylim(0,1)
    ax.set_title(names[model]+' chat empty rate');ax.tick_params(axis='x',rotation=30)
    ax.grid(axis='y',alpha=.2)
finish(fig,'bold_generation_interventions')
""")

code("""
fig,ax=plt.subplots(figsize=(8,4))
for r in bold_interventions.itertuples():
    if not np.isfinite(r.median_first_eos_position): continue
    ax.scatter(r.median_first_eos_position,r.median_words,
               s=90,color=colors[r.model])
    ax.annotate(f'{names[r.model]} {r.condition}',
                (r.median_first_eos_position,r.median_words),
                xytext=(5,4),textcoords='offset points',fontsize=8)
ax.set_xlabel('Median position of first EOS token among outputs containing EOS')
ax.set_ylabel('Median visible output words')
ax.grid(alpha=.2)
finish(fig,'bold_eos_position_vs_length')
""")

code("""
# Paired candidate bootstrap intervals for the small BOLD probe; exploratory.
bold_changes=[]
for model in ['llada','dream']:
    baseline=bold[(bold.model==model)&(bold['mode']=='chat')&bold.id.isin(bold_probe_ids)].set_index('id')
    for condition in ['short12','step48','long96','min20']+(
            ['eos_step48','eos_long96'] if model=='llada' else []):
        path=ROOT/f'results/exploration/bold_{condition}_{model}.jsonl'
        if not complete(path,60): continue
        probe={r['id']:r for r in read_jsonl(path)}
        ids=sorted(baseline.index)
        changes=np.array([int(not any(ch.isalnum() for ch in probe[i]['output']))-
                          int(baseline.loc[i,'no_alnum']) for i in ids])
        rng=np.random.default_rng(42)
        draw=np.mean(changes[rng.integers(60,size=(2000,60))],axis=1)
        lo,hi=np.quantile(draw,[.025,.975])
        bold_changes.append((model,condition,changes.mean(),lo,hi))
bold_change=pd.DataFrame(bold_changes,columns=['model','condition','change_no_alnum','ci_low','ci_high'])
display(bold_change)
""")

md("""### Nonoverlapping BOLD replication with stop-token decoding

The 60-prompt short-step probe suggested that Dream writes longer continuations with 12 instead of 24 diffusion steps. The original full BOLD adapter removed special tokens during decoding but did not cut off text at the **first EOS**; some post-EOS text could therefore remain visible. A separate 600-prompt sample (120 per domain, with no overlap with the first 60) reruns **both** 24-step and 12-step Dream generation with first-EOS truncation. This paired comparison holds model, prompt, 48-token budget, and decoding policy fixed. The baseline full BOLD figures above describe the original fixed-span decoder; treat them separately from this corrected-stop experiment.""")

code("""
replicate_bold_path=ROOT/'results/summary/bold_replicate_600.tsv'
if replicate_bold_path.exists():
    bold_rep=pd.read_csv(replicate_bold_path,sep='\\t')
    bold_rep_change=pd.read_csv(ROOT/'results/summary/bold_replicate_paired_changes.tsv',sep='\\t')
    display(bold_rep.round(4),bold_rep_change.round(4))
    domains=sorted(d for d in bold_rep.domain.unique() if d!='ALL')
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),sharey=True)
    for ax,metric,title in zip(axes,['no_alnum_fraction','fraction_ge20'],
                               ['No lexical content','At least 20 words']):
        x=np.arange(len(domains));w=.36
        for j,condition in enumerate(['baseline24_eos','short12_eos']):
            values=bold_rep[(bold_rep.domain.isin(domains))&
                            (bold_rep.condition==condition)].set_index('domain').loc[domains,metric]
            ax.bar(x+(j-.5)*w,values,w,label='24 steps' if j==0 else '12 steps',
                   color=['#4477AA','#EE6677'][j])
        ax.set_xticks(x,[d.replace('_',' ') for d in domains],rotation=25)
        ax.set_ylim(0,1);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[1].legend(frameon=False)
    finish(fig,'bold_replication_domains')
    fig,ax=plt.subplots(figsize=(8.5,3.5))
    labels={'empty':'Empty','no_alnum':'No lexical content',
            'ge20':'At least 20 words','first_eos':'First token is EOS'}
    for j,r in enumerate(bold_rep_change.itertuples()):
        x=r.short12_minus_baseline24
        ax.errorbar(x,j,xerr=[[x-r.ci_low],[r.ci_high-x]],fmt='o',capsize=3,
                    color='#EE6677')
    ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(len(bold_rep_change)),
                                                    [labels[x] for x in bold_rep_change.metric])
    ax.set_xlabel('12-step minus 24-step fraction (95% domain-stratified bootstrap CI)')
    ax.grid(axis='x',alpha=.2)
    finish(fig,'bold_replication_paired_changes')
else:
    print('First-EOS BOLD replication pending.')
""")

md("""### LLaDA EOS-confidence replication on the same 600 prompts

The LLaDA pilot at 48/48 steps showed that assigning EOS low fill priority moved its first occurrence to the final token slot and restored much longer continuations. We rerun 48/48 with and without that single selection change on the 600 new BOLD prompts, using first-EOS truncation in both conditions. This measures a **coverage** effect. It does not show that the extra text is true, coherent, or less biased; the examples need separate review for those claims.""")

code("""
llada_rep_path=ROOT/'results/summary/bold_llada_replicate_600.tsv'
if llada_rep_path.exists():
    llada_rep=pd.read_csv(llada_rep_path,sep='\\t')
    llada_rep_change=pd.read_csv(ROOT/'results/summary/bold_llada_replicate_paired_changes.tsv',sep='\\t')
    display(llada_rep.round(4),llada_rep_change.round(4))
    domains=sorted(d for d in llada_rep.domain.unique() if d!='ALL')
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),sharey=True)
    for ax,metric,title in zip(axes,['no_alnum_fraction','fraction_ge20'],
                               ['No lexical content','At least 20 words']):
        x=np.arange(len(domains));w=.36
        for j,condition in enumerate(['step48_stop','eos_step48_stop']):
            values=llada_rep[(llada_rep.domain.isin(domains))&
                             (llada_rep.condition==condition)].set_index('domain').loc[domains,metric]
            ax.bar(x+(j-.5)*w,values,w,label='Original selection' if j==0 else 'EOS last',
                   color=['#4477AA','#EE6677'][j])
        ax.set_xticks(x,[d.replace('_',' ') for d in domains],rotation=25)
        ax.set_ylim(0,1);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axes[0].legend(frameon=False)
    finish(fig,'bold_llada_eos_replication')
else:
    print('LLaDA first-EOS replication pending.')
""")

md("""### Coverage against the AR baselines on the same 600 prompts

For context, the completed Llama 3 and Qwen2.5 instructed-chat runs provide AR results on these same prompts with a 48-token maximum. Their autoregressive generation stops at EOS. The selected diffusion settings below were chosen after examining outputs, so this is a descriptive coverage comparison, not an architecture-controlled or fairness ranking.""")

code("""
if replicate_bold_path.exists() and llada_rep_path.exists():
    replicate_ids={r['id'] for r in read_jsonl(ROOT/'data/processed/bold_replicate_600.jsonl')}
    rows=[]
    for model in ['llama','qwen']:
        sub=bold[(bold.model==model)&(bold['mode']=='chat')&bold.id.isin(replicate_ids)]
        rows.append((names[model]+' baseline',len(sub),sub.no_alnum.mean(),
                     (sub.words>=20).mean(),sub.words.median()))
    for model,condition in [('llada','eos_step48_stop'),('dream','short12_eos')]:
        tab=llada_rep if model=='llada' else bold_rep
        r=tab[(tab.domain=='ALL')&(tab.condition==condition)].iloc[0]
        rows.append((names[model]+' selected decoder',r.n,r.no_alnum_fraction,
                     r.fraction_ge20,r.median_words))
    cross=pd.DataFrame(rows,columns=['setting','n','no_alnum','ge20','median_words'])
    display(cross.round(4))
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),sharey=True)
    for ax,metric,title in zip(axes,['no_alnum','ge20'],
                               ['No lexical content','At least 20 words']):
        ax.bar(cross.setting,cross[metric],color=['#4477AA','#228833','#66CCEE','#EE6677'])
        ax.set_ylim(0,1);ax.set_title(title);ax.tick_params(axis='x',rotation=25)
        ax.grid(axis='y',alpha=.2)
    finish(fig,'bold_ar_selected_coverage')
""")

md("""### Coverage versus repetition

Longer output is not automatically a better continuation. We measure two narrow diagnostics on the 600-item replications: whether an output starts by copying the entire prompt, and the fraction of repeated three-word sequences among outputs of at least 20 words. A sequence with at least 20% repeated trigrams is flagged as heavily repetitive. These checks catch obvious failure modes, but they cannot measure factuality, fluency, or harmful stereotypes. The 24-step Dream condition has no 20-word outputs, so its long-output repetition rate is undefined. The 16- and 20-step Dream runs were added after inspecting the 12-step coverage gain and repetition; this is a post hoc sweep.""")

code("""
quality_path=ROOT/'results/summary/generation_quality_600.tsv'
if quality_path.exists():
    quality=pd.read_csv(quality_path,sep='\\t')
    display(quality.round(4))
    sweep_changes=pd.read_csv(ROOT/'results/summary/bold_dream_step_sweep_paired.tsv',sep='\\t')
    display(sweep_changes.round(4))
    dream_steps={'short12_eos':12,'steps16_eos':16,'steps20_eos':20,'baseline24_eos':24}
    dream_q=quality[quality.model=='dream'].copy()
    dream_q['steps']=dream_q.condition.map(dream_steps)
    dream_q=dream_q.sort_values('steps')
    dream_cov=bold_rep[(bold_rep.domain=='ALL')&
                       (bold_rep.condition.isin(dream_steps))].copy()
    dream_cov['steps']=dream_cov.condition.map(dream_steps)
    dream_cov=dream_cov.sort_values('steps')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    axes[0].plot(dream_cov.steps,dream_cov.fraction_ge20,'o-',color=colors['dream'])
    axes[0].set_ylabel('Fraction with at least 20 words')
    axes[0].set_ylim(0,1);axes[0].grid(alpha=.2)
    axes[1].plot(dream_q.steps,dream_q.heavy_repetition_fraction_ge20,'o-',
                 color=colors['dream'])
    for r in dream_q.itertuples():
        if np.isfinite(r.heavy_repetition_fraction_ge20):
            axes[1].annotate(f'n={r.n_ge20_words}',
                             (r.steps,r.heavy_repetition_fraction_ge20),
                             xytext=(4,4),textcoords='offset points',fontsize=8)
    axes[1].set_ylabel('Heavily repetitive among 20+ word outputs')
    axes[1].set_ylim(0,1);axes[1].grid(alpha=.2)
    for ax in axes: ax.set_xlabel('Dream diffusion steps (48-token budget)')
    finish(fig,'bold_dream_step_tradeoff')
    fig,ax=plt.subplots(figsize=(8,3.5))
    subset=quality[(quality.model=='llada')&
                   (quality.condition.isin(['step48_stop','eos_step48_stop']))]
    ax.bar(['Original selection','EOS depriority'],
           subset.set_index('condition').loc[['step48_stop','eos_step48_stop'],
                                             'prompt_recopy_fraction'],
           color=['#4477AA','#EE6677'])
    ax.set_ylabel('Fraction starting by copying the full prompt')
    ax.set_ylim(0,1);ax.grid(axis='y',alpha=.2)
    finish(fig,'bold_llada_prompt_recopy')
    paired_quality_path=ROOT/'results/summary/bold_dream_repetition_paired.tsv'
    if paired_quality_path.exists():
        quality_pairs=pd.read_csv(paired_quality_path,sep='\\t')
        display(quality_pairs.round(4))
        fig,ax=plt.subplots(figsize=(7.5,3.2))
        for j,r in enumerate(quality_pairs.itertuples()):
            x=r.heavy_repetition_change
            ax.errorbar(x,j,xerr=[[x-r.heavy_ci_low],[r.heavy_ci_high-x]],
                        fmt='o',capsize=3,color=colors['dream'])
        ax.axvline(0,color='black',lw=.8)
        ax.set_yticks(range(len(quality_pairs)),
                      ['20 steps minus '+('16' if r.reference=='steps16_eos' else '12')
                       for r in quality_pairs.itertuples()])
        ax.set_xlabel('Heavy repetition change on prompts long in both conditions (95% paired CI)')
        ax.grid(axis='x',alpha=.2)
        finish(fig,'bold_dream_repetition_paired')
else:
    print('Generation quality and intermediate-step sweep pending.')
""")

code("""
# Illustrative examples chosen after inspecting outputs; not frequency evidence.
examples=[]
for model,item_id,conditions in [
    ('dream','political_ideology_prompt_wiki.json:162:8',
     ['baseline24_eos','short12_eos','steps16_eos','steps20_eos']),
    ('llada','gender_prompt_wiki.json:1013:0',
     ['step48_stop','eos_step48_stop'])]:
    for condition in conditions:
        path=ROOT/f'results/exploration/bold_replicate_{condition}_{model}.jsonl'
        if not path.exists():continue
        hit=[r for r in read_jsonl(path) if r['id']==item_id]
        if len(hit)==1:
            r=hit[0]
            examples.append((model,condition,r['prompt'],r['output'][:260]))
display(pd.DataFrame(examples,columns=['model','condition','prompt','continuation excerpt']))
""")

code("""
bbq_probe_ids={r['id'] for r in read_jsonl(ROOT/'data/processed/bbq_explore_132.jsonl')}
probe_bbq=[]
for model in models:
    base=bbq[(bbq.model==model)&bbq.id.isin(bbq_probe_ids)]
    for cond in ['ambig','disambig']:
        sub=base[base.condition==cond]
        probe_bbq.append((model,'baseline 32/16',cond,len(sub),sub.correct.mean(),sub.unknown.mean()))
    for intervention in (['step32','explicit'] if model in ['llada','dream'] else ['explicit']):
        path=ROOT/f'results/exploration/bbq_{intervention}_{model}.jsonl'
        if not complete(path,132): continue
        for cond in ['ambig','disambig']:
            rows=[r for r in read_jsonl(path) if r['context_condition']==cond]
            unknown={i:source[i][4] for i in [r['id'] for r in rows]}
            probe_bbq.append((model,intervention,cond,len(rows),
                              np.mean([r['prediction']==r['label'] for r in rows]),
                              np.mean([r['prediction']==unknown[r['id']] for r in rows])))
bbq_interventions=pd.DataFrame(probe_bbq,columns=['model','intervention','condition','n','accuracy','unknown'])
display(bbq_interventions)
fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
for ax,cond in zip(axes,['ambig','disambig']):
    sub=bbq_interventions[bbq_interventions.condition==cond]
    for model in models:
        rows=sub[sub.model==model]
        ax.plot(rows.intervention,rows.accuracy,'o-',label=names[model],color=colors[model])
    ax.set_title(cond+' accuracy on fixed subset');ax.set_ylim(0,1);ax.tick_params(axis='x',rotation=30)
    ax.grid(alpha=.2)
axes[1].legend(frameon=False,bbox_to_anchor=(1.02,1),loc='upper left')
finish(fig,'bbq_prompt_interventions')
""")

code("""
bbq_changes=[]
for model in models:
    baseline=bbq[(bbq.model==model)&bbq.id.isin(bbq_probe_ids)].set_index('id')
    for condition in (['step32','explicit'] if model in ['llada','dream'] else ['explicit']):
        path=ROOT/f'results/exploration/bbq_{condition}_{model}.jsonl'
        if not complete(path,132): continue
        probe={r['id']:r for r in read_jsonl(path)}
        for context in ['ambig','disambig']:
            ids=sorted(baseline[baseline.condition==context].index)
            changes=np.array([int(probe[i]['prediction']==probe[i]['label'])-
                              int(baseline.loc[i,'correct']) for i in ids])
            rng=np.random.default_rng(42)
            draw=np.mean(changes[rng.integers(len(ids),size=(2000,len(ids)))],axis=1)
            lo,hi=np.quantile(draw,[.025,.975])
            bbq_changes.append((model,condition,context,len(ids),changes.mean(),lo,hi))
bbq_change=pd.DataFrame(bbq_changes,columns=['model','condition','context','n',
                                              'accuracy_change','ci_low','ci_high'])
display(bbq_change)
""")

md("""### Nonoverlapping BBQ replication

The initial 132-item prompt result was inspected during exploration. A second deterministic sample of 1,320 **different** BBQ items (30 per category × context × polarity stratum) tests whether it persists. Dream receives both 32/32 steps with the original prompt and 32/32 with the explicit evidence rule, so their direct contrast isolates the prompt change at a fixed decoding budget. Qwen receives the same explicit rule as an AR control. Baseline predictions come from the completed full benchmark, without new inference. Intervals resample items within each of the 22 category × polarity strata per context. They quantify sampling variation on this fixed benchmark extract; they do not remove checkpoint or prompt-selection uncertainty.""")

code("""
replicate_path=ROOT/'results/summary/bbq_replicate_1320.tsv'
if replicate_path.exists():
    replicate=pd.read_csv(replicate_path,sep='\\t')
    display(replicate.round(4))
    labels={('dream','step32','baseline'):'Dream: more steps',
            ('dream','explicit','baseline'):'Dream: steps + evidence rule',
            ('dream','explicit','step32'):'Dream: evidence rule at 32 steps',
            ('qwen','explicit','baseline'):'Qwen: evidence rule'}
    fig,axes=plt.subplots(1,2,figsize=(11.5,4),sharey=True)
    for ax,context in zip(axes,['ambig','disambig']):
        sub=replicate[replicate.context==context]
        for y,(key,label) in enumerate(labels.items()):
            r=sub[(sub.model==key[0])&(sub.condition==key[1])&
                  (sub.reference==key[2])].iloc[0]
            x=r.accuracy_change
            ax.errorbar(x,y,xerr=[[x-r.ci_low],[r.ci_high-x]],fmt='o',
                        color=colors[key[0]],capsize=3)
        ax.axvline(0,color='black',lw=.8);ax.grid(axis='x',alpha=.2)
        ax.set_title(context);ax.set_xlabel('Paired accuracy change (95% stratified bootstrap CI)')
        ax.set_yticks(range(len(labels)),list(labels.values()))
    finish(fig,'bbq_replication')
else:
    print('Replication job pending; this figure will appear after validation.')
""")

code("""
# Check whether the Dream evidence-rule gain is concentrated in one category.
rep_source={r['id']:r for r in read_jsonl(ROOT/'data/processed/bbq_replicate_1320.jsonl')}
rep_step={r['id']:r for r in read_jsonl(ROOT/'results/exploration/bbq_replicate_step32_dream.jsonl')}
rep_exp={r['id']:r for r in read_jsonl(ROOT/'results/exploration/bbq_replicate_explicit_dream.jsonl')}
rep_categories=sorted({r['category'] for r in rep_source.values()})
category_changes=[]
for category in rep_categories:
    row=[]
    for context in ['ambig','disambig']:
        ids=[i for i,r in rep_source.items()
             if r['category']==category and r['context_condition']==context]
        if len(ids)!=60:raise ValueError('replication category balance changed')
        change=np.mean([int(rep_exp[i]['prediction']==rep_source[i]['label'])-
                        int(rep_step[i]['prediction']==rep_source[i]['label']) for i in ids])
        row.append(change)
    category_changes.append(row)
category_changes=np.array(category_changes)
display(pd.DataFrame(category_changes,index=rep_categories,
                     columns=['Ambiguous change','Disambiguated change']).round(3))
fig,ax=plt.subplots(figsize=(7.5,5.5))
im=ax.imshow(category_changes,cmap='RdBu',vmin=-.3,vmax=.3,aspect='auto')
ax.set_xticks([0,1],['Ambiguous','Disambiguated'])
ax.set_yticks(np.arange(len(rep_categories)),
              [c.replace('_',' ') for c in rep_categories])
for i in range(len(rep_categories)):
    for j in range(2):
        ax.text(j,i,f'{category_changes[i,j]*100:+.1f}',ha='center',va='center',
                color='white' if abs(category_changes[i,j])>.17 else 'black')
fig.colorbar(im,ax=ax,label='Dream explicit minus original prompt at 32 steps')
finish(fig,'bbq_replication_categories')
""")

code("""
# Decompose the prompt's change to the BBQ ambiguous signed bias score.
meta=pd.read_csv(ROOT/'data/processed/bbq_additional_metadata.csv',
                 dtype={'example_id':str})
meta_map={(r.category,r.example_id):r.target_loc for r in meta.itertuples()}
prompt_bias=[]
for condition,preds in [('original 32 steps',rep_step),('explicit evidence rule',rep_exp)]:
    ids=[i for i,r in rep_source.items() if r['context_condition']=='ambig']
    correct=0;guesses=0;targets=0
    for i in ids:
        row=rep_source[i];pred=preds[i]['prediction']
        unknown=next(j for j in range(3) if row['answer_info'][f'ans{j}'][1].lower()=='unknown')
        target=meta_map[(row['category'],str(row['source_row']))]
        if pd.isna(target):continue
        target=int(target)
        correct+=pred==row['label']
        if pred is not None and pred!=unknown:
            guesses+=1
            targets+=pred==target
    accuracy=correct/len(ids)
    guess_fraction=guesses/len(ids)
    target_given_guess=targets/guesses
    bias=(2*target_given_guess-1)*(1-accuracy)
    prompt_bias.append((condition,accuracy,guess_fraction,target_given_guess,bias))
prompt_bias=pd.DataFrame(prompt_bias,columns=['condition','ambiguous_accuracy',
    'guess_fraction','target_given_guess','signed_bias_score'])
display(prompt_bias.round(4))
fig,ax=plt.subplots(figsize=(7.2,3.8))
for r in prompt_bias.itertuples():
    ax.scatter(r.guess_fraction,r.target_given_guess,s=110)
    ax.annotate(f'{r.condition}  (bias {r.signed_bias_score:.3f})',
                (r.guess_fraction,r.target_given_guess),
                xytext=(8,5),textcoords='offset points')
ax.set_xlim(0,.7);ax.set_ylim(.5,.9)
ax.set_xlabel('Fraction of ambiguous questions answered with a person')
ax.set_ylabel('Stereotype-target share among mapped guesses')
ax.grid(alpha=.2)
finish(fig,'bbq_prompt_bias_decomposition')
""")

md("""## Interpretation and decision rule

1. If extra steps change outcomes on the same prompts, decoding budget contributes to the baseline difference. The within-model paired change is the evidence; the model-paper claims alone are not.
2. If the explicit evidence rule raises Dream's ambiguous accuracy while preserving disambiguated accuracy, uncertainty handling is a useful improvement. If AR models improve similarly, this is a general prompt effect rather than a diffusion-specific repair.
3. If BOLD output length increases, compare sentiment at similar lengths and report empty rates. A lower negative rate with almost no output is not evidence of less social bias.
4. If corrected Dream pair/MIA rankings move substantially from the old values, that difference is a **measurement correction**, not improved checkpoint behavior.
5. Separate architectural hypotheses from established causes. LLaDA and Dream differ in initialization, supervision, head alignment, and decoding. A two-model comparison alone cannot identify which training difference causes a benchmark gap.

The next validation stage should replicate any promising intervention across multiple random subsets and seeds, then rerun only the best-defined condition on the complete benchmark. Human review of a small, blinded BOLD sample would add a direct quality measure beyond VADER sentiment.""")

N.cells = cells
N.metadata['kernelspec'] = {'display_name':'dllm-eval','language':'python','name':'dllm-eval'}
N.metadata['language_info'] = {'name':'python','version':'3.11'}
target = ROOT / 'notebooks/divergence_analysis.ipynb'
nbf.write(N,target)
print(target, len([c for c in cells if c.cell_type=='code']), 'code cells')
