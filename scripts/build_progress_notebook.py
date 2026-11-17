#!/usr/bin/env python3
"""Execute the sections whose full outputs have already passed validation."""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
full = nbf.read(ROOT / "notebooks/final_comparison.ipynb", as_version=4)
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = [
    md("# Completed-run comparison while remaining GPU jobs run\n\n"
       "Full BBQ, instructed BOLD, and controlled synthetic memorization are complete "
       "for all assigned models. Raw BOLD and pair benchmarks are complete for Llama 3, "
       "Llama 3.1, LLaDA, and Dream. The queued final notebook will compare all full "
       "runs after the remaining jobs finish."),
    md("The BBQ bias metric uses author target mappings. BOLD VADER sentiment is a lexical "
       "proxy and empty continuations are reported separately. Pair scores are exploratory "
       "within-model preferences. Synthetic MIA is fine-tuning membership, with one seed."),
    full.cells[2], full.cells[4], full.cells[5], full.cells[6],
    full.cells[7], full.cells[8], full.cells[9], full.cells[10], full.cells[11],
    md("## Instructed BOLD · all five models, 23,674 valid source prompts each"),
    code("""
bold_chat = pd.read_csv(ROOT/'results/summary/bold_chat_full_sentiment.tsv', sep='\\t')
domains = ['gender','race','profession','religious_ideology','political_ideology']
b = bold_chat[(bold_chat.category=='ALL') & (bold_chat.domain.isin(domains))].copy()
b['empty_rate'] = b['empty']/b['n']
fig, axes = plt.subplots(1,2,figsize=(14.5,4.8))
for ax, metric, title in zip(axes,
    ['negative_fraction_nonempty','empty_rate'],
    ['Negative VADER sentiment | nonempty outputs','Empty continuation rate']):
    mat = b.pivot(index='model',columns='domain',values=metric).reindex(index=MODELS,columns=domains)
    mat.index = [LABEL[m] for m in MODELS]
    sns.heatmap(mat,ax=ax,cmap='YlOrRd',annot=True,fmt='.1%',vmin=0,
                cbar_kws={'label':'Fraction'})
    ax.set_title(title);ax.set_xlabel('');ax.set_ylabel('')
    ax.tick_params(axis='x',rotation=30)
fig.tight_layout();fig.savefig(FIGURES/'bold_chat_progress.png',bbox_inches='tight');plt.show()
overall_bold=bold_chat.query("domain=='ALL' and category=='ALL'").copy()
overall_bold['empty_fraction']=overall_bold['empty']/overall_bold['n']
display(overall_bold[['model','n','empty_fraction','negative_fraction_nonempty']])
"""),
    md("## Raw BOLD versus instructed BOLD · four completed models"),
    code("""
raw_bold=pd.read_csv(ROOT/'results/summary/bold_raw_completed_four_sentiment.tsv',sep='\\t')
mode_comparison=pd.concat([raw_bold,bold_chat])
mode_comparison=mode_comparison.query("domain=='ALL' and category=='ALL' and model!='qwen'").copy()
mode_comparison['empty_fraction']=mode_comparison['empty']/mode_comparison['n']
display(mode_comparison[['model','prompt_mode','n','empty_fraction','negative_fraction_nonempty']]
        .sort_values(['model','prompt_mode']))
four=['llama','llada','dream','llama31']
fig,ax=plt.subplots(figsize=(8.5,4.3))
for offset,mode,color in [(-.18,'raw_prefix','#526fad'),(.18,'chat_continuation','#e6a24c')]:
    part=mode_comparison.query('prompt_mode==@mode').set_index('model').reindex(four)
    ax.bar(np.arange(4)+offset,part.empty_fraction,.36,label=mode,color=color)
ax.set_xticks(np.arange(4),[LABEL[m] for m in four])
ax.set_ylabel('Empty continuation fraction');ax.legend(frameon=False)
ax.set_title('Prompt-format sensitivity in four completed BOLD runs')
fig.tight_layout();fig.savefig(FIGURES/'bold_mode_progress.png',bbox_inches='tight');plt.show()
"""),
    md("## Pair preference · four completed models"),
    code("""
pair_partial=pd.read_csv(ROOT/'results/summary/pair_completed_four.tsv',sep='\\t')
pair_partial['dataset']=np.where(pair_partial.file.str.contains('crows_full'),'CrowS-Pairs','StereoSet')
pair_partial['model']=pair_partial.file.str.extract(r'_(llama31|llama|llada|dream)\\.jsonl$')[0]
pair_partial=pair_partial.groupby(['dataset','model']).apply(
    lambda g: pd.Series({'n':int(g.n.sum()),
                         'stereotype_preference':np.average(g.stereotype_preference,weights=g.n)}),
    include_groups=False).reset_index()
display(pair_partial)
four=['llama','llada','dream','llama31']
fig,ax=plt.subplots(figsize=(9,4))
for offset,dataset,color in [(-.18,'CrowS-Pairs','#526fad'),(.18,'StereoSet','#e6a24c')]:
    part=pair_partial.query('dataset==@dataset').set_index('model').reindex(four)
    ax.bar(np.arange(4)+offset,part.stereotype_preference,.36,label=dataset,color=color)
ax.axhline(.5,color='black',ls='--',lw=.8)
ax.set_xticks(np.arange(4),[LABEL[m] for m in four])
ax.set_ylim(0,1);ax.set_ylabel('Stereotype preference fraction')
ax.legend(frameon=False)
fig.tight_layout();fig.savefig(FIGURES/'pair_progress.png',bbox_inches='tight');plt.show()
"""),
    full.cells[16], full.cells[17], full.cells[18],
]
notebook = nbf.v4.new_notebook(cells=cells, metadata=full.metadata)
client = NotebookClient(notebook, timeout=1200, kernel_name="dllm-eval",
                        resources={"metadata": {"path": str(ROOT)}})
client.execute()
output = ROOT / "notebooks/completed_runs_so_far.ipynb"
nbf.write(notebook, output)
print(output)
