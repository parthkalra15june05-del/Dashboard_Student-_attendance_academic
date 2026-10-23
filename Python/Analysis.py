"""Reproduce the UCI analysis offline. Run: python 03_Python/Analysis.py

Original_Dataset.csv is the unchanged, semicolon-delimited Mathematics file.
Original_Portuguese.csv is the unchanged Portuguese file. IDs identify course
records, not unique people across subjects. No attendance percentage is derived.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '01_Dataset'
OUT = ROOT / '04_Visualizations' / 'All_Charts'
STATS = ROOT / '03_Python' / 'Outputs'
STUDY = {1:'<2 hours', 2:'2-5 hours', 3:'5-10 hours', 4:'>10 hours'}
NUM = ['age','Medu','Fedu','traveltime','studytime','failures','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2','G3']
RANGES = {'age':(15,22), 'Medu':(0,4), 'Fedu':(0,4), 'traveltime':(1,4), 'studytime':(1,4), 'failures':(0,4), 'absences':(0,93), **{x:(1,5) for x in ['famrel','freetime','goout','Dalc','Walc','health']}, **{x:(0,20) for x in ['G1','G2','G3']}}
CATS = {'school':['GP','MS'],'sex':['F','M'],'address':['U','R'],'famsize':['LE3','GT3'],'Pstatus':['T','A'],'Mjob':['teacher','health','services','at_home','other'],'Fjob':['teacher','health','services','at_home','other'],'reason':['home','reputation','course','other'],'guardian':['mother','father','other'], **{x:['yes','no'] for x in ['schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']}}
PURPLE, BLUE, INK = '#7455D8', '#3184E8', '#202842'

def clean(filename, subject, prefix):
    raw = pd.read_csv(DATA / filename, sep=';')
    d = raw.copy()
    for c in CATS:
        d[c] = d[c].astype('string').str.strip()
    d = d.replace('', pd.NA)
    missing = int(d.isna().sum().sum())
    # Fail visibly on missing/invalid values. Never silently impute final marks.
    if missing:
        raise ValueError(f'{filename}: {missing} null cells; investigate source before proceeding')
    for c in NUM:
        v = pd.to_numeric(d[c], errors='raise')
        if not (v % 1 == 0).all(): raise ValueError(f'Non-integer values: {c}')
        d[c] = v.astype('int64')
        if not d[c].between(*RANGES[c]).all(): raise ValueError(f'Out-of-range values: {c}')
    for c, allowed in CATS.items():
        if not d[c].isin(allowed).all(): raise ValueError(f'Invalid categories: {c}')
    duplicates = int(d.duplicated().sum())
    # No stable person key: exact matching profiles are not automatically duplicates.
    if duplicates: raise ValueError('Exact repeated profiles need manual provenance review')
    q1,q3 = d.absences.quantile([.25,.75]); fence = float(q3+1.5*(q3-q1))
    d.insert(0,'Record_ID',[f'{prefix}-{i:04d}' for i in range(1,len(d)+1)])
    d.insert(1,'Subject',subject)
    d['Pass_Status'] = np.where(d.G3>=10,'Pass','Fail')
    d['Performance_Category'] = pd.cut(d.G3,[-1,9,11,13,15,20],labels=['Fail (0-9)','Satisfactory (10-11)','Good (12-13)','Very good (14-15)','Excellent (16-20)']).astype(str)
    d['Absence_Group'] = pd.cut(d.absences,[-1,0,5,10,93],labels=['0','1-5','6-10','11+']).astype(str)
    d['Absence_Group_Order'] = d.Absence_Group.map({'0':0,'1-5':1,'6-10':2,'11+':3})
    d['Study_Time_Category'] = d.studytime.map(STUDY)
    d['High_Absence_Flag'] = (d.absences>fence).astype(int)
    d['Grade_Change'] = d.G3-d.G1
    log = {'subject':subject,'rows_in':len(raw),'rows_out':len(d),'null_cells':missing,'exact_duplicate_rows':duplicates,'invalid_values':0,'absences_iqr_upper_fence':fence,'high_absence_records_retained':int(d.High_Absence_Flag.sum()),'zero_final_grades_retained':int((d.G3==0).sum()),'original_sha256':hashlib.sha256((DATA/filename).read_bytes()).hexdigest()}
    return d,log

def metrics(d):
    a=d[['absences','studytime','failures','G1','G2','G3']]
    nonzero=d[d.G3>0]
    return {'n':len(d),'mean':float(d.G3.mean()),'median':float(d.G3.median()),'sd':float(d.G3.std()),'pass_n':int((d.G3>=10).sum()),'fail_n':int((d.G3<10).sum()),'pass_pct':float((d.G3>=10).mean()*100),'mean_absences':float(d.absences.mean()),'median_absences':float(d.absences.median()),'max_absences':int(d.absences.max()),'pearson':a.corr()['G3'].to_dict(),'spearman':a.rank().corr()['G3'].to_dict(),'nonzero_n':len(nonzero),'nonzero_absence_r':float(nonzero.absences.corr(nonzero.G3)),'study_groups':d.groupby('studytime').agg(n=('G3','size'),mean=('G3','mean')).reset_index().to_dict('records'),'absence_groups':d.groupby('Absence_Group',sort=False).agg(n=('G3','size'),mean=('G3','mean'),pass_pct=('G3',lambda s:(s>=10).mean()*100)).reset_index().to_dict('records')}

def charts(mat,por):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'axes.titleweight':'bold','axes.titlesize':15,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
    def save(fig,name):
        fig.text(.02,.012,'Source: UCI Student Performance | Mathematics unless otherwise noted | Grades 0-20',fontsize=8,color='#697389')
        fig.tight_layout(rect=[0,.045,1,1]); fig.savefig(OUT/name,dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.6)); vals=[int((mat.G3>=10).sum()),int((mat.G3<10).sum())]
    bars=ax.bar(['Pass (G3 >= 10)','Fail (G3 < 10)'],vals,color=[BLUE,PURPLE],width=.5);ax.bar_label(bars,padding=5);ax.set_ylim(0,310);ax.set_ylabel('Students in Mathematics');ax.set_title('01  Pass and fail distribution');save(fig,'01_Pass_Fail.png')
    fig,ax=plt.subplots(figsize=(9,4.6));ax.hist(mat.G3,bins=np.arange(-.5,21,1),color=PURPLE,edgecolor='white');ax.axvline(9.5,color=BLUE,ls='--',label='Pass boundary');ax.set(xticks=range(0,21,2),xlabel='Final grade G3 (0-20)',ylabel='Students',title='02  Final grade distribution');ax.legend();save(fig,'02_Grade_Distribution.png')
    fig,ax=plt.subplots(figsize=(9,4.6));ax.scatter(mat.absences,mat.G3,c=BLUE,alpha=.4,s=32);ax.set(xlabel='Number of recorded absences',ylabel='Final grade G3 (0-20)',title=f'03  Absences and final grades | Pearson r = {mat.absences.corr(mat.G3):.3f}');save(fig,'03_Absences_Marks.png')
    fig,ax=plt.subplots(figsize=(9,4.6));g=mat.groupby('studytime').G3.agg(['mean','count']);bars=ax.bar([STUDY[i] for i in g.index],g['mean'],color=[PURPLE,BLUE,PURPLE,BLUE]);ax.bar_label(bars,labels=[f'{v:.2f}\nn={n}' for v,n in zip(g['mean'],g['count'])],padding=4);ax.set(ylim=(0,16),xlabel='Reported weekly study time',ylabel='Mean final grade (0-20)',title='04  Study time and performance');save(fig,'04_Study_Time.png')
    cols=['absences','studytime','failures','G1','G2','G3'];corr=mat[cols].corr();fig,ax=plt.subplots(figsize=(8,6));im=ax.imshow(corr,vmin=-1,vmax=1,cmap='PuOr');ax.set_xticks(range(6),cols);ax.set_yticks(range(6),cols)
    for i in range(6):
        for j in range(6):ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',color='white' if abs(corr.iloc[i,j])>.65 else INK)
    fig.colorbar(im,ax=ax,label='Pearson r');ax.set_title('05  Correlation matrix');save(fig,'05_Correlation_Heatmap.png')
    fig,ax=plt.subplots(figsize=(9,4.6));g=mat.groupby('Absence_Group').G3.agg(['mean','count']).reindex(['0','1-5','6-10','11+']);bars=ax.bar(g.index,g['mean'],color=[PURPLE,BLUE,PURPLE,BLUE]);ax.bar_label(bars,labels=[f'{v:.2f}\nn={n}' for v,n in zip(g['mean'],g['count'])],padding=4);ax.set(ylim=(0,16),xlabel='Recorded absence group',ylabel='Mean final grade (0-20)',title='06  Performance by absence group');save(fig,'06_Absence_Groups.png')
    fig,axs=plt.subplots(1,2,figsize=(10,4.7));
    for ax,c in zip(axs,['G1','G2']):
        ax.scatter(mat[c],mat.G3,s=25,alpha=.35,color=PURPLE);ax.plot([0,20],[0,20],ls='--',color='#8A92A8');ax.set(xlabel=f'{c} grade (0-20)',ylabel='G3 final grade (0-20)',title=f'{c} vs G3 | r={mat[c].corr(mat.G3):.3f}',xlim=(-1,21),ylim=(-1,21))
    fig.suptitle('07  Previous grades and final grades',fontweight='bold');save(fig,'07_Previous_Final.png')
    fig,ax=plt.subplots(figsize=(9,4.6));vals=[mat.G3.mean(),por.G3.mean()];bars=ax.bar(['Mathematics (n=395)','Portuguese (n=649)'],vals,color=[PURPLE,BLUE],width=.5);ax.bar_label(bars,fmt='%.2f',padding=5);ax.set(ylim=(0,16),ylabel='Mean final grade (0-20)',title='08  Subject cohort comparison');ax.text(.5,.88,'Different, overlapping cohorts; not a paired subject-effect estimate',transform=ax.transAxes,ha='center',fontsize=9);save(fig,'08_Subject_Comparison.png')
    fig,ax=plt.subplots(figsize=(9,4.6));ax.boxplot([mat[mat.studytime==i].G3 for i in range(1,5)],tick_labels=list(STUDY.values()),patch_artist=True,boxprops={'facecolor':'#DDD5F6'},medianprops={'color':PURPLE,'linewidth':2});ax.set(xlabel='Reported weekly study time',ylabel='Final grade G3 (0-20)',title='09  Grade spread within study-time groups');save(fig,'09_Study_Time_Boxplot.png')

def main():
    OUT.mkdir(parents=True,exist_ok=True);STATS.mkdir(parents=True,exist_ok=True)
    mat,lm=clean('Original_Dataset.csv','Mathematics','MAT')
    por,lp=clean('Original_Portuguese.csv','Portuguese','POR')
    mat.to_csv(DATA/'Cleaned_Dataset.csv',index=False)
    por.to_csv(DATA/'Cleaned_Portuguese.csv',index=False)
    pd.concat([mat,por],ignore_index=True).to_csv(DATA/'All_Subject_Records.csv',index=False)
    for name,d in [('Mathematics',mat),('Portuguese',por)]:
        d.describe(include='all').to_csv(STATS/f'{name}_Descriptive_Statistics.csv')
        d[NUM].corr().to_csv(STATS/f'{name}_Pearson_Correlations.csv')
    result={'Mathematics':metrics(mat),'Portuguese':metrics(por),'cleaning':[lm,lp]}
    (STATS/'Analysis_Results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (STATS/'Cleaning_Log.json').write_text(json.dumps([lm,lp],indent=2),encoding='utf-8')
    charts(mat,por)
    import runpy
    runpy.run_path(str(ROOT/'03_Python/Build_Dashboard.py'), run_name='__main__')
    print(json.dumps(result,indent=2));print('Completed: 9 PNG charts, 3 cleaned CSVs, descriptive statistics and cleaning logs.')

if __name__=='__main__':main()
