"""Build Russian narrative and standalone figures from completed saved experiments."""
import sys,json,argparse,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import polars as pl
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from SaTP.common import config,dump,digest
from Data_loader.pipeline import prepare,load
from scripts.frozen_run import frozen_data
from Reward_models.models import SequenceReward,RewardEnsemble
from SaTP.training import load_model,reward_predictions
from SaTP.run import cat_features
from catboost import CatBoostRegressor

LABELS={'B0':'B0: GPT-2','A1':'A1: reranking','A2':'A2: RL','A3':'A3: pessimism','A4':'A4: slate','control_sft':'SFT control'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    parent=(ROOT/args.run).resolve()
    if not parent.is_relative_to(ROOT):raise ValueError('Outside project')
    status=json.loads((parent/'status.json').read_text())
    if status['state']!='completed':raise ValueError('Primary matrix incomplete')
    if not (parent/'ablations_seed42/manifest.json').exists():raise ValueError('Supplementary controls missing')
    if json.loads((parent/'ablations_seed42/manifest.json').read_text())['state']!='completed':raise ValueError('Supplementary controls incomplete')
    source=pl.read_parquet(parent/'summary.parquet')
    columns=['ndcg10','reward','diagnostic_reward','familiar_fraction','mean_duration_seconds','redundancy']
    means={}
    for name in LABELS:
        part=source.filter(pl.col('solution')==name)
        means[name]={col:float(part[col].mean()) for col in columns}
        means[name]['ndcg_std']=float(part['ndcg10'].std() or 0.)
    figures=ROOT/'reports/figures';figures.mkdir(exist_ok=True)
    exported=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    def finish(fig,name):
        path=figures/f'{parent.name}_{name}.png';fig.tight_layout();fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig);exported.append(path)
    fig,ax=plt.subplots(figsize=(8,5))
    baseline=means['B0']['ndcg10'];ax.axvline(.98*baseline,color='gray',ls='--',lw=1,label='98% среднего NDCG baseline')
    for i,(name,row) in enumerate(means.items()):
        ax.errorbar(row['ndcg10'],row['reward'],xerr=row['ndcg_std'],fmt='o',capsize=3)
        ax.annotate(name,(row['ndcg10'],row['reward']),xytext=({'A2':(8,-16),'A3':(8,18),'A4':(8,-14)}.get(name,(6,9))),textcoords='offset points')
    ax.set(xlabel='NDCG@10',ylabel='Средняя оценка RM',title='Награда и релевантность: среднее трёх seeds')
    ax.legend(loc='best',fontsize=8);finish(fig,'tradeoff')
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,col,title in zip(axes.flat,['ndcg10','reward','diagnostic_reward','mean_duration_seconds'],['NDCG@10','Оценка основного RM','Оценка диагностического бустинга (raw)','Средняя длительность, секунды']):
        ax.bar(list(means),[means[n][col] for n in means],color=['#666666','#3478b8','#3f9a75','#9b6abb','#d58a36','#aa5656'])
        ax.set_title(title);ax.tick_params(axis='x',rotation=20)
    finish(fig,'main_metrics')
    behavior=pl.read_parquet(parent/'behavior.parquet')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,col,title in zip(axes,['interuser_jaccard','top_one_percent_share'],['Сходство списков разных пользователей','Доля рекомендаций top 1% популярных треков']):
        values=[float(behavior.filter(pl.col('solution')==name)[col].mean()) for name in means]
        ax.bar(list(means),values,color='#3478b8');ax.set_title(title);ax.tick_params(axis='x',rotation=20);ax.set_ylim(0,max(values)*1.2 if max(values)>0 else 1)
    finish(fig,'concentration')
    ab=pl.read_parquet(parent/'ablations_seed42/summary.parquet')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    names=['A4_H1','A4_H3','A4_H5']
    for ax,col,title in zip(axes,['ndcg10','reward'],['Горизонт и NDCG@10','Горизонт и оценка RM']):
        values=[float(ab.filter(pl.col('name')==name)[col][0]) for name in names]
        ax.plot([1,3,5],values,marker='o');ax.set(xlabel='Горизонт обучения',title=title);ax.set_xticks([1,3,5])
    fig.suptitle('Дополнительный контроль: seed 42, common Next-K');finish(fig,'horizons')
    # Independent logged-action calibration analysis. Never fit a model in the report.
    c,folder=frozen_data(parent);meta=dict(np.load(folder/'metadata.npz'));size=len(np.load(folder/'item_ids.npy'))
    device='cuda' if torch.cuda.is_available() else 'cpu';models=[]
    for i in range(c['training']['rm_members']):
        checkpoint=parent/'reward_models'/f'rm_{i}.pt'
        if not checkpoint.exists():raise ValueError('Missing reward model')
        m=SequenceReward(size).to(device);load_model(checkpoint,m);models.append(m)
    ensemble=RewardEnsemble(models,meta['duration'],meta['first'],device);ensemble.offset=json.loads((parent/'reward_models/calibration.json').read_text())['offset']
    data=load(folder,'T');pred,y=reward_predictions(ensemble,data,device,c['training']['batch_size']);pred=np.clip(pred+ensemble.offset,0,1)
    valid=(data['target']>0)&data['reward_eligible']&np.isfinite(data['reward'])
    diagnostic=CatBoostRegressor();diagnostic.load_model(str(parent/'reward_models/diagnostic.cbm'))
    dp=np.clip(diagnostic.predict(cat_features({k:v[valid] for k,v in data.items()})),0,1)
    fit=load(folder,'F');constant=float(np.mean(fit['reward'][fit['reward_eligible']&np.isfinite(fit['reward'])]))
    quality=dict(rows=len(y),neural_mae=float(np.mean(abs(pred-y))),diagnostic_clipped_mae=float(np.mean(abs(dp-y))),constant_mae=float(np.mean(abs(constant-y))))
    valid_index=np.flatnonzero(valid);uncertainty=[]
    for start in range(0,len(valid_index),c['training']['batch_size']):
        ix=valid_index[start:start+c['training']['batch_size']]
        history=torch.as_tensor(data['history'][ix],device=device,dtype=torch.long)
        actions=torch.as_tensor(data['target'][ix],device=device,dtype=torch.long)
        timestamps=torch.as_tensor(data['timestamp'][ix],device=device)
        _,std=ensemble.score(ensemble.encode(history),actions,timestamps)
        uncertainty.extend(std[:,0].cpu().numpy())
    uncertainty=np.asarray(uncertainty);error=np.abs(pred-y)
    from scipy.stats import spearmanr
    quality['uncertainty_abs_error_spearman']=float(spearmanr(uncertainty,error).statistic)
    quality['uncertainty_low_quintile_mae']=float(error[uncertainty<=np.quantile(uncertainty,.2)].mean())
    quality['uncertainty_high_quintile_mae']=float(error[uncertainty>=np.quantile(uncertainty,.8)].mean())
    quality['uncertainty_note']='Descriptive on observed supported mature test actions; ensemble disagreement is not a confidence bound.'
    dump(parent/'reward_calibration.json',quality)
    bins=[]
    for index,lo in enumerate(np.linspace(0,.9,10)):
        mask=(pred>=lo)&(pred<lo+.1) if index<9 else pred>=lo
        bins.append(dict(bin=float(lo),count=int(mask.sum()),predicted=float(pred[mask].mean()) if mask.any() else None,observed=float(y[mask].mean()) if mask.any() else None))
    pl.DataFrame(bins).write_parquet(parent/'reward_calibration.parquet')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    usable=[r for r in bins if r['count']]
    axes[0].plot([r['predicted'] for r in usable],[r['observed'] for r in usable],marker='o');axes[0].plot([0,1],[0,1],ls='--',color='gray')
    axes[0].set(xlabel='Прогноз доли прослушивания',ylabel='Средний наблюдаемый отклик',title='Калибровка на logged actions')
    axes[1].bar(['RM','Бустинг','Константа'],[quality['neural_mae'],quality['diagnostic_clipped_mae'],quality['constant_mae']],color=['#3478b8','#3f9a75','#888888']);axes[1].set_title('MAE на созревших метках test')
    finish(fig,'reward_quality')
    position_rows=[]
    for name in ['B0','A3','A4']:
        predictions=pl.read_parquet(parent/'seed_42'/f'{name}_nextk_predictions.parquet')
        items=np.asarray(predictions['items'].to_list());mus=[];stds=[]
        for start in range(0,len(items),c['training']['batch_size']):
            ix=np.arange(start,min(start+c['training']['batch_size'],len(items)))
            h=torch.as_tensor(data['history'][ix],device=device,dtype=torch.long)
            a=torch.as_tensor(items[ix],device=device,dtype=torch.long)
            t=torch.as_tensor(data['timestamp'][ix],device=device)
            mu,std=ensemble.score(ensemble.encode(h),a,t)
            mus.append((mu+ensemble.offset).clamp(0,1).cpu().numpy());stds.append(std.cpu().numpy())
        mus=np.concatenate(mus);stds=np.concatenate(stds)
        for position in range(items.shape[1]):
            familiar=np.array([int(a in h[h>0]) for a,h in zip(items[:,position],data['history'])])
            frame=pl.DataFrame(dict(user=data['user'],reward=mus[:,position],uncertainty=stds[:,position],familiar=familiar))
            avg=frame.group_by('user').agg(pl.col('reward').mean(),pl.col('uncertainty').mean(),pl.col('familiar').mean()).select(pl.exclude('user').mean()).to_dicts()[0]
            position_rows.append(dict(solution=name,position=position+1,**avg))
    pl.DataFrame(position_rows).write_parquet(parent/'generation_positions.parquet')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for name in ['B0','A3','A4']:
        part=[row for row in position_rows if row['solution']==name]
        for ax,col in zip(axes,['reward','uncertainty']):ax.plot([r['position'] for r in part],[r[col] for r in part],marker='o',label=name)
    for ax,title in zip(axes,['Оценка RM по позиции','Расхождение ансамбля по позиции']):
        ax.set(xlabel='Позиция предложения в Next-K',title=title);ax.legend()
    fig.suptitle('Seed 42, RM видит исходную реальную историю');finish(fig,'generation_positions')
    import nbformat as nbf
    report=ROOT/'reports'/f'{parent.name}_comparison.ipynb';notebook=nbf.read(report,as_version=4)
    notebook.cells[0]=nbf.v4.new_markdown_cell('# Yambda: сравнение GPT-2 и RL-дообучения\n\nПять основных решений, три seeds, общая временная политика и отдельный контроль дополнительного SFT. Ниже — фактически выполненные результаты ограниченной пользовательской когорты.')
    table=['| Решение | NDCG@10 | Оценка RM | Диагностическая оценка |','|---|---:|---:|---:|']
    table.extend(f"| {name} | {row['ndcg10']:.6f} | {row['reward']:.4f} | {row['diagnostic_reward']:.4f} |" for name,row in means.items())
    text='## Краткая сводка\n\n'+'\n'.join(table)+'\n\nОбласть: 931 пользователь в исходной когорте, 166104 объекта начального каталога, 13749 тестовых запросов от 780 пользователей. Обучающие окна ограничены по пользователям. Это исследование на Yambda-50M, с обучением на подвыборке.\n\nОценка RM — модельный proxy. Проверка реального удержания и причинного эффекта не выполнялась. Буфер созревания меток — консервативное допущение; точное время завершения не предоставлено.\n\nКонтрольные варианты с дополнительным обучением и популярностью служат для оценки силы исходного baseline. Их вычислительные расходы и объём информации различаются.\n\nИнтервалы по пользователям и разброс трёх training seeds показаны отдельно. Дополнительные абляции выполнены на одном seed и имеют описательный статус.'
    selection=json.loads((parent/'selection.json').read_text())
    selected_table=['| Решение | Настройки, выбранные на V | Ограничение −2% на V |','|---|---|---|']
    for name,choice in selection.items():
        settings=', '.join(f'{key}={value}' for key,value in choice['params'].items())
        gate='пройдено' if choice['meets_relevance_constraint'] else 'НЕ пройдено; диагностический вариант'
        selected_table.append(f'| {name} | {settings} | {gate} |')
    comparisons=json.loads((parent/'comparisons.json').read_text())
    tested=['| Решение | Изменение NDCG | 95% CI разницы NDCG | Рост RM и подтверждённая не меньшая релевантность |','|---|---:|---|---|']
    for name,result in comparisons.items():
        relative=result.get('relative_delta')
        change=f'{100*relative:+.2f}%' if relative is not None else 'не определено'
        interval=f"[{result['ci_low']:.6f}; {result['ci_high']:.6f}]"
        passed='да' if result['proxy_gain_with_noninferiority_ci'] else 'не подтверждено'
        tested.append(f'| {name} | {change} | {interval} | {passed} |')
    text+='\n\n### Выбор настроек до test\n\n'+'\n'.join(selected_table)
    text+='\n\n'+ '\n'.join(tested)
    text+='\n\nПорог на V — правило выбора, а пользовательский доверительный интервал на test — отдельная проверка. Непрохождение правила выбора не скрывается. Отсутствие значимого различия не доказывает эквивалентность. A3 с нулевым штрафом совпадает с A2; это не независимое улучшение. Положительные штрафы анализируются отдельно. У A4 выбрано eta=0, поэтому выбранный вариант не подтверждает пользу штрафа за избыточность. В обучении этот штраф использует замороженные embeddings B0; поведенческая метрика redundancy — embeddings первого RM. Они не эквивалентны семантическому разнообразию музыки.'
    sft_gain=means['control_sft']['ndcg10']/means['B0']['ndcg10']-1 if means['B0']['ndcg10'] else 0
    text+=f'\n\nДополнительный SFT изменил средний NDCG относительно B0 на {sft_gain:+.1%}. Это контроль свежести и дополнительного обучения; его бюджет не выровнен с RL по GPU-секундам. Высокий модельный reward сам по себе не показывает улучшения отклика пользователей.'
    text+='\n\n### Несколько шагов и долгосрочная цель\n\nВ Next-K политика получает предыдущие предложения как контекст построения списка. Reward model каждый объект оценивает по исходной реальной истории. Мы не считаем предложения принятыми действиями и не имитируем пользовательский переход. Контроль H=1/3/5 описывает зависимость от горизонта slate-обучения, но не измеряет удержание. Common Next-K унифицирует последовательное построение списка; A1 сохраняет reranking на каждом шаге. Поэтому это контроль части эффекта декодирования, а не полное уравнивание всех алгоритмов выдачи.\n\nДля продолжения исследования подходят вероятность следующего визита и активные дни после рекомендации. Потребуются подтверждённые единицы времени, фиксированный горизонт созревания, обработка цензурирования и отдельная модель пользовательских переходов. Предсказательная проверка на логах возможна; причинный эффект на удержание требует данных об экспозициях и обоснованного offline policy evaluation либо онлайн-эксперимента.'
    cohorts=pl.read_parquet(parent/'cohorts.parquet').filter(pl.col('cohort')=='all')
    def behavior_mean(name,col):return float(cohorts.filter(pl.col('solution')==name)[col].mean())
    native_seconds=behavior_mean('A4','mean_duration_seconds')-behavior_mean('B0','mean_duration_seconds')
    common_seconds=behavior_mean('A4_nextk','mean_duration_seconds')-behavior_mean('B0_nextk','mean_duration_seconds')
    native_repeats=behavior_mean('A4','familiar_fraction')-behavior_mean('B0','familiar_fraction')
    common_repeats=behavior_mean('A4_nextk','familiar_fraction')-behavior_mean('B0_nextk','familiar_fraction')
    text+=f'\n\n### Наблюдаемое поведение и проверки альтернатив\n\nРазница A4−B0 по средней длительности: {native_seconds:+.2f} секунды при native decoding и {common_seconds:+.2f} при common Next-K. Разница доли знакомых объектов: {100*native_repeats:+.2f} и {100*common_repeats:+.2f} процентных пункта соответственно. Различие этих двух сравнений показывает, почему нельзя приписать весь native-эффект одному RL objective. Это описательные средние трёх seeds, без отдельного статистического теста поведения.'
    baseline_next=float(ab.filter(pl.col('name')=='B0_nextk')['ndcg10'][0])
    for control_name in ['popularity_same_anchors','popularity_full_F','positive_sft','auxiliary_sft']:
        control_ndcg=float(ab.filter(pl.col('name')==control_name)['ndcg10'][0])
        text+=f'\n\nКонтроль `{control_name}`: NDCG@10={control_ndcg:.6f}; B0 при том же Next-K на seed42: {baseline_next:.6f}. Популярность по всему F имеет больше обучающей информации, чем ограниченные окна B0, поэтому эти два контроля популярности разделены.' if control_name=='popularity_full_F' else f'\n\nКонтроль `{control_name}`: NDCG@10={control_ndcg:.6f}; B0 при том же Next-K на seed42: {baseline_next:.6f}.'
    popularity=ab.filter(pl.col('name')=='popularity_same_anchors').to_dicts()[0]
    b0next=ab.filter(pl.col('name')=='B0_nextk').to_dicts()[0]
    text+=f"\n\nНеперсонализированный контроль популярности имеет RM-score {popularity['reward']:.4f} против {b0next['reward']:.4f} у B0 Next-K, при NDCG {popularity['ndcg10']:.6f} против {b0next['ndcg10']:.6f}. Его список один для всех пользователей. Этот конкретный контроль показывает, почему рост RM-score нельзя принимать за достаточное доказательство качества рекомендаций."
    text+=f"\n\nMAE RM на наблюдавшихся созревших test-действиях: {quality['neural_mae']:.4f}; константы из F: {quality['constant_mae']:.4f}; независимого бустинга после clipping: {quality['diagnostic_clipped_mae']:.4f}. Корреляция Spearman расхождения ансамбля с абсолютной ошибкой: {quality['uncertainty_abs_error_spearman']:.3f}. Даже положительная корреляция не делает ensemble std доверительной границей и не подтверждает точность на ненаблюдавшихся рекомендациях."
    if quality['neural_mae']>=quality['constant_mae']:
        text+='\n\nНа этом test-срезе нейронный RM не превзошёл простую константу по MAE. Это существенное ограничение качества оценщика: рост его прогнозов после оптимизации особенно требует проверки независимыми метриками и новыми данными.'
    else:
        text+='\n\nНейронный RM превзошёл константу по MAE на наблюдавшихся действиях. Это проверка предсказания внутри логов; перенос на ненаблюдавшиеся рекомендации из неё не следует.'
    notebook.cells.insert(1,nbf.v4.new_markdown_cell(text))
    for cell in notebook.cells:
        if cell.cell_type=='markdown':cell.source=cell.source.replace('Pending extensions: full cross-decoding evaluation, richer behavioral slices, all preregistered ablations, strict completion-availability evidence, and final resource study.','Remaining limitations: restricted cohort, finite training budget, one-seed supplementary controls, uncertain exact completion availability, and no causal retention measurement.')
    notebook.cells.append(nbf.v4.new_markdown_cell('## Графики и качество reward model\n\nНа графике компромисса горизонтальные интервалы показывают стандартное отклонение по трём seeds, не пользовательский доверительный интервал. Калибровка оценивается только на наблюдавшихся действиях, известных каталогу, с созревшими метками.'))
    for path in exported:
        notebook.cells.append(nbf.v4.new_code_cell("from IPython.display import display, Image\ndisplay(Image(filename="+repr(str(path))+"))"))
    notebook.cells.append(nbf.v4.new_code_cell("json.loads((root/'reward_calibration.json').read_text())"))
    notebook.cells.append(nbf.v4.new_markdown_cell('## Фактические ресурсы\n\nВремя основной матрицы включает обучение reward models, поиск настроек и оценку. Переиспользованная B0 и остановленный предварительный прогон указаны отдельно. Память CUDA — максимум allocated, не вся зарезервированная память; пик RAM не измерялся.'))
    notebook.cells.append(nbf.v4.new_code_cell("json.loads((root/'resources.json').read_text())"))
    notebook.cells.append(nbf.v4.new_code_cell("json.loads((root/'attribute_audit.json').read_text())"))
    notebook.metadata['analysis_script_sha256']=digest(__file__)
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[str(ROOT/'.venv/Scripts/python.exe'),'-m','ipykernel_launcher','-f','{connection_file}']
    NotebookClient(notebook,timeout=120,km=km,resources={'metadata':{'path':str(ROOT)}}).execute(cleanup_kc=True);nbf.write(notebook,report)
    shutil.copyfile(report,ROOT/'reports/comparison.ipynb')
    dump(parent/'final_analysis.json',dict(means=means,reward_quality=quality,figures=[str(p.relative_to(ROOT)) for p in exported],analysis_script_sha256=digest(__file__)))
    print('FINAL REPORT',report,flush=True)
if __name__=='__main__':main()
