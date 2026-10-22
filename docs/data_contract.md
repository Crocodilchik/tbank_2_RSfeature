# Контракт данных

Raw: неизменённый multi_event.parquet и manifest(repo,file,revision,sha256,bytes).
Processed: F/C/R/V/T.npz, item_ids.npy, metadata.npz, manifest.json, raw_profile.json.

Модельная строка: history int32[N,L] (0 PAD,1..V catalog,V+1 UNK), target int64[N] (0 OOV), reward float32[N] (наблюдаемый clipped ratio, NaN при отсутствии), user int64[N], timestamp int64[N], duration float32[N] (log1p(seconds)/10,0 unknown), original_target int64[N]. Последние target/reward/original_target — supervision/evaluator, не автоматические входы policy.

Отдельные arrays target/reward никогда не включаются в history. Inference policy получает только history. RM получает history,target candidate,duration. Непрослушанные пары не маркируются отказами. Catalog не расширяется на C/R/V/T.

Предсказание: query,user,timestamp,target,original_target,items(list),reward,uncertainty,coverage,ndcg10,recall5,recall10; дополнительные диагностические поля. Одинаковые query keys и targets обязательны во всех моделях и seeds. Common-Next-K predictions хранятся отдельно.

Время хранится в исходной шкале. Точный epoch и calendar conversion не утверждаются. Отбор eval — разные timestamp на пользователя, затем равномерное ограничение max_eval_per_user. Статистика сначала per-user. Стабильная hash-когорта определяется user-ID и константой, без будущей активности.

Хеши: raw checksum; processed config+pipeline source; experiment config+полный отпечаток основных модулей. При изменении кода не использовать старый experiment directory как новый запуск. Восстановление только из доверенных локальных checkpoints.
## Версия 2

Добавлены reward_available_at:int64[N] и reward_eligible:bool[N]. reward_eligible означает созревание метки до конца соответствующего блока. Это не feature. Доля прослушивания, её доступность и признак созревания не поступают в GPT-2. RM и calibration исключают неготовые labels; next-item ID targets остаются прежними.
