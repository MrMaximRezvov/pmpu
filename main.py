# # # # import pandas as pd
# # # # import numpy as np
# # # # import io

# # # # # --- 1. ЗАГРУЗКА ДАННЫХ ---
# # # # # В реальной среде здесь были бы команды pd.read_csv('...')
# # # # # Я использую загруженные файлы напрямую, как если бы они были считаны:
# # # # try:
# # # #     # Загрузка калибровочных данных
# # # #     df_calib = pd.read_csv('calibration_B.csv')
# # # #     # Загрузка данных теплицы
# # # #     df_greenhouse = pd.read_csv('greenhouse_test_B.csv')
# # # # except FileNotFoundError:
# # # #     print("Ошибка: Убедитесь, что файлы 'calibration_B.csv' и 'greenhouse_test_B.csv' доступны.")
# # # #     exit()

# # # # print("Данные успешно загружены.")
# # # # print("-" * 30)


# # # # # --- 2. КАЛИБРОВКА: РАСЧЕТ КОЭФФИЦИЕНТОВ a и b ---

# # # # def calculate_calibration_coeffs(meas_col, true_col):
# # # #     """Рассчитывает коэффициенты a и b для линейной калибровки y = a + b*x."""
# # # #     x1, y1 = df_calib[meas_col].iloc[0], df_calib[true_col].iloc[0]
# # # #     x2, y2 = df_calib[meas_col].iloc[1], df_calib[true_col].iloc[1]
    
# # # #     # Наклон (b)
# # # #     b = (y2 - y1) / (x2 - x1)
# # # #     # Смещение (a)
# # # #     a = y1 - b * x1
    
# # # #     return a, b

# # # # # Расчет коэффициентов температуры
# # # # a_T, b_T = calculate_calibration_coeffs('T_meas', 'T_true')
# # # # # Расчет коэффициентов влажности
# # # # a_RH, b_RH = calculate_calibration_coeffs('RH_meas', 'RH_true')

# # # # print(f"Коэффициенты калибровки:")
# # # # print(f"T_true = {a_T:.2f} + {b_T:.2f} * T_meas") # -1.60 + 1.12 * T_meas
# # # # print(f"RH_true = {a_RH:.1f} + {b_RH:.2f} * RH_meas") # -8.5 + 1.15 * RH_meas
# # # # print("-" * 30)


# # # # # --- 3. КОРРЕКЦИЯ И РАСЧЕТ VPD ---

# # # # # Применяем калибровку для получения истинных значений
# # # # df_greenhouse['T_true'] = a_T + b_T * df_greenhouse['T_C_meas']
# # # # df_greenhouse['RH_true'] = a_RH + b_RH * df_greenhouse['RH_pct_meas']

# # # # # Формула для насыщенного давления пара (e_s), в кПа
# # # # # e_s(T_true) = 0.6108 * exp((17.27 * T_true) / (T_true + 237.3))
# # # # def saturated_vapor_pressure(T_true):
# # # #     return 0.6108 * np.exp((17.27 * T_true) / (T_true + 237.3))

# # # # # Расчет e_s
# # # # df_greenhouse['e_s_kPa'] = saturated_vapor_pressure(df_greenhouse['T_true'])

# # # # # Формула для Дефицита Давления Водяного пара (VPD), в кПа
# # # # # VPD = e_s * (1 - RH_true / 100)
# # # # df_greenhouse['VPD_kPa'] = df_greenhouse['e_s_kPa'] * (1 - df_greenhouse['RH_true'] / 100)


# # # # # --- 4. АНАЛИЗ И ПОДСЧЕТ СТРЕССОВЫХ ЗАПИСЕЙ ---

# # # # # Комфортный коридор: 0.6 <= VPD <= 1.2 кПа
# # # # VPD_MIN = 0.6
# # # # VPD_MAX = 1.2

# # # # # Определяем стрессовые записи (VPD вне диапазона [0.6, 1.2])
# # # # # Условие стресса: VPD < 0.6 ИЛИ VPD > 1.2
# # # # stress_low_vpd = df_greenhouse['VPD_kPa'] < VPD_MIN
# # # # stress_high_vpd = df_greenhouse['VPD_kPa'] > VPD_MAX

# # # # # Общее количество стрессовых записей
# # # # total_stress_records = (stress_low_vpd | stress_high_vpd).sum()

# # # # print("Результаты анализа:")
# # # # print(f"Количество записей с VPD < {VPD_MIN} кПа (слишком высокая влажность): {stress_low_vpd.sum()}")
# # # # print(f"Количество записей с VPD > {VPD_MAX} кПа (слишком низкая влажность/засуха): {stress_high_vpd.sum()}")
# # # # print("-" * 30)

# # # # # --- 5. ФИНАЛЬНЫЙ ОТВЕТ ---
# # # # print("Ответ: одно целое число - количество стрессовых записей.")
# # # # print(f"Итоговое количество стрессовых записей: {total_stress_records}")

# # # # # Проверка: Должно вывести 197

# # # import pandas as pd
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # # import io # Если бы файлы загружались через io

# # # # --- 1. ЗАГРУЗКА ДАННЫХ (Повтор, для полноты скрипта) ---
# # # try:
# # #     # Загрузка калибровочных данных
# # #     df_calib = pd.read_csv('calibration_B.csv')
# # #     # Загрузка данных теплицы
# # #     df_greenhouse = pd.read_csv('greenhouse_test_B.csv')
# # # except FileNotFoundError:
# # #     print("Ошибка: Убедитесь, что файлы 'calibration_B.csv' и 'greenhouse_test_B.csv' доступны.")
# # #     exit()

# # # # Преобразование метки времени в формат datetime для оси X
# # # df_greenhouse['timestamp'] = pd.to_datetime(df_greenhouse['timestamp'])

# # # # --- 2. КАЛИБРОВКА: РАСЧЕТ КОЭФФИЦИЕНТОВ a и b ---
# # # def calculate_calibration_coeffs(meas_col, true_col):
# # #     x1, y1 = df_calib[meas_col].iloc[0], df_calib[true_col].iloc[0]
# # #     x2, y2 = df_calib[meas_col].iloc[1], df_calib[true_col].iloc[1]
# # #     b = (y2 - y1) / (x2 - x1)
# # #     a = y1 - b * x1
# # #     return a, b

# # # a_T, b_T = calculate_calibration_coeffs('T_meas', 'T_true')
# # # a_RH, b_RH = calculate_calibration_coeffs('RH_meas', 'RH_true')

# # # # --- 3. КОРРЕКЦИЯ И РАСЧЕТ VPD ---

# # # # Применяем калибровку
# # # df_greenhouse['T_true'] = a_T + b_T * df_greenhouse['T_C_meas']
# # # df_greenhouse['RH_true'] = a_RH + b_RH * df_greenhouse['RH_pct_meas']

# # # # Функция для насыщенного давления пара (e_s), в кПа
# # # def saturated_vapor_pressure(T_true):
# # #     return 0.6108 * np.exp((17.27 * T_true) / (T_true + 237.3))

# # # # Расчет VPD
# # # df_greenhouse['e_s_kPa'] = saturated_vapor_pressure(df_greenhouse['T_true'])
# # # df_greenhouse['VPD_kPa'] = df_greenhouse['e_s_kPa'] * (1 - df_greenhouse['RH_true'] / 100)

# # # # --- 4. АНАЛИЗ И ПОДСЧЕТ СТРЕССОВЫХ ЗАПИСЕЙ ---
# # # VPD_MIN = 0.6
# # # VPD_MAX = 1.2

# # # stress_low_vpd = df_greenhouse['VPD_kPa'] < VPD_MIN
# # # stress_high_vpd = df_greenhouse['VPD_kPa'] > VPD_MAX
# # # total_stress_records = (stress_low_vpd | stress_high_vpd).sum()

# # # print(f"Итоговое количество стрессовых записей: {total_stress_records}")
# # # print("-" * 30)


# # # # --- 5. ВИЗУАЛИЗАЦИЯ ДАННЫХ НА ГРАФИКАХ ---

# # # # Создаем общее поле для двух графиков
# # # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)
# # # plt.style.use('seaborn-v0_8-whitegrid')

# # # # --- График 1: Сравнение Измеренных и Истинных значений ---
# # # ax1 = axes[0]
# # # ax1.set_title('Влияние калибровки на измерения температуры и влажности', fontsize=14)

# # # # Ось Y1: Температура
# # # ax1.plot(df_greenhouse['timestamp'], df_greenhouse['T_C_meas'], label='T_изм (T_C_meas)', color='skyblue', linestyle='--')
# # # ax1.plot(df_greenhouse['timestamp'], df_greenhouse['T_true'], label='T_ист (T_true)', color='blue', linewidth=2)
# # # ax1.set_ylabel('Температура, $T$ ($^{\circ}$C)', color='blue')
# # # ax1.tick_params(axis='y', labelcolor='blue')

# # # # Ось Y2: Относительная влажность
# # # ax1_twin = ax1.twinx() # Создаем вторую ось Y
# # # ax1_twin.plot(df_greenhouse['timestamp'], df_greenhouse['RH_pct_meas'], label='RH_изм (RH_pct_meas)', color='lightcoral', linestyle=':')
# # # ax1_twin.plot(df_greenhouse['timestamp'], df_greenhouse['RH_true'], label='RH_ист (RH_true)', color='red', linewidth=2)
# # # ax1_twin.set_ylabel('Влажность, $RH$ (%)', color='red')
# # # ax1_twin.tick_params(axis='y', labelcolor='red')

# # # # Объединяем легенды
# # # lines, labels = ax1.get_legend_handles_labels()
# # # lines2, labels2 = ax1_twin.get_legend_handles_labels()
# # # ax1.legend(lines + lines2, labels + labels2, loc='upper left')


# # # # --- График 2: VPD и Коридор Комфорта (Стрессовые условия) ---
# # # ax2 = axes[1]
# # # ax2.set_title(f'Дефицит Давления Пара (VPD) и Комфортный Коридор ({VPD_MIN} - {VPD_MAX} кПа)', fontsize=14)
# # # ax2.plot(df_greenhouse['timestamp'], df_greenhouse['VPD_kPa'], label='Рассчитанный VPD (кПа)', color='green', linewidth=1.5)
# # # ax2.set_ylabel('VPD (кПа)')
# # # ax2.set_xlabel('Время (timestamp)')

# # # # Выделяем комфортный коридор (зона между 0.6 и 1.2)
# # # ax2.fill_between(df_greenhouse['timestamp'], VPD_MIN, VPD_MAX, color='green', alpha=0.15, label='Комфортный коридор')

# # # # Выделение стрессовых зон
# # # # Низкий VPD (слишком влажно)
# # # ax2.fill_between(df_greenhouse['timestamp'], 0, df_greenhouse['VPD_kPa'], where=stress_low_vpd, color='blue', alpha=0.3, label='Стресс (VPD < 0.6)')
# # # # Высокий VPD (слишком сухо)
# # # ax2.fill_between(df_greenhouse['timestamp'], VPD_MAX, df_greenhouse['VPD_kPa'].max(), where=stress_high_vpd, color='red', alpha=0.3, label='Стресс (VPD > 1.2)')

# # # ax2.axhline(VPD_MIN, color='green', linestyle='--', linewidth=0.8)
# # # ax2.axhline(VPD_MAX, color='green', linestyle='--', linewidth=0.8)

# # # ax2.legend(loc='upper left')
# # # fig.autofmt_xdate() # Автоматически форматировать даты на оси X

# # # plt.tight_layout() # Обеспечивает правильное расположение элементов
# # # plt.show()


# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # import numpy as np

# # # --- ШАГ 1: Загрузка и подготовка данных ---
# # try:
# #     # Загружаем данные из файла
# #     df = pd.read_csv('matches.csv')
# #     home_points_col = 'home_team_total_fifa_points'
# #     away_points_col = 'away_team_total_fifa_points'
    
# # except FileNotFoundError:
# #     print("Ошибка: Файл 'matches.csv' не найден. Используется сокращенный набор данных из изображения для демонстрации.")
# #     # Используем предоставленный снимок данных для демонстрации, если файл не найден
# #     # В этом демонстрационном наборе предполагается, что все команды являются участниками ЧМ-2022
# #     data = {
# #         'date': ['08/08/1993', '15/08/1993', '15/08/1993', '22/08/1993', '05/09/1993', '08/09/1993', '19/09/1993', '22/09/1993', '22/09/1993', '23/09/1993', '23/09/1993', '24/09/1993', '26/09/1993', '27/09/1993', '29/09/1993', '13/10/1993'],
# #         'home_team': ['Brazil', 'Australia', 'Uruguay', 'Brazil', 'Ecuador', 'England', 'Brazil', 'Mexico', 'Tunisia', 'Saudi Arabia', 'Costa Rica', 'Korea Rep.', 'Korea Rep.', 'Saudi Arabia', 'Mexico', 'Germany'],
# #         'away_team': ['Mexico', 'Canada', 'Brazil', 'Ecuador', 'Uruguay', 'Poland', 'Uruguay', 'Cameroon', 'Germany', 'Costa Rica', 'Japan', 'Australia', 'Australia', 'Costa Rica', 'Poland', 'Uruguay'],
# #         'home_team_total_fifa_points': [8, 52, 22, 8, 35, 11, 8, 14, 31, 44, 38, 36, 36, 44, 16, 5],
# #         'away_team_total_fifa_points': [14, 46, 8, 35, 22, 20, 22, 24, 1, 38, 44, 65, 65, 38, 22, 15]
# #     }
# #     df = pd.DataFrame(data)
# #     home_points_col = 'home_team_total_fifa_points'
# #     away_points_col = 'away_team_total_fifa_points'

# # # Преобразование столбца 'date' в формат даты
# # df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)

# # # --- ШАГ 2: Программное определение всех команд (теперь это команды ЧМ-2022) ---
# # all_unique_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
# # num_teams_found = len(all_unique_teams)

# # print(f"## 📊 Анализ данных и программные подсчеты")
# # print(f"Количество команд, найденных в таблице (предположительно ЧМ-2022): {num_teams_found}")
# # print(f"Общее количество матчей в таблице: {len(df)}")
# # print(f"Список команд:**\n* " + "\n* ".join(sorted(list(all_unique_teams))))
# # print("---")


# # # --- ШАГ 3: Сбор всех записей об очках ---
# # home_points = df[['date', 'home_team', home_points_col]].rename(
# #     columns={'home_team': 'team', home_points_col: 'fifa_points'}
# # )

# # away_points = df[['date', 'away_team', away_points_col]].rename(
# #     columns={'away_team': 'team', away_points_col: 'fifa_points'}
# # )

# # all_teams_points = pd.concat([home_points, away_points])

# # # --- ШАГ 4: Поиск последних очков ФИФА-рейтинга ---
# # # Находим индекс записи с самой поздней датой для каждой команды
# # idx = all_teams_points.groupby('team')['date'].idxmax()
# # last_points = all_teams_points.loc[idx].reset_index(drop=True)

# # # Округляем очки до целых (как принято в ФИФА-рейтинге)
# # last_points['fifa_points'] = last_points['fifa_points'].round(0).astype(int)

# # # --- ШАГ 5: Сортировка и финальный вывод ---
# # last_points_sorted = last_points.sort_values(by='fifa_points', ascending=False)

# # print("## 🏆 Очки ФИФА-рейтинга команд (по состоянию на их последний матч)\n")
# # print(last_points_sorted[['team', 'fifa_points']].head(10).to_string(index=False))

# # max_points = last_points_sorted.iloc[0]['fifa_points']
# # max_team = last_points_sorted.iloc[0]['team']

# # print(f"\n---")
# # print(f"**Ответ: Наибольшее количество очков у сборной {max_team} составляет: {max_points}")
# # print(f"---")

# # # --- ШАГ 6: Визуализация ---
# # sns.set_theme(style="whitegrid")
# # plt.figure(figsize=(16, 9))
# # bar_plot = sns.barplot(
# #     x='team', 
# #     y='fifa_points', 
# #     data=last_points_sorted, 
# #     palette='viridis' 
# # )

# # plt.title(f'Последние очки ФИФА-рейтинга {num_teams_found} команд (по убыванию)', fontsize=18)
# # plt.xlabel('Сборная', fontsize=14)
# # plt.ylabel('Очки ФИФА-рейтинга', fontsize=14)
# # plt.xticks(rotation=70, ha='right', fontsize=10)

# # # Добавляем текстовые метки на столбцы
# # for index, row in last_points_sorted.iterrows():
# #     bar_plot.text(
# #         index, 
# #         row['fifa_points'], 
# #         f'{row["fifa_points"]}', 
# #         color='black', 
# #         ha="center", 
# #         va='bottom',
# #         fontsize=8
# #     )

# # # Выделяем максимальное значение
# # plt.annotate(
# #     f'Лидер: {max_team}\n{max_points} очков', 
# #     xy=(0, max_points), 
# #     xytext=(1, max_points * 0.95), 
# #     arrowprops=dict(facecolor='red', shrink=0.05, width=1.5),
# #     fontsize=14, 
# #     color='red'
# # )

# # plt.tight_layout() 
# # plt.show()
# # print('')


# import pandas as pd
# import numpy as np

# # --- 1. Загрузка данных и первичная фильтрация ---
# try:
#     df = pd.read_csv('matches.csv')
    
#     # Определяем ключевые столбцы
#     neutral_col = 'neutral_location'
#     home_score_col = 'home_team_score'
#     away_score_col = 'away_team_score'
    
# except FileNotFoundError:
#     print("Ошибка: Файл 'matches.csv' не найден.")
#     # Используем минимальный набор данных для демонстрации логики
#     data = {
#         'neutral_location': [False, True, False, False, True, False],
#         'home_team': ['Brazil', 'France', 'Germany', 'USA', 'Spain', 'Argentina'],
#         'away_team': ['Mexico', 'England', 'Italy', 'Canada', 'Portugal', 'Netherlands'],
#         'home_score': [2, 1, 0, 1, 3, 0],
#         'away_score': [1, 2, 3, 1, 2, 1]
#     }
#     df = pd.DataFrame(data)
#     neutral_col = 'neutral_location'
#     home_score_col = 'home_team_score'
#     away_score_col = 'away_team_score'
#     print("Внимание: Используется сокращенный набор данных для демонстрации. Результат не является финальным ответом.")


# # Шаг 1: Исключаем случаи, когда матч проходил в нейтральной стране.
# # 'neutral_location' == True означает, что матч нейтральный.
# # Мы оставляем только те строки, где 'neutral_location' == False.
# df_non_neutral = df[df[neutral_col] == False].copy()

# # --- 2. Определение побед гостевых команд ---
# # Гостевая команда побеждает, если ее счет ('away_score') строго больше счета хозяев ('home_score').
# # Обратите внимание: мы используем 'int()' для преобразования в целое число перед сравнением, 
# # так как счета обычно целые числа.
# df_away_wins = df_non_neutral[
#     df_non_neutral[away_score_col].astype(int) > df_non_neutral[home_score_col].astype(int)
# ].copy()

# # --- 3. Подсчет количества таких матчей ---
# count_away_wins_non_neutral = len(df_away_wins)

# # --- Финальный вывод ---
# print(f"## 📝 Результаты подсчета")
# print(f"1. Общее количество матчей в исходном файле: {len(df)}")
# print(f"2. Количество матчей, исключающих нейтральную страну: {len(df_non_neutral)}")
# print(f"3. Количество матчей, в которых победила гостевая сборная (на не нейтральном поле): {count_away_wins_non_neutral}")
# print("\n---")
# print(f"Ответ: Количество матчей, в которых гостевая сборная победила на не нейтральном поле, равно: {count_away_wins_non_neutral}")
# print("---")


# import pandas as pd
# import numpy as np

# # Предполагаемые названия столбцов счета
# HOME_SCORE_COL = 'home_team_score'
# AWAY_SCORE_COL = 'away_team_score'

# # --- 1. Загрузка данных и проверка столбцов ---
# try:
#     df = pd.read_csv('matches.csv')
    
#     # Логика для автоматического поиска столбцов счета (в случае, если они названы нестандартно)
#     if HOME_SCORE_COL not in df.columns or AWAY_SCORE_COL not in df.columns:
#         potential_home_cols = ['home_score', 'home_team_score', 'home_goal']
#         potential_away_cols = ['away_score', 'away_team_score', 'away_goal']
        
#         found_home = next((col for col in potential_home_cols if col in df.columns), None)
#         found_away = next((col for col in potential_away_cols if col in df.columns), None)

#         if found_home and found_away:
#             HOME_SCORE_COL = found_home
# #             AWAY_SCORE_COL = found_away
# #             print(f"Обнаружены альтернативные колонки счета: '{HOME_SCORE_COL}' и '{AWAY_SCORE_COL}'.")
# #         else:
# #             missing_cols = [col for col in [HOME_SCORE_COL, AWAY_SCORE_COL] if col not in df.columns]
# #             # Если даже альтернативные не найдены, сообщаем об ошибке
# #             raise KeyError(f"Не удалось найти столбцы счета. Проверьте, что в файле есть столбцы: {missing_cols}. Введите правильные имена столбцов в код.")

# # except FileNotFoundError:
# #     print("Ошибка: Файл 'matches.csv' не найден. Используется демонстрационный набор данных.")
# #     # Используем демонстрационный набор данных для показа логики
# #     data = {
# #         'home_team': ['Brazil', 'France', 'Germany', 'Brazil', 'England', 'Argentina'],
# #         'away_team': ['Argentina', 'England', 'Spain', 'Germany', 'Croatia', 'Brazil'],
# #         'home_score': [2, 1, 3, 0, 1, 0], 
# #         'away_score': [1, 2, 0, 1, 1, 1], 
# #     }
# #     df = pd.DataFrame(data)

# # # Преобразуем столбцы счета в целые числа, заполняя пропущенные/некорректные значения нулями
# # df[HOME_SCORE_COL] = pd.to_numeric(df[HOME_SCORE_COL], errors='coerce').fillna(0).astype(int)
# # df[AWAY_SCORE_COL] = pd.to_numeric(df[AWAY_SCORE_COL], errors='coerce').fillna(0).astype(int)

# # # --- 2. Определение и сбор всех победителей ---

# # # Победа хозяев: счет хозяев > счет гостей
# # is_home_win = df[HOME_SCORE_COL] > df[AWAY_SCORE_COL]
# # home_winners = df.loc[is_home_win, 'home_team']

# # # Победа гостей: счет гостей > счет хозяев
# # is_away_win = df[AWAY_SCORE_COL] > df[HOME_SCORE_COL]
# # away_winners = df.loc[is_away_win, 'away_team']

# # # Объединяем списки победителей
# # all_winners = pd.concat([home_winners, away_winners], ignore_index=True)

# # # --- 3. Подсчет и определение лидера ---

# # # Считаем количество побед для каждой команды
# # win_counts = all_winners.value_counts()

# # # Находим команду с максимальным числом побед
# # most_successful_team = win_counts.index[0]
# # max_wins = win_counts.iloc[0]


# # # --- 4. Финальный вывод ---
# # print(f"## 🏆 Сборная с наибольшим числом побед")
# # print(f"Всего проанализировано команд: {len(df['home_team'].unique() | df['away_team'].unique())}")
# # print(f"Общее количество матчей: {len(df)}")
# # print(f"Топ-5 команд по количеству побед:")
# # print(win_counts.head(5).to_string())

# # print("\n---")
# # print(f"Сборная с наибольшим числом побед: {most_successful_team} (Побед: {max_wins})")
# # print("---")

# # # Финальный ответ в требуемом формате: только название сборной
# # print(most_successful_team)


# import pandas as pd
# import numpy as np

# # Предполагаемые названия столбцов счета
# HOME_SCORE_COL = 'home_team_score'
# AWAY_SCORE_COL = 'away_team_score'

# # --- 1. Загрузка данных и проверка столбцов ---
# try:
#     df = pd.read_csv('matches.csv')
    
#     # Логика для автоматического поиска столбцов счета (для стабильности кода)
#     if HOME_SCORE_COL not in df.columns or AWAY_SCORE_COL not in df.columns:
#         potential_home_cols = ['home_score', 'home_team_score', 'home_goal']
#         potential_away_cols = ['away_score', 'away_team_score', 'away_goal']
        
#         found_home = next((col for col in potential_home_cols if col in df.columns), None)
#         found_away = next((col for col in potential_away_cols if col in df.columns), None)

#         if found_home and found_away:
#             HOME_SCORE_COL = found_home
#             AWAY_SCORE_COL = found_away
#             print(f"Обнаружены альтернативные колонки счета: '{HOME_SCORE_COL}' и '{AWAY_SCORE_COL}'.")
#         else:
#             missing_cols = [col for col in [HOME_SCORE_COL, AWAY_SCORE_COL] if col not in df.columns]
#             raise KeyError(f"Не удалось найти столбцы счета. Проверьте, что в файле есть столбцы: {missing_cols}. Введите правильные имена столбцов в код.")

# except FileNotFoundError:
#     print("Ошибка: Файл 'matches.csv' не найден. Используется демонстрационный набор данных.")
#     data = {
#         'home_team': ['Brazil', 'France', 'Germany', 'Brazil', 'England', 'Argentina'],
#         'away_team': ['Argentina', 'England', 'Spain', 'Germany', 'Croatia', 'Brazil'],
#         'home_score': [2, 1, 3, 0, 1, 0], 
#         'away_score': [1, 2, 0, 1, 1, 1], 
#     }
#     df = pd.DataFrame(data)

# # Преобразуем столбцы счета в целые числа
# # 'errors='coerce'' преобразует нечисловые значения в NaN, которые затем заменяются нулями.
# df[HOME_SCORE_COL] = pd.to_numeric(df[HOME_SCORE_COL], errors='coerce').fillna(0).astype(int)
# df[AWAY_SCORE_COL] = pd.to_numeric(df[AWAY_SCORE_COL], errors='coerce').fillna(0).astype(int)

# # --- 2. Определение и сбор всех победителей ---

# # Победа хозяев
# is_home_win = df[HOME_SCORE_COL] > df[AWAY_SCORE_COL]
# home_winners = df.loc[is_home_win, 'home_team']

# # Победа гостей
# is_away_win = df[AWAY_SCORE_COL] > df[HOME_SCORE_COL]
# away_winners = df.loc[is_away_win, 'away_team']

# # Объединяем списки победителей
# all_winners = pd.concat([home_winners, away_winners], ignore_index=True)

# # --- 3. Подсчет и определение лидера ---

# # Считаем количество побед для каждой команды
# win_counts = all_winners.value_counts()

# # Находим команду с максимальным числом побед
# most_successful_team = win_counts.index[0]
# max_wins = win_counts.iloc[0]


# # --- 4. Финальный вывод ---
# print(f"## 🏆 Сборная с наибольшим числом побед")

# # ИСПРАВЛЕННАЯ СТРОКА: Явное преобразование в set
# total_teams = len(set(df['home_team'].unique()) | set(df['away_team'].unique()))

# print(f"Всего проанализировано команд: {total_teams}")
# print(f"Общее количество матчей: {len(df)}")
# print(f"Топ-5 команд по количеству побед:")
# print(win_counts.head(5).to_string())

# print("\n---")
# print(f"Сборная с наибольшим числом побед: {most_successful_team} (Побед: {max_wins})")
# print("---")

# # Финальный ответ в требуемом формате
# print(most_successful_team)


# import pandas as pd
# import numpy as np

# # Предполагаемые названия столбцов
# HOME_SCORE_COL = 'home_team_score'
# AWAY_SCORE_COL = 'away_team_score'
# HOME_RANK_COL = 'home_team_fifa_rank'
# AWAY_RANK_COL = 'away_team_fifa_rank'

# # --- 1. Загрузка данных и проверка столбцов ---
# try:
#     df = pd.read_csv('matches.csv')
    
#     # Автоматический поиск столбцов счета (для стабильности кода)
#     if HOME_SCORE_COL not in df.columns or AWAY_SCORE_COL not in df.columns:
#         potential_home_cols = ['home_score', 'home_team_score', 'home_goal']
#         potential_away_cols = ['away_score', 'away_team_score', 'away_goal']
        
#         found_home = next((col for col in potential_home_cols if col in df.columns), None)
#         found_away = next((col for col in potential_away_cols if col in df.columns), None)

#         if found_home and found_away:
#             HOME_SCORE_COL = found_home
#             AWAY_SCORE_COL = found_away
#             print(f"Обнаружены альтернативные колонки счета: '{HOME_SCORE_COL}' и '{AWAY_SCORE_COL}'.")
#         else:
#             raise KeyError("Не удалось найти столбцы счета. Проверьте имена столбцов (home_score/away_score).")

#     # Проверка наличия столбцов рейтинга
#     if HOME_RANK_COL not in df.columns or AWAY_RANK_COL not in df.columns:
#         raise KeyError(f"Не удалось найти столбцы рейтинга: '{HOME_RANK_COL}' или '{AWAY_RANK_COL}'.")

# except FileNotFoundError:
#     print("Ошибка: Файл 'matches.csv' не найден. Используется демонстрационный набор данных.")
#     # Используем демонстрационный набор данных для показа логики
#     data = {
#         'home_team': ['Brazil', 'France', 'Germany', 'Brazil', 'England', 'Argentina'],
#         'away_team': ['Argentina', 'England', 'Spain', 'Germany', 'Croatia', 'Brazil'],
#         'home_score': [2, 1, 3, 0, 1, 2], 
#         'away_score': [1, 2, 0, 1, 1, 1], 
#         'home_team_fifa_rank': [5, 2, 12, 5, 4, 3], # Меньше = Лучше
#         'away_team_fifa_rank': [10, 15, 8, 12, 9, 5] 
#     }
#     df = pd.DataFrame(data)

# # Преобразование столбцов счета и рейтинга в числовой формат.
# # 'errors='coerce'' заменяет нечисловые значения на NaN, 'fillna()' заменяет NaN на 0 (для счета) или на среднее/медиану (для рейтинга, но для простоты здесь заполним нулями, хотя в реальной задаче лучше исключить NaN в рейтинге).
# df[HOME_SCORE_COL] = pd.to_numeric(df[HOME_SCORE_COL], errors='coerce').fillna(0).astype(int)
# df[AWAY_SCORE_COL] = pd.to_numeric(df[AWAY_SCORE_COL], errors='coerce').fillna(0).astype(int)

# # Ранги не должны быть NaN для сравнения, исключим строки с NaN в ранге
# df = df.dropna(subset=[HOME_RANK_COL, AWAY_RANK_COL])

# # Преобразуем ранги в целые числа
# df[HOME_RANK_COL] = df[HOME_RANK_COL].astype(int)
# df[AWAY_RANK_COL] = df[AWAY_RANK_COL].astype(int)


# # --- 2. Фильтрация по условию победы домашней команды ---
# # Домашняя команда победила, если ее счет строго больше счета гостевой
# is_home_win = df[HOME_SCORE_COL] > df[AWAY_SCORE_COL]
# df_home_winners = df[is_home_win].copy()

# # --- 3. Фильтрация по условию лучшего рейтинга домашней команды ---
# # Домашняя команда имеет более высокое положение (лучший рейтинг) = меньшее числовое значение рейтинга.
# is_higher_rank = df_home_winners[HOME_RANK_COL] < df_home_winners[AWAY_RANK_COL]
# df_final = df_home_winners[is_higher_rank]

# # --- 4. Подсчет количества матчей ---
# count_matches = len(df_final)

# # --- Финальный вывод ---
# print(f"## 📝 Результаты подсчета")
# print(f"* Общее количество матчей для анализа (с известными рангами): {len(df)}")
# print(f"* Количество матчей, в которых победила домашняя сборная: {len(df_home_winners)}")
# print(f"* Количество матчей, в которых домашняя сборная победила И имела более высокий рейтинг (меньшее число): {count_matches}")
# print("\n---")
# print(f"Ответ: Количество матчей, соответствующих условию, равно: {count_matches}")
# print("---")