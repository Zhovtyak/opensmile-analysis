import tkinter as tk
import os
import audiowork as aw
import numpy as np
import pyaudio
import wave
import threading
import matplotlib.pyplot as plt
from const import (RESULTS_ROOT_PATH, CONF, CHUNK,
                   FORMAT, CHANNELS, RATE, RECORD_SECONDS)
from tkinter import filedialog, ttk, Toplevel, PhotoImage
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.fftpack import fft


audiofiles_root_path = None
df = None
directory_for_save = None


def open_root_window():
    # Скрываем главное окно
    main_window.withdraw()
    # Показываем окно root
    root.deiconify()


def close_root_window():
    # Скрываем окно root
    root.withdraw()
    # Показываем главное окно
    main_window.deiconify()


def replace_values(row):
    if row == 0:
        return 'Счастливый'
    elif row == 1:
        return 'Грустный'
    elif row == 2:
        return 'Нейтральный'
    elif row == 3:
        return 'Агрессивный'


def choose_directory():
    '''Выбор директории'''
    directory = filedialog.askdirectory()
    directory += '/'

    if directory:
        label_directory.config(text=f"Выбранная директория: {directory}")
        display_wav_files(directory)
    global audiofiles_root_path
    audiofiles_root_path = directory


def display_wav_files(directory):
    '''Отображения файлов .wav в указанной директории'''
    files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    listbox.delete(0, tk.END)

    for file in files:
        listbox.insert(tk.END, file)


def dataframe_window():
    data_window = tk.Toplevel(root)
    data_window.title("Анализ файлов")
    data_window.geometry("400x600")

    tree = ttk.Treeview(data_window)
    df_new = df[['name', 'class']]
    df_new['class'] = df_new['class'].apply(replace_values)
    df_new.columns = ['Название файла', 'Эмоция']

    tree["columns"] = list(df_new.columns)
    tree["show"] = "headings"

    for column in df_new.columns:
        tree.heading(column, text=column)
        tree.column(column, anchor='center')

    for index, row in df_new.iterrows():
        tree.insert("", "end", values=list(row))

    tree.pack(expand=True, fill='both')


def statistic_window():
    statistic_window = Toplevel(root)
    statistic_window.title("Общая статистика анализа")
    statistic_window.geometry("500x600")
    statistic_window.configure(bg='#4a6fa5')

    df_new = df[['name', 'class']]
    df_new['class'] = df_new['class'].apply(replace_values)
    class_counts = df_new['class'].value_counts()

    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(class_counts, autopct='%1.1f%%',
                                      startangle=90)
    ax.axis('equal')
    ax.legend(wedges, class_counts.index, title="Классификация эмоций",
              loc='upper left',
              bbox_to_anchor=(1, 1))

    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=statistic_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

    label = tk.Label(statistic_window,
                     text=f"Всего обработано файлов: {len(df)}",
                     font=label_font, bg='#DCDCDC', fg='#000000')
    label.pack(pady=10)


def start_analysis():
    global label_error
    files_to_analize = aw.get_wav_files(audiofiles_root_path)
    if len(files_to_analize) == 0:
        if not label_error:
            label_error = tk.Label(root,
                                   text="Ошибка! В директории нет файлов .wav",
                                   font=label_font, bg='#DCDCDC', fg='#900020')
            label_error.pack()
        return 1
    else:
        if label_error:
            label_error.grid_remove()
            label_error = None

    aw.opensmile_analysis(files_to_analize, CONF, audiofiles_root_path,
                          RESULTS_ROOT_PATH)
    global df
    df = (aw.forming_dataframe(files_to_analize, RESULTS_ROOT_PATH))
    df = aw.cnn_prediction(df)
    print('Fineshed analysis!')
    dataframe_window()
    statistic_window()


def choose_directory_save():
    '''Выбор директории для сохранения'''
    global directory_for_save
    directory = filedialog.askdirectory()
    if directory:
        directory_for_save = directory
        filename = entry.get() or "output.csv"
        filepath = f"{directory}/{filename}.csv"
        label_directory_save.config(
            text=f"Выбранная директория и файл: {filepath}")


def save_dataframe():
    '''Сохранение dataframe в CSV файл'''
    global directory_for_save
    filename = entry.get() or "output.csv"

    if not filename.endswith(".csv"):
        filename += ".csv"

    if directory_for_save:
        filepath = f"{directory_for_save}/{filename}"
        df.to_csv(filepath, index=False)
        label_directory_save.config(text=f"Файл сохранён здесь: {filepath}")
    else:
        label_directory_save.config(text="Ошибка: Директория не выбрана")


class EmotionClassificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Классификация эмоций")

        self.label_top = tk.Label(root,
                                  text="Классификация эмоций в real-time",
                                  font=label_font)
        self.label_top.pack()

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  frames_per_buffer=CHUNK)

        self.is_recording = True
        self.update_spectrum()

        self.label_bottom = tk.Label(root,
                                     text="Говорящий испытывает эмоцию:",
                                     font=label_font)
        self.label_bottom.pack()

        self.root.update_idletasks()
        self.record_audio_periodically()

    def update_spectrum(self):
        if self.is_recording:
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            magnitude_spectrum = (
                np.abs(fft(audio_data))[:CHUNK // 2] * 2.0 / CHUNK)
            frequency = np.fft.fftfreq(CHUNK, 1.0 / RATE)[:CHUNK // 2]

            self.ax.clear()
            self.ax.set_facecolor('white')
            self.ax.plot(frequency, 20 * np.log10(magnitude_spectrum),
                         color='blue', linestyle='-', linewidth=1.5)

            self.ax.set_ylim(0, 100)
            self.ax.set_xlim(0, max(frequency))
            self.ax.set_xlabel('Частота, Гц', fontsize=12)
            self.ax.set_ylabel('Амплитуда, дБ', fontsize=12)

            self.ax.grid(True, which='both', linestyle='--', linewidth=0.5)

            self.ax.tick_params(axis='both', which='major', labelsize=10)
            self.canvas.draw()
            self.root.after(50, self.update_spectrum)

    def record_audio(self):
        if self.is_recording:
            frames = []
            for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = self.stream.read(CHUNK)
                frames.append(data)

            wave_file = wave.open(os.path.join('logs/', "recording.wav"), 'wb')
            wave_file.setnchannels(CHANNELS)
            wave_file.setsampwidth(self.p.get_sample_size(FORMAT))
            wave_file.setframerate(RATE)
            wave_file.writeframes(b''.join(frames))
            wave_file.close()

            audio_to_analize = aw.get_wav_files('logs/')
            aw.opensmile_analysis(audio_to_analize, CONF, 'logs/', 'logs/')
            df = (aw.forming_dataframe(audio_to_analize, 'logs/'))
            df = aw.cnn_prediction(df)
            df['class'] = df['class'].apply(replace_values)
            emotion = df.at[0, 'class']

            self.label_bottom.config(text=f"Говорящий испытывает эмоцию: \
                                     {emotion}")

    def record_audio_periodically(self):
        if self.is_recording:
            thread = threading.Thread(target=self.record_audio)
            thread.start()
            self.root.after(5000, self.record_audio_periodically)

    def on_closing(self):
        self.is_recording = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.root.destroy()


def open_real_time_classification_window():
    real_time_classification_window = tk.Toplevel()
    app = EmotionClassificationApp(real_time_classification_window)
    real_time_classification_window.protocol("WM_DELETE_WINDOW",
                                             app.on_closing)


# Применение шрифтов и стилей
label_font = ("Helvetica", 12, "bold")
button_font = ("Helvetica", 12)
listbox_font = ("Helvetica", 12)

# Главное окно
main_window = tk.Tk()
main_window.title("Главное меню")
main_window.geometry("800x600")
main_window.configure(bg='#DCDCDC')

# Лого главного окна
logo = PhotoImage(file="logo.png")
logo_label = tk.Label(main_window, image=logo)
logo_label.pack(pady=10)

# Текст названия программы
label_title = tk.Label(main_window, text="AUDIOWORK",
                       font=label_font, bg='#DCDCDC', fg='#000000')
label_title.pack(pady=5)

# Кнопка в главном окне для перехода к окну root
open_root_button = tk.Button(main_window,
                             text="Классификация датасета",
                             command=open_root_window, font=button_font,
                             bg='#4a6fa5', fg='#ffffff',
                             activebackground='#578acc',
                             activeforeground='#ffffff')
open_root_button.pack(pady=20)


# Кнопка в главном окне для перехода к окну real-time
open_root_button = tk.Button(main_window,
                             text="Классификация эмоций real-time",
                             command=open_real_time_classification_window,
                             font=button_font,
                             bg='#4a6fa5', fg='#ffffff',
                             activebackground='#578acc',
                             activeforeground='#ffffff')
open_root_button.pack(pady=20)


# Окно классификации на основе датасета
root = tk.Toplevel()
root.title("Классификация речевых аудиозаписей")
root.geometry("1024x768")
root.configure(bg='#DCDCDC')


# Текст названия программы
label_title = tk.Label(root, text="AUDIOWORK",
                       font=label_font, bg='#DCDCDC', fg='#000000')
label_title.pack(pady=5)

# Текст выбранной директории
label_directory = tk.Label(root, text="Директория датасета не выбрана",
                           font=label_font, bg='#DCDCDC', fg='#000000')
label_directory.pack(pady=5)

# Отображение директории
listbox = tk.Listbox(root, width=50, font=listbox_font, bg='#f5f5f5',
                     fg='#2e3f4f', selectbackground='#4a6fa5',
                     selectforeground='#000000')
listbox.pack(pady=5)

label_error = None

# Кнопка выбора директории
button_directory = tk.Button(root, text="Выбрать директорию",
                             command=choose_directory, font=button_font,
                             bg='#4a6fa5', fg='#ffffff',
                             activebackground='#578acc',
                             activeforeground='#ffffff')
button_directory.pack(pady=10)


# Кнопка начала работа программы
button_start = tk.Button(root, text="АНАЛИЗ", command=start_analysis,
                         font=button_font, bg='#4a6fa5', fg='#ffffff',
                         activebackground='#578acc',
                         activeforeground='#ffffff')
button_start.pack(pady=10)


# Блок сохранения результата
label_save = tk.Label(root, text="СОХРАНЕНИЕ РЕЗУЛЬТАТОВ в CSV-файле",
                      font=label_font, bg='#DCDCDC', fg='#000000')
label_save.pack(pady=5)

# Текст директории для сохранения выходных данных
label_directory_save = tk.Label(
    root, text="Назовите файл сохрания и выберите директорию",
    font=label_font, bg='#DCDCDC', fg='#000000')
label_directory_save.pack(pady=5)

# Поле названия файла
entry = tk.Entry(root)
entry.pack(pady=20)

# Кнопка выбора директории сохранения
button_save = tk.Button(root, text="Выбор директории для сохранения",
                        command=choose_directory_save, font=button_font,
                        bg='#4a6fa5', fg='#ffffff',
                        activebackground='#578acc',
                        activeforeground='#ffffff')
button_save.pack(pady=10)

# Кнопка сохранения результата
button_export = tk.Button(root, text="Сохранить результат",
                          command=save_dataframe, font=button_font,
                          bg='#4a6fa5', fg='#ffffff',
                          activebackground='#578acc',
                          activeforeground='#ffffff')
button_export.pack(pady=10)


# Кнопка возвращения в главное окно из root
button_export = tk.Button(root, text="Вернуться в меню",
                          command=close_root_window, font=button_font,
                          bg='#4a6fa5', fg='#ffffff',
                          activebackground='#578acc',
                          activeforeground='#ffffff')
button_export.pack(pady=10)


root.withdraw()

main_window.mainloop()

