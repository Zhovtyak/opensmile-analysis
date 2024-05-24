import tkinter as tk
import os
import audiowork as aw
from const import RESULTS_ROOT_PATH, CONF
from tkinter import filedialog, ttk, Toplevel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


audiofiles_root_path = None
df = None


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
    statistic_window.configure(bg='#2e3f4f')

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
                     font=label_font, bg='#2e3f4f', fg='#ffffff')
    label.pack(pady=10)


def start_analysis():
    global label_error
    files_to_analize = aw.get_wav_files(audiofiles_root_path)
    if len(files_to_analize) == 0:
        if not label_error:
            label_error = tk.Label(root,
                                   text="Ошибка! В директории нет файлов .wav",
                                   font=label_font, bg='#2e3f4f', fg='#ffffff')
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
    print(df)
    dataframe_window()
    statistic_window()


root = tk.Tk()
root.title("Классификация речевых аудиозаписей")
root.geometry("1024x768")
root.configure(bg='#2e3f4f')

label_font = ("Helvetica", 12, "bold")
button_font = ("Helvetica", 12)
listbox_font = ("Helvetica", 12)

label_directory = tk.Label(root, text="Директория не выбрана",
                           font=label_font, bg='#2e3f4f', fg='#ffffff')
label_directory.pack(pady=5)

listbox = tk.Listbox(root, width=50, font=listbox_font, bg='#f5f5f5',
                     fg='#2e3f4f', selectbackground='#4a6fa5',
                     selectforeground='#ffffff')
listbox.pack(pady=5)

button_directory = tk.Button(root, text="Выбрать директорию",
                             command=choose_directory, font=button_font,
                             bg='#4a6fa5', fg='#ffffff',
                             activebackground='#578acc',
                             activeforeground='#ffffff')
button_directory.pack(pady=10)

button_start = tk.Button(root, text="АНАЛИЗ", command=start_analysis,
                         font=button_font, bg='#4a6fa5', fg='#ffffff',
                         activebackground='#578acc',
                         activeforeground='#ffffff')
button_start.pack(pady=10)

label_error = None

root.mainloop()
