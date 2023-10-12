import nltk
from pathlib import Path
from tkinter import *
import tkinter.ttk as ttk
from tkinter import messagebox
from tkinter import filedialog
from nltk import *
from nltk.corpus import stopwords
from string import punctuation
from wordfreq import top_n_list
import webbrowser
from langdetect import detect_langs
from nn import train_nn, predict_nn


docs = []
langs = []

nltk.download("stopwords")
nltk.download("punkt")

stopwords_german = set(stopwords.words('german'))
stopwords_spanish = set(stopwords.words('spanish'))

root = Tk()
root['background']='#f8e8fa'
space0 = Label(root, text='\n')
aboutButton = Button(root, text='About', width=8, height=2, bg='#f8adff')
space1 = Label(root, text='\n')
chooseDocsButton = Button(root, text='Choose html', width=55, height=2, bg='#f8adff')
space2 = Label(root, text='\n')
resultTree = ttk.Treeview(root,
                          columns=("File", "Language (freq words)", "Language (alphabet)", "Language (stopwords)", 'Distance'),
                          selectmode='browse', height=11)
resultTree.heading('File', text="File", anchor=W)
resultTree.heading('Language (freq words)', text="Language (freq words)", anchor=W)
resultTree.heading('Language (alphabet)', text="Language (alphabet)", anchor=W)
resultTree.heading('Language (stopwords)', text="Language (NBC)", anchor=W)
resultTree.heading('Distance', text="Distance", anchor=W)
resultTree.column('#0', stretch=NO, minwidth=0, width=0)
resultTree.column('#1', stretch=NO, minwidth=277, width=277)
resultTree.column('#2', stretch=NO, minwidth=277, width=277)
resultTree.column('#3', stretch=NO, minwidth=277, width=277)
resultTree.column('#4', stretch=NO, minwidth=277, width=277)
resultTree.column('#5', stretch=NO, minwidth=277, width=277)
space3 = Label(root, text='\n')
detectButton = Button(root, text='Detect language', width=55, height=2, bg='#f8adff')
space4 = Label(root, text='\n')
saveButton = Button(root, text='Save', width=55, height=2, bg='#f8adff')
space5 = Label(root, text='\n')
openButton = Button(root, text='Open document', width=55, height=2, bg='#f8adff')
space6 = Label(root, text='\n')



def nameOf(path):
    return Path(path).stem


def chooseDocsClicked():
    global docs, langs
    docs = []
    langs = []
    resultTree.delete(*resultTree.get_children())
    files = filedialog.askopenfilename(multiple=True)
    splitlist = root.tk.splitlist(files)
    for doc in splitlist:
        docs.append(
            (nameOf(doc), Path(doc, encoding="UTF-8", errors='ignore').read_text(encoding="UTF-8", errors='ignore')))
        resultTree.insert('', 'end', values=(nameOf(doc), '', '', '', ''))


def detect_freqwords_method(text):
    words = set()
    for sentence in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sentence):
            if word not in punctuation:
                words.add(word)
    if len(words.intersection(top_n_list('de', 30))) > len(words.intersection(top_n_list('es', 30))):
        return f"German - {round(len(words.intersection(top_n_list('de', 30))) / 30 * 100, 1)}%"
    else:
        return f"Spanish - {round(len(words.intersection(top_n_list('es', 30))) / 30 * 100, 1)}%"


def detect_alphabet_method(text):
    german_alphabet = set("ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜß".lower())
    spanish_alphabet = set("ABCDEFGHIJKLMNOPQRSTUVWXYZáéíóúñü".lower())
    chars = set()
    for sentence in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sentence):
            if word not in punctuation:
                chars.update(set(word.lower()))
    if len(german_alphabet.intersection(chars)) > len(spanish_alphabet.intersection(chars)):
        return f"German - {round(len(german_alphabet.intersection(chars)) / len(german_alphabet) * 100, 1)}%"
    else:
        return f"Spanish - {round(len(spanish_alphabet.intersection(chars)) / len(spanish_alphabet) * 100, 1)}%"


def detect_nn_method(text):
    return predict_nn([text])


def detectButtonClicked():
    resultTree.delete(*resultTree.get_children())
    for doc in docs:
        freqwords_method = detect_freqwords_method(doc[1])
        alphabet_method = detect_alphabet_method(doc[1])
        nn_method = detect_nn_method(doc[1])
        distance = detect_langs(doc[1])
        lang = "German" if str(distance[0]).split(":")[0] == 'de' else "Spanish"
        dist = round(float(str(distance[0]).split(":")[1]), 1)
        resultTree.insert('', 'end', values=(doc[0], freqwords_method, alphabet_method, nn_method, f"{lang} - {dist}"))


def save():
    with open("results.txt", "w") as file:
        for k in resultTree.get_children(""):
            for i in resultTree.item(k)['values']:
                file.write(i)
                file.write(" - ")
            file.write("\n")


def open_web():
    if resultTree.item(resultTree.focus())['values'] != '':
        name = resultTree.item(resultTree.focus())['values'][0]
        webbrowser.open(f'{name}.html')


def aboutButtonClicked():
    messagebox.showinfo("Clue",
                        "Для определения языка текста выберите файл, используя Choose html, далее нажмите на кнопку Detect language.\nДля сохранения результата нажмите кнопку Save.\nДля открытия документа по активной ссылке выберите необходимы документ в таблице и нажмите кнопку Open.")


if __name__ == '__main__':
    aboutButton.config(command=aboutButtonClicked)
    chooseDocsButton.config(command=chooseDocsClicked)
    detectButton.config(command=detectButtonClicked)
    saveButton.config(command=save)
    openButton.config(command=open_web)

    space0.pack()
    aboutButton.pack()
    space1.pack()
    chooseDocsButton.pack()
    space2.pack()
    resultTree.pack()
    space3.pack()
    detectButton.pack()
    space4.pack()
    saveButton.pack()
    space5.pack()
    openButton.pack()
    space6.pack()
    train_nn()
    root.mainloop()

