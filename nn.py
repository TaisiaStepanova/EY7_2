from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Подготовка данных (простой пример)


texts = [
    "hola", "adiós", "gracias", "por favor", "amigo", "amor", "feliz", "triste", "bueno",
    "malo", "hermoso", "feo", "comida", "agua", "vino", "café", "playa", "montaña", "sol",
    "lluvia", "nube", "cielo", "estrella", "luna", "gato", "perro", "casa", "coche", "bicicleta",
    "avión", "tren", "autobús", "libro", "papel", "pluma", "teléfono", "ordenador", "internet",
    "familia", "padre", "madre", "hermano", "hermana", "hijo", "hija", "abuelo", "abuela",
    "comer", "beber", "dormir", "correr", "nadar", "viajar", "aprender", "enseñar", "trabajar",
    "descansar", "jugar", "caminar", "escuchar", "ver", "hablar", "pensar", "sentir", "amar",
    "odiar", "cantar", "bailar", "reír", "llorar", "soñar", "esperar", "ayudar", "comprender",
    "estudiar", "celebrar", "cocinar", "visitar", "regresar", "comprar", "vender", "recibir",
    "dar", "recoger", "dejar", "encontrar", "perder", "ganar", "creer", "dudar", "mirar",
    "leer", "escribir", "pintar", "dibujar", "viajar", "explorar", "descubrir", "navegar",
    "correr", "saltar", "volar", "conducir", "nadar", "nadar", "bucear", "escaladar", "esquiar",
    "patinar", "andar en patineta", "jugar fútbol", "jugar baloncesto", "jugar tenis",
    "jugar golf", "jugar ajedrez", "jugar videojuegos", "ver películas", "escuchar música",
    "tocar música", "bailar salsa", "bailar flamenco", "bailar tango", "bailar reguetón",
    "cocinar paella", "cocinar tapas", "disfrutar del flamenco", "visitar la Sagrada Familia",
    "visitar el Prado", "visitar el Alhambra", "visitar la Catedral de Sevilla", "visitar la Plaza Mayor",
    "recoger conchas en la playa", "dejar un mensaje de voz", "encontrar un tesoro", "perderse en la ciudad",
    "ganar un concurso", "creer en el destino", "dudar de la realidad", "mirar las estrellas",
    "leer un libro interesante", "escribir un poema", "pintar un cuadro", "dibujar un retrato",
    "viajar por el mundo", "explorar nuevas culturas", "descubrir lugares mágicos", "navegar por el océano",
    "correr una maratón", "saltar desde un trampolín", "volar en un globo", "conducir un coche rápido",
    "nadar en aguas cristalinas", "bucear en el arrecife de coral", "escaladar una montaña",
    "esquiar en las montañas", "patinar sobre hielo", "andar en patineta en el parque",
    "jugar fútbol en la playa", "jugar baloncesto en el parque", "jugar tenis en la cancha",
    "jugar golf en el campo", "jugar ajedrez con amigos", "jugar videojuegos en línea",
    "ver películas en el cine", "escuchar música en conciertos", "tocar música en una banda",
    "bailar salsa en una fiesta", "bailar flamenco en el tablao", "bailar tango en un salón",
    "bailar reguetón en la discoteca", "cocinar paella en casa", "cocinar tapas para amigos",
    "disfrutar del flamenco en vivo", "visitar la Sagrada Familia en Barcelona",
    "visitar el Museo del Prado en Madrid", "visitar el Alhambra en Granada",
    "visitar la Catedral de Sevilla en Andalucía", "visitar la Plaza Mayor en Madrid",
    "recoger conchas en la Costa Brava", "dejar un mensaje de voz a un ser querido",
    "encontrar un tesoro escondido en la playa", "perderse en las calles estrechas de un pueblo",
    "ganar un concurso de talentos", "creer en el poder de la imaginación", "dudar de la veracidad de las noticias",
    "mirar las estrellas en una noche despejada", "leer un libro clásico de la literatura",
    "escribir un diario personal", "pintar un paisaje en acuarela", "dibujar un retrato a lápiz",
    "viajar por Europa en tren", "explorar las maravillas naturales de América del Sur",
    "descubrir la rica historia de Asia", "navegar por el río Amazonas en un barco",
    "correr una maratón en Nueva York", "saltar desde un trampolín en una piscina",
    "volar en un globo aerostático sobre los Alpes", "conducir un coche rápido en un circuito",
    "nadar en aguas cristalinas del Caribe", "bucear en el arrecife de coral de Australia",
    "escaladar una montaña en los Andes", "esquiar en las montañas de los Alpes Suizos",
    "patinar sobre hielo en una pista de patinaje", "andar en patineta en un parque de skate",
    "jugar fútbol en la playa con amigos",
    "hallo", "guten tag", "wie geht es dir?", "danke", "bitte", "freund", "liebe", "glücklich", "traurig", "gut",
    "schlecht", "schön", "hässlich", "essen", "wasser", "wein", "kaffee", "strand", "berg", "sonne",
    "regen", "wolke", "himmel", "stern", "mond", "katze", "hund", "haus", "auto", "fahrrad",
    "flugzeug", "zug", "bus", "buch", "papier", "stift", "telefon", "computer", "internet",
    "familie", "vater", "mutter", "bruder", "schwester", "sohn", "tochter", "opa", "oma",
    "essen", "trinken", "schlafen", "laufen", "schwimmen", "reisen", "lernen", "lehren", "arbeiten",
    "ruhe", "spielen", "gehen", "hören", "sehen", "sprechen", "denken", "fühlen", "lieben",
    "hassen", "singen", "tanzen", "lachen", "weinen", "träumen", "warten", "helfen", "verstehen",
    "studieren", "feiern", "kochen", "besuchen", "zurückkehren", "kaufen", "verkaufen", "erhalten",
    "geben", "sammeln", "verlassen", "finden", "verlieren", "gewinnen", "glauben", "zweifeln", "schauen",
    "lesen", "schreiben", "malen", "zeichnen", "reisen", "erkunden", "entdecken", "segeln",
    "laufen", "springen", "fliegen", "fahren", "schwimmen", "tauchen", "klettern", "skifahren",
    "skaten", "skateboarden", "fußball spielen", "basketball spielen", "tennis spielen",
    "golf spielen", "schach spielen", "videospiele spielen", "filme ansehen", "musik hören",
    "musik spielen", "salsa tanzen", "flamenco tanzen", "tango tanzen", "reggaeton tanzen",
    "paella kochen", "tapas kochen", "flamenco genießen", "die sagrada familia besuchen",
    "das prado-museum besuchen", "die alhambra besuchen", "die kathedrale von sevilla besuchen",
    "die plaza mayor besuchen", "muscheln am strand sammeln", "eine voicemail hinterlassen",
    "einen schatz finden", "sich in der stadt verirren", "einen wettbewerb gewinnen", "an das schicksal glauben",
    "an der realität zweifeln", "die sterne betrachten", "ein klassisches buch lesen", "ein tagebuch schreiben",
    "ein landschaftsbild malen", "ein portrait zeichnen", "durch europa reisen", "naturwunder in südamerika erkunden",
    "die reiche geschichte asiens entdecken", "den amazonasfluss befahren", "einen marathon laufen", "vom trampolin springen",
    "in einem heißluftballon über den alpen schweben", "einen schnellen sportwagen fahren", "in der kristallklaren karibik schwimmen",
    "im australischen korallenriff tauchen", "auf einem anden-gipfel klettern", "in den schweizer alpen skifahren",
    "auf einer eisbahn schlittschuh laufen", "in einem skatepark skateboard fahren", "am strand fußball spielen",
    "im park basketball spielen", "auf dem tennisplatz tennis spielen", "auf dem golfplatz golf spielen",
    "eine partie schach mit freunden spielen", "videospiele online spielen", "filme im kino ansehen",
    "musik in konzerten hören", "musik in einer band spielen", "salsa in einer tanzbar tanzen",
    "flamenco in einer tablao-show genießen", "tango in einem saal tanzen", "reggaeton in der disco tanzen",
    "paella zu hause kochen", "tapas für freunde zubereiten", "flamenco live erleben",
    "die sagrada familia in barcelona besichtigen", "das prado-museum in madrid besuchen",
    "die alhambra in granada erkunden", "die kathedrale von sevilla in andalusien besichtigen",
    "die plaza mayor in madrid besuchen", "muscheln an der costa brava sammeln",
    "eine sprachnachricht für einen geliebten menschen hinterlassen", "einen versteckten schatz am strand finden",
    "sich in den engen gassen einer stadt verirren", "einen talentwettbewerb gewinnen", "an die kraft der vorstellungskraft glauben",
    "an der wahrheit der nachrichten zweifeln", "an einem klaren nachthimmel die sterne betrachten",
    "ein klassisches literaturbuch lesen", "ein persönliches tagebuch schreiben",
    "ein landschaftsbild in wasserfarben malen", "ein bleistiftportrait zeichnen", "mit dem zug durch europa reisen",
    "die naturschönheiten südamerikas erkunden", "die reiche geschichte asiens entdecken",
    "auf einem boot den amazonasfluss hinunterfahren", "einen marathon in new york laufen"]



labels = [*["Spanish"]*205, *["German"]*(len(texts)-205)]
vectorizer = []
classifier = []
label_dict = []

def train_nn():
    # Создание и настройка векторизатора
    global vectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)  # texts - список текстов

    # Преобразование меток языка в числовые метки
    global label_dict
    label_dict = {label: idx for idx, label in enumerate(set(labels))}
    y = [label_dict[label] for label in labels]

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание и обучение наивного байесовского классификатора
    global classifier
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)


def predict_nn(text):
    X_new = vectorizer.transform(text)

    # Предсказание языка для новых текстов
    predicted_languages = [list(label_dict.keys())[label] for label in classifier.predict(X_new)]

    #print(f"Predicted languages: {predicted_languages}")
    return predicted_languages

