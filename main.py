import nltk
import string
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from nltk.probability import FreqDist
from wordcloud import WordCloud
import spacy
import unicodedata
from collections import Counter, defaultdict

# Configurações de matplotlib
rcParams['font.family'] = 'Arial'

# Baixar os recursos necessários do NLTK
nltk.download('stopwords')

# Carregar o modelo de spaCy para espanhol
nlp = spacy.load("es_core_news_sm")

# Ajustar o limite de caracteres do SpaCy
nlp.max_length = 1500000  # Ajuste para um valor maior que o texto

# Função para remover acentos
def remove_acentos(texto):
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

# Função para dividir o texto em blocos menores
def dividir_texto(texto, tamanho_bloco=100000):
    """Divide o texto em blocos menores de tamanho especificado."""
    return [texto[i:i + tamanho_bloco] for i in range(0, len(texto), tamanho_bloco)]

# Função para processar texto
def processar_texto(texto, stopwords_personalizadas):
    """Processa o texto para remover stopwords e pontuações, além de lematizar."""
    texto = texto.lower()  # Converter para minúsculas
    texto = remove_acentos(texto)  # Remover acentos
    texto = ''.join([p for p in texto if p not in string.punctuation])  # Remover pontuação
    doc = nlp(texto)  # Processar com SpaCy

    # Filtrar tokens (remover stopwords e pontuações)
    tokens = [token.lemma_ for token in doc if token.text not in stopwords_personalizadas and not token.is_punct]
    return tokens

# Abertura do arquivo em .txt
with open(r'C:\Users\ANA CELIA\Desktop\Google Drive desktop\Mineração python\.venv\2023-cepal-sexto-informe-anual-america-latina-agenda-2030.txt', mode='r', encoding='utf-8') as f:
    texto = f.read()

# Lista de stopwords
stopwords_list = nltk.corpus.stopwords.words('spanish')
stopwords_personalizadas = stopwords_list + ['agenda', 'caribe', 'comisión', 'economica', 'cepal', 'américa', 'latina', '2023', '2021', '2022', '2030', 'además', 'debe', 'naciones', 'unidas', 'ello', 'hacia']

# Dividir o texto em blocos menores
blocos = dividir_texto(texto)

# Processar cada bloco e coletar os tokens
tokens = []
for bloco in blocos:
    tokens.extend(processar_texto(bloco, stopwords_personalizadas))

# Frequência das palavras mais comuns
freq = FreqDist(tokens)
freq_most_common = freq.most_common(20)
freq_cinquenta_mais = freq.most_common(50)

# Gráfico de barras com frequência de palavras
def plot_frequencia(freq_most_common, titulo, arquivo_saida):
    palavras, frequencias = zip(*freq_most_common)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=frequencias, y=palavras, palette="viridis")
    plt.title(titulo, fontsize=16)
    plt.xlabel("Frequência", fontsize=12)
    plt.ylabel("Palavras", fontsize=12)
    plt.tight_layout()
    plt.savefig(arquivo_saida)
    plt.close()

# Gráfico das 20 palavras mais frequentes
plot_frequencia(freq_most_common, "20 palavras mais frequentes", "frequencia_palavras_20.png")

# Gráfico das 50 palavras mais frequentes
plot_frequencia(freq_cinquenta_mais, "50 palavras mais frequentes", "frequencia_palavras_50.png")

# Gerar a nuvem de palavras
nuvem = WordCloud(
    background_color='white',
    stopwords=set(stopwords_personalizadas),
    height=1080,
    width=1080,
    max_words=50
)
nuvem.generate(' '.join(tokens))

# Salvar a nuvem de palavras
plt.figure(figsize=(8, 8))
plt.imshow(nuvem, interpolation='bilinear')
plt.axis("off")
plt.title("Nuvem de Palavras (50)", fontsize=16)
plt.tight_layout()
plt.savefig("nuvem_palavras.png")
plt.close()

# Lista de radicais de interesse
radicais_interesse = ['sosten', 'adapt', 'resilien', 'clim']

# Calcular frequência de palavras por radical
frequencias_por_radical = defaultdict(int)
frequencias_variacoes = defaultdict(Counter)

for token in tokens:
    for radical in radicais_interesse:
        if token.startswith(radical):
            frequencias_por_radical[radical] += 1
            frequencias_variacoes[radical][token] += 1

# Gráfico de barras para frequência total por radical
def plot_frequencia_radicais(frequencias_por_radical):
    radicais, frequencias = zip(*frequencias_por_radical.items())
    plt.figure(figsize=(10, 6))
    sns.barplot(x=frequencias, y=radicais, hue=radicais, palette="muted", legend=False)
    plt.title("Frequência Total por Radical", fontsize=16)
    plt.xlabel("Frequência", fontsize=12)
    plt.ylabel("Radicais", fontsize=12)
    plt.tight_layout()
    plt.savefig("frequencia_radicais.png")
    plt.close()

plot_frequencia_radicais(frequencias_por_radical)

# Gráfico de barras para variações por radical
def plot_frequencia_variacoes(frequencias_variacoes):
    for radical, variacoes in frequencias_variacoes.items():
        palavras, frequencias = zip(*variacoes.items())
        plt.figure(figsize=(12, 8))
        sns.barplot(x=frequencias, y=palavras, hue=palavras, palette="pastel", legend=False)
        plt.title(f"Variações do Radical '{radical}'", fontsize=16)
        plt.xlabel("Frequência", fontsize=12)
        plt.ylabel("Palavras", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"frequencia_variacoes_{radical}.png")
        plt.close()

plot_frequencia_variacoes(frequencias_variacoes)

# Salvar palavras mais frequentes
with open("frequencia_palavras.txt", "w", encoding="utf-8") as f:
    f.write("Palavras mais frequentes e suas quantidades:\n")
    for palavra, frequencia in freq_most_common:
        f.write(f"{palavra}: {frequencia}\n")

# Salvar frequência por radicais
with open("frequencia_radicais.txt", "w", encoding="utf-8") as f:
    f.write("Frequência total por radical:\n")
    for radical, frequencia in frequencias_por_radical.items():
        f.write(f"{radical}: {frequencia}\n")

# Salvar variações por radical
with open("frequencia_variacoes.txt", "w", encoding="utf-8") as f:
    f.write("Frequência de variações por radical:\n")
    for radical, variacoes in frequencias_variacoes.items():
        f.write(f"\nRadical: {radical}\n")
        for palavra, frequencia in variacoes.items():
            f.write(f"  {palavra}: {frequencia}\n")
