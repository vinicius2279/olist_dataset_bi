import pandas as pd
from sklearn.linear_model import LinearRegression

# import do CSV criado via Power Query que contém apenas as vendas e datas
df = pd.read_csv("vendas_data_olist.csv")

# renomeando a coluna para melhor exibição dos dados
df = df.rename(columns={
    'Soma de price': 'vendas'
})

# removendo formato moeda para leitura correta na regressão linear
df['vendas'] = (
    df['vendas']
    .astype(str)
    .str.replace('R$', '', regex=False)
    .str.replace(',', '.', regex=False)
    .str.strip()
)

# tratando a coluna de data
df['mes_ano'] = pd.to_datetime(df['mes_ano'])

# transformando em valor numérico
df['vendas'] = pd.to_numeric(df['vendas'])

# removendo linhas com dados faltantes ou inválidos que podem afetar a previsão
df = df.drop(index=[0, 1, 2, 23, 24, 25])
df = df.reset_index(drop=True)

# criando um índice
df['indice'] = range(len(df))

# transformando colunas em variáveis, sendo x a variável independente e y a variável alvo
x = df[['indice']]
y = df['vendas']

# treinamento do modelo de regressão linear
modelo = LinearRegression()
modelo.fit(x, y)

# criando índices futuros para geração das previsões
futuro = pd.DataFrame({'indice': range(len(df), len(df) + 7)})

# previsão para os índices criados
previsoes = modelo.predict(futuro)

# criando um DataFrame com os dados futuros
futuro_df = pd.DataFrame({

    'mes_ano': pd.date_range(
        start=df['mes_ano'].iloc[-1],
        periods=8,
        freq='ME'
    )[1:],

    'indice': futuro['indice'],
    'vendas_reais': pd.NA,
    'previsao': previsoes
})

# criando um DataFrame com o histórico
historico = pd.DataFrame({

    'mes_ano': df['mes_ano'],
    'indice': df['indice'],
    'vendas_reais': y,
    'previsao': pd.NA
})

# concatenando histórico e futuro
resultado = pd.concat([historico, futuro_df])

# concatenação criou vendas_reais e previsao como tipo objeto, aqui estou transformando em numérico
resultado['previsao'] = pd.to_numeric(resultado['previsao'], errors='coerce')
resultado['vendas_reais'] = pd.to_numeric(resultado['vendas_reais'], errors='coerce')

# arredondando casas decimais
resultado['previsao'] = resultado['previsao'].round(2)
resultado['vendas_reais'] = resultado['vendas_reais'].round(2)

# resultado final com histórico e previsões
print(resultado)

resultado.to_csv('vendas_previsao.csv')