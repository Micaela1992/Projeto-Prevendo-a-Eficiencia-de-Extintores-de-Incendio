    ### MACHINE LEARNING NA SEGURANÇA DO TRABALHO
### PREVENDO A EFICIÊNCIA DE EXTINTORES DE INCÊNDIO

# Carregando os pacotes necessários
library(dplyr)
library(ggplot2)
library(readxl)
library(caret)
library(randomForest)
library(xgboost)
library(openxlsx)

# Carregando os dados
dados <- read_excel('Acoustic_Extinguisher_Fire_Dataset/Acoustic_Extinguisher_Fire_Dataset.xlsx')

# Entendendo os dados
dim(dados)
str(dados)
View(dados)
summary(dados)

# Data Munging
# Verificando se há dados em branco
colSums(is.na(dados))

# Convertendo a variável categórica para fator, para melhorar as análises
dados$FUEL <- as.factor(dados$FUEL)
dados$STATUS <- as.factor(dados$STATUS)
str(dados)

# Análise Exploratória

# Tentando entender como as variáveis se comportam em
# relação se as chamas foram extintas (status = 1) ou não (status = 0)
str(dados)
ggplot(dados, aes(x = FUEL, fill = STATUS)) +
  geom_bar(position = 'fill') +
  labs(title = 'Distribuição do Status por tipo de combustível',
       x = 'Fuel', y = 'Proporção') +
  scale_fill_manual(values = c('0' = 'blue', '1' = 'red'))

ggplot(dados, aes(x = SIZE, fill = STATUS)) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribuição do Tamanho por Status",
       x = "Tamanho", y = "Densidade") +
  scale_fill_manual(values = c("0" = "blue", "1" = "red"))

ggplot(dados, aes(x = DISTANCE, fill = STATUS)) +
  geom_density(alpha = 0.5) +
  labs(title = 'Distribuição da Distância por Status',
       x = 'Distância', y = 'Densidade') +
  scale_fill_manual(values = c("0" = "blue", "1" = "red"))

ggplot(dados, aes(x = DESIBEL, fill = STATUS)) +
  geom_density(alpha = 0.5) +
  labs(title = 'Distribuição dos Decibeis por Status',
       x = 'Decibel', y = 'Densidade') +
  scale_fill_manual(values = c('0' = 'blue', '1' = 'red'))

ggplot(dados, aes(x = AIRFLOW, fill = STATUS)) +
  geom_density(alpha = 0.5) +
  labs(title = 'Distribuição do Airflow por Status',
       x = 'Airflow', y = 'Densidade') +
  scale_fill_manual(values = c('0' = 'blue', '1' = 'red'))

ggplot(dados, aes(x = FREQUENCY, fill = STATUS)) +
  geom_density(alpha = 0.5) +
  labs(title = 'Distribuição de Frequency por Status',
       x = 'Frequency', y = 'Densidade') +
  scale_fill_manual(values = c('0' = 'blue', '1' = 'red'))

# Alguns insights:
# kerosene e thinner foram os combustíveis que os extintores menos apagaram.
# Extintores de tamanho entre 4 e 5 foram os que menos apagaram as chamas.
# Quanto maior a distância do extintor para a chama, menos eficiente ele é.
# Não encontrei uma relação justificável entre os decibeis e a eficiência do
# extintor.
# Quanto mais fluída for a saída do air flow, melhor o desempenho do extintor.
# Frequências de 0-35 Hz são melhores para o desempenho dos extintores.

# Machine Learning

# Dividindo os dados em treino e teste
split <- createDataPartition(y = dados$STATUS, p = 0.7, list = FALSE)
treino <- dados[split, ]
teste <- dados[-split, ]

# Primeira versão do modelo com Regressão Logística
modelo1 <- glm(STATUS ~ ., data = treino, family = binomial)

summary(modelo1)

# Fazendo as previsões da versão 1 do modelo
previsoes1 <- predict(modelo1, newdata = teste, type = 'response')

# Convertendo estas previsões em valores binários
previsoes1 <- ifelse(previsoes1 > 0.5, 1, 0)

# Calculando a acurácia da versão 1
acuracia1 <- mean(previsoes1 == teste$STATUS)
print(acuracia1)

# A acurácia de 90,31% é excelente, mas vamos testar outros modelos e compará-los.

# Criando uma segunda versão do mesmo modelo sem a variável Size.
modelo2 <- glm(STATUS ~ FUEL +
                 DISTANCE +
                 DESIBEL +
                 AIRFLOW +
                 FREQUENCY, data = treino, family = binomial)

summary(modelo2)

# Fazendo as previsões da versão 2 do modelo
previsoes2 <- predict(modelo2, newdata = teste, type = 'response')

# Convertendo estas previsões em valores binários
previsoes2 <- ifelse(previsoes2 > 0.5, 1, 0)

# Calculando a acurácia da versão 2
acuracia2 <- mean(previsoes2 == teste$STATUS)
print(acuracia2)

# Terceira versão do modelo com Random Forest e sem a variável Size
modelo3 <- randomForest(STATUS ~ FUEL +
                          DISTANCE +
                          DESIBEL +
                          AIRFLOW +
                          FREQUENCY,
                        data = treino)

print(modelo3)

# Fazendo as previsões com a versão 3 do modelo
previsoes3 <- predict(modelo3, newdata = teste)

# Calculando a acurácia da versão 3
acuracia3 <- mean(previsoes3 == teste$STATUS)
print(acuracia3)

# Na versão 3, a acurácia ficou melhor do que a segunda versão
# (88,91% contra 88,03%), mas abaixo da primeira versão (90,31%).
# Sendo assim, iremos analisar a matriz de confusão de cada um para tomar
# a melhor decisão.

# Criando as matrizes de confusão
matriz_confusao_modelo1 <- confusionMatrix(table(previsoes1, teste$STATUS))
matriz_confusao_modelo2 <- confusionMatrix(table(previsoes2, teste$STATUS))
matriz_confusao_modelo3 <- confusionMatrix(table(previsoes3, teste$STATUS))

print(matriz_confusao_modelo1)
print(matriz_confusao_modelo2)
print(matriz_confusao_modelo3)

# Conclusão: Realmente o modelo 1 se saiu melhor do que os demais, ele teve
# uma taxa menor de erros (tanto em relação de Falso Positivo e Falso Positivo).
# Logo, iremos finalizar esse projeto com a versão 1 do modelo.

# Criando um dataframe com os resultados
teste$STATUS_PREVISTO <- predict(modelo1, newdata = teste, type = 'response')

# Fim