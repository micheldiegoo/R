## Prevendo a Inadimplência de Clientes com Machine Learning e Power BI


#Definição do Problema

install.packages("Amelia")
install.packages("caret")
install.packages("ggplot2")
install.packages("dyplr")
install.packages("reshape")
install.packages("randomForest")
install.packages("e1071")

library(Amelia)
library(caret)
library(ggplot2)
library(dplyr)
library(reshape)
library(randomForest)
library(e1071)

setwd('D:/ARQUIVOS/POWER BI PRÃTICAS/DSA/Cap15 ml')
getwd()

dados <- read.csv("dados/dataset.csv")
View(dados)
str(dados)
summary(dados)


### Analise Exploratória, Limpeza e Transformação


# Removendo a coluna ID

dados$ID = NULL
View(dados)
dim(dados)


## Renomeando a coluna de classe

colnames(dados)
colnames(dados)[24] <- "Inadimplente"
colnames(dados)
View(dados)


## Verificando valores ausentes e removendo do dataset

sapply(dados, function(x) sum(is.na(x)))

missmap(dados, main = "Valores Missing Observados")
dados <- na.omit(dados)

## Renomeando colunas

str(dados)

colnames(dados)
colnames(dados)[2] <- "Genero"
colnames(dados)[3] <- "Escolaridade"
colnames(dados)[4] <- "Estado_Civil"
colnames(dados)[5] <- "Idade"

colnames(dados)

View(dados)

### Genero

View(dados$Genero)
str(dados$Genero)
summary(dados$Genero)

dados$Genero <- cut(dados$Genero, c(0,1,2), labels = c("Masculino", "Feminino"))


View(dados$Genero)

summary(dados$Genero)


## Escolaridade

View(dados$Escolaridade)
str(dados$Escolaridade)
summary(dados$Escolaridade)

dados$Escolaridade <- cut(dados$Escolaridade, c(0,1,2,3,4), 
                          labels = c("Pos Graduado", 
                                     "Graduado",
                                     "Ensino Medio",
                                     "Outros"))
View(dados$Escolaridade)
summary(dados$Escolaridade)



## Estado Civil


View(dados$Estado_Civil)
str(dados$Estado_Civil)
summary(dados$Estado_Civil)

dados$Estado_Civil <- cut(dados$Estado_Civil, c(-1,0,1,2,3), 
                          labels = c("Desconhecido", 
                                     "Casado",
                                     "Solteiro",
                                     "Outro"))
View(dados$Estado_Civil)
summary(dados$Estado_Civil)



## Idade

View(dados$Idade)
str(dados$Idade)
summary(dados$Idade)
hist(dados$Idade)

dados$Idade <- cut(dados$Idade, c(0,30,50,100), 
                          labels = c("Jovem", 
                                     "Adulto",
                                     "Idoso"))
View(dados$Idade)
summary(dados$Idade)


### Conversão da variável que indica pagamento para tipo fator

dados$PAY_0 <- as.factor(dados$PAY_0)
dados$PAY_2 <- as.factor(dados$PAY_2)
dados$PAY_3 <- as.factor(dados$PAY_3)
dados$PAY_4 <- as.factor(dados$PAY_4)
dados$PAY_5 <- as.factor(dados$PAY_5)
dados$PAY_6 <- as.factor(dados$PAY_6)

str(dados)

View(dados)



## Dataset após as conversoes

str(dados)

sapply(dados, function(x) sum(is.na(x)))
missmap(dados, main = "Valores Missing Observados")

dados <- na.omit(dados)



## Convertendo a variável target para o tipo fator


str(dados$Inadimplente)
dados$Inadimplente <- as.factor(dados$Inadimplente)

str(dados$Inadimplente)

table(dados$Inadimplente)
summary(dados$Inadimplente)

prop.table(table(dados$Inadimplente))


## Plot da distribuição

qplot(Inadimplente, data = dados, geom = "bar") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

## Set seed

set.seed(12345)

## Amostragem Estratificada
# Seleciona as linhas de acordo com a variável inadimplente como strata

indice <- createDataPartition(dados$Inadimplente, p = 0.75, list = FALSE)
dim(indice)


## Definimos os dados de treinamento como subconjunto do conjunto de dados original
## com números de indice de linha (conforme identificado acima) e todas as colunas

dados_treino <- dados[indice,]
table(dados$Inadimplente)

## Veja porcentagens entre as classes

prop.table(table(dados$Inadimplente))


## Número de registros dataset de treino

dim(dados_treino)

## Comparação das porcentagens entre as classes de treinamentos e dados originais

compara_dados <- cbind(prop.table(table(dados_treino$Inadimplente)),
                       prop.table(table(dados$Inadimplente)))
colnames(compara_dados) <- c("Treinamento", "Original")

compara_dados



## Melt Data - Converte colunas em linhas

melt_compara_dados <- melt(compara_dados)
melt_compara_dados


## Plot para visualizar a distribuição do treinamento vs original

ggplot(melt_compara_dados, aes(x = X1, y = value)) +
  geom_bar(aes(fill = X2), stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


## Tudo que não está no dataset de treino, está no dataset de teste

dados_teste <- dados[-indice,]

dim(dados_treino)
dim(dados_teste)

########################### MODELO DE MACHINE LEARNING ########################

## Construindo a primeira versão do modelo

?randomForest
modelo_V1 <- randomForest(Inadimplente ~ ., data = dados_treino)
modelo_V1

## Avaliando o modelo

plot(modelo_V1)


## Previsoes com o modelo de teste

previsoes_v1 <- predict(modelo_V1, dados_teste)


## Confusion Matrix

cm_v1 <- caret::confusionMatrix(previsoes_v1,dados_teste$Inadimplente, positive = "1")
cm_v1


## Calculando Precision, Recall e F1-Score, métricas de avaliação do modelo preditivo

y <- dados_teste$Inadimplente
y_pred_v1 <- previsoes_v1


precision <- posPredValue(y_pred_v1, y)
precision


recall <- sensitivity(y_pred_v1, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1


## Balanceamento de classe

install.packages(c("zoo","xts","quantmod"))

install.packages( "D:/POWER BI PRÃƒTICAS/DSA/Cap15/DMwR_0.4.1.tar.gz",repos = NULL, type="source" )
install.packages( c("xts","quantmod") )
install.packages("abind")
install.packages("ROCR")
install.packages("lattice")
install.packages("grid")
install.packages("ggplot2")
install.packages("randomforest")

library(lattice)
library(abind)
library(ROCR)
library(grid)
library(DMwR)
library(ggplot2)
library(randomForest)

## Aplicando o SMOTE


table(dados_treino$Inadimplente)
prop.table(table(dados_treino$Inadimplente))
set.seed(9560)
dados_treino_bal <-SMOTE(Inadimplente ~ ., data = dados_treino)
table(dados_treino_bal$Inadimplente)
prop.table(table(dados_treino_bal$Inadimplente))



## Construindo a segunda versão do modelo

modelo_v2 <- randomForest(Inadimplente ~ ., data = dados_treino_bal)
modelo_v2


## Avaliando o modelo


plot(modelo_v2)


## Previsoes com o modelo versao 2

previsoes_v2 <- predict(modelo_v2, dados_teste)


## Confusion Matrix

cm_v2 <- caret::confusionMatrix(previsoes_v2,dados_teste$Inadimplente, positive = "1")
cm_v2


## Calculando Precision, Recall e F1-Score, mÃƒÂ©tricas de avaliaÃƒÂ§ÃƒÂ£o do modelo preditivo

y2 <- dados_teste$Inadimplente
y2_pred_v2 <- previsoes_v2


precision <- posPredValue(y2_pred_v2, y2)
precision


recall <- sensitivity(y2_pred_v2, y2)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1


## Importância das variáveis preditoras para o modelo preditivo

varImpPlot(modelo_v2)

## Obtendo as variáveis mais importantes

imp_var <- importance(modelo_v2)
  varImportance <- data.frame(variables = row.names(imp_var),
                              Importance = round(imp_var[ , 'MeanDecreaseGini'],2))
  
## Criando o rank de variáveis baseado na importância
  
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#', dense_rank((desc(Importance)))))

## Usando ggplot2 para visualizar a importância relativa das variáveis

ggplot(rankImportance,
       aes(x = reorder(variables, Importance),
           y = Importance,
           fill = Importance)) +
  geom_bar(stat = 'identity') +
  geom_text(aes(x = variables, y = 0.5, label = Rank),
            hjust = 0,
            vjust = 0.55,
            size = 4,
            colour = 'red') +
  labs(x = 'Variables') +
  coord_flip()
  
################### Usar regressão logistica, KNN,nayve bayes, SVM, otimizacao de hiperparametros, padronizar as variaveis de entrada


## Construindo a terceira versao do modelo apenas com variaveis mais importantes

modelo_v3 <- randomForest(Inadimplente ~ PAY_0 + PAY_2 + PAY_3 + PAY_AMT1 + PAY_AMT2 + PAY_5 + BILL_AMT1,
                          data = dados_treino_bal)

modelo_v3



## Avaliando o modelo


plot(modelo_v3)


## Previsoes com o modelo versao 3

previsoes_v3 <- predict(modelo_v3, dados_teste)


## Confusion Matrix

cm_v3 <- caret::confusionMatrix(previsoes_v3,dados_teste$Inadimplente, positive = "1")
cm_v3


## Calculando Precision, Recall e F1-Score, métricas de avaliação do modelo preditivo

y3 <- dados_teste$Inadimplente
y3_pred_v3 <- previsoes_v3


precision <- posPredValue(y3_pred_v3, y3)
precision


recall <- sensitivity(y2_pred_v2, y2)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1


## Salvando o modelo em disco

saveRDS(modelo_v3, file = "modelo/modelo_v3.rds")


## Dados dos Clientes

PAY_0 <- c(0, 0, 0)
PAY_2 <- c(0, 0, 0)
PAY_3 <- c(1, 0, 0)
PAY_AMT1 <- c(1100, 1000, 1200)
PAY_AMT2 <- c(1500, 1300, 1150)
PAY_5 <- c(0, 0, 0)
BILL_AMT1 <- c(350, 420, 280)

## Concatena em um dataframe

novos_clientes <- data.frame(PAY_0, PAY_2,PAY_3,PAY_AMT1,PAY_AMT2,PAY_5,BILL_AMT1)
View(novos_clientes)
str(dados_treino_bal)
str(novos_clientes)

## Previsoes

previsoes_novos_cliente <- predict(modelo_v3, novos_clientes)

## Convertendo os tipos de dados


novos_clientes$PAY_0 <- factor(novos_clientes$PAY_0, levels = levels(dados_treino_bal$PAY_0))
novos_clientes$PAY_2 <- factor(novos_clientes$PAY_2, levels = levels(dados_treino_bal$PAY_2))
novos_clientes$PAY_3 <- factor(novos_clientes$PAY_3, levels = levels(dados_treino_bal$PAY_3))
novos_clientes$PAY_5 <- factor(novos_clientes$PAY_5, levels = levels(dados_treino_bal$PAY_5))

str(novos_clientes)


## Previsoes

previsoes_novos_clientes <- predict(modelo_v3, novos_clientes)


View(previsoes_novos_clientes)





















































