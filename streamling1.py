import streamlit
import streamlit  as st
import pandas as pd
import seaborn as sns
import numpy as np
import os
import sys

from narwhals.selectors import categorical
#import tkinter as tk
from statsmodels.stats.multicomp import MultiComparison


import matplotlib.pyplot as plt
from bleach import clean
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import anderson
from scipy.stats import shapiro, levene
from scipy.stats import anderson
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from streamlit import selectbox
tab1, tab2, tab3 = st.tabs(["Pré-processamento e Análise descritiva", "Exploraçao  pressupostos  Anova", "tukey test e gráficos "])

with tab1:
    st.title("Aplicativo anova")
    st.write("O objetivo é fazer uma **Anova** com no máximo 4 fatores .")
    meu_label = "Envie seu arquivo CSV"
    arquivo = st.file_uploader(meu_label, type="csv")# criopu um upload
    #data = pd.read_csv(arquivo, encoding='UTF-8', sep=';')# data em dataFrame

    if arquivo is  None:
        st.warning('Aguardando a escolha dos dados ')

    else:
        st.success(f"O arquivo selecionado foi: {arquivo.name}")

    data = pd.read_csv(arquivo, encoding='UTF-8', sep=';')
    data_copia = data.copy

    escolha = st.radio("Você deseja ver seus dados ?", ["Sim", "Não"]).upper().strip()
    if escolha == 'SIM':
        st.dataframe(data)

    variavel = st.radio('Quantas variáveis categóricas você deseja analisar?', [1,2,3], horizontal = True)

    data1 = data.to_dict()
    chaves = data1.keys()
    chaves1 = list(chaves)


    escolhas = []
    if variavel == 1:
        categorica= st.selectbox('Escolha as variável categórica',['Selecione']+ chaves1, key = '1')
        if categorica != 'Selecione':
            st.success(f"Você escolheu a variável categórica: {categorica}") # escolha essa primeiro

        continua = st.selectbox('Escolha a variável contínua',['Selecione'] + chaves1, key = '2')
        if continua != 'Selecione': #Escolha essa depois que a primeira é escolhida
            st.success(f"Você escolheu a variável contínua: {continua}")

            if categorica != 'Selecione' and continua != 'Selecione':
                escolhas.append(categorica)
                escolhas.append(continua)
                data = data[escolhas]# escolhi e armazenei as variáveis que quero trabalhar
                st.write(data)

                data_na = data.isna().sum()
                #fazer uma função aqui! def retirarana(data na , data) e chamo novamente em outro lugar
                if data_na.sum() == 0:
                    st.write(f'Você Não tem **NA** nas  variáveis de seus dados  ')
                    st.dataframe(data_na)
                else:
                    st.write(f'Você  tem **NA** nas  variáveis de seus dados  ')
                    st.dataframe(data_na)
                    st.write('Você gostaria de retira  as **NAs** ou substituir por valores médios?')
                    escolha_2 = st.radio("Você deseja ?", ["Substituir por Valores médios", "Retirar Na"])
                    if escolha_2 == "Substituir por Valores médios":
                        data = data.fillna(data.median(numeric_only=True))
                        st.write('Dados com valores médios substituidos no lugar de NA')
                        st.dataframe(data)
                    else:
                        data = data.dropna(axis=1)
                        st.dataframe(data)  # manter o mes


                #somente se essa condição for respeitada, então fazemos a anpalise
                data_grouped = data.groupby(categorica)[continua].describe()
                st.write(f"Análise descritiva da variável {continua}")
                st.dataframe(data_grouped)
                cv = data.loc[:,continua].values# transforma em array numpy  e pega os valores, para o cálculo
                #st.write(cv)
                #cálculo do cv
                cv2 =  np.std(cv) / np.mean(cv) * 100
                st.write(f"CV% = {cv2}")

                st.subheader('Z-score ')
                zscore = (data[continua] - np.mean(data[continua])) / np.std(data[continua])
                data2 = data.copy()
                data2['zscore'] = zscore
                st.write(data2)
                # print(zscore)

                #plotar a curva de KDE
                fig2, ax = plt.subplots()
                sns.kdeplot(data=data2, x='zscore', fill=True, alpha=0.3)
                ax.set_title("Curva de KDE para visualização de normalidade ")
                plt.axvline(0, color='red', linestyle='dashed', linewidth=1)  # Linha central em 0
                #sns.stripplot(x=zscore, color='black', jitter=True, alpha=0.5, ax=ax)
                st.pyplot(fig2)


                # Plotar o boxplot dos z-scores
                fig, ax = plt.subplots()
                sns.boxplot(x=zscore, ax=ax)
                sns.stripplot(x=zscore, color='black', jitter=True, alpha=0.5, ax=ax)
                ax.set_title("Boxplot dos Z-Scores")
                st.pyplot(fig)



                #cálculo de outliers:
                st.subheader('Outlier ')
                st.write('O cálculo de outlier consiste em identificar os dados que estão acima ou abaixo  de 3 desvios padrão do Z-score e utiliza-se o método do IQR')


                Q1 = data.loc[:, continua].quantile(0.25)
                Q3 = data.loc[:, continua].quantile(0.75)
                # print(Q1)
                # print(Q3)

                IQR = Q3 - Q1
                LS = Q3 + 1.5 * IQR
                LI = Q1 - 1.5 * IQR
                print()
                linha = 70 * '='
                print(linha)

                print(linha)


                #Outliers acima e abaixo:
                st.write('Limite superior = ', LS)
                acima = data[(data.loc[:, continua ] > LS)]
                if acima.empty : # Usa-se empty, porque estamos tratando de um dataframe
                    st.write('Você não tem outliers acima do limite superior  ')
                    st.write(acima)
                else:
                    st.write('Você tem alguns outliers acima do limite superior')
                    st.write(acima)








                st.write("limite inferior = ", LI)
                abaixo = data[(data.loc[:, continua] < LI)]
                if abaixo.empty : # Usa-se empty, porque estamos tratando de um dataframe
                    st.write('Você não tem outliers abaixo do limite inferior  ')
                    st.write(abaixo)
                else:
                    st.write('Você tem alguns outliers abaixo  do limite inferior')
                    st.write(abaixo)

                escolha_3 = st.radio("Você deseja retirar os outliers ?", ["SIM", "Não "], horizontal=True)
                if escolha_3 == 'SIM':
                    data = data[(data[continua] < LS) & (data[continua]>LI)]
                    st.success('os outliers foram tirados com sucesso ')
                    escolha_4 = st.radio("Você gostaria de ver os dados sem outliers?", ['Sim', 'Não'])
                    if escolha_4 == 'Sim':
                        st.write('Seus dados sem outliers')
                        st.dataframe(data)
                    escolha_5 = st.radio('Você deseja ver os gráficos boxplot e KDE', ['Sim', 'Não'], horizontal=True)
                    if escolha_5 == 'Sim':
                        st.subheader('Z-score ')
                        zscore = (data[continua] - np.mean(data[continua])) / np.std(data[continua])
                        data['zscore'] = zscore

                        # plotar a curva de KDE
                        fig2, ax = plt.subplots()
                        sns.kdeplot(data=data, x='zscore', fill=True, alpha=0.3)
                        ax.set_title("Curva de KDE para visualização de normalidade ")
                        plt.axvline(0, color='red', linestyle='dashed', linewidth=1)  # Linha central em 0
                        # sns.stripplot(x=zscore, color='black', jitter=True, alpha=0.5, ax=ax)
                        st.pyplot(fig2)

                        st.dataframe(data)
                        # Plotar o boxplot dos z-scores
                        fig, ax = plt.subplots()
                        sns.boxplot(x=zscore, ax=ax)
                        sns.stripplot(x=zscore, color='black', jitter=True, alpha=0.5, ax=ax)
                        ax.set_title("Boxplot dos Z-Scores")
                        st.pyplot(fig)


                st.write('Análise descritiva dos seus dados ')
                data_grouped = data.groupby(categorica)[continua].describe()
                st.dataframe(data_grouped)
                cv = data.loc[:, continua].values  # transforma em array numpy  e pega os valores, para o cálculo
                # st.write(cv)
                # cálculo do cv
                cv2 = np.std(cv) / np.mean(cv) * 100
                st.write(f"CV% = {cv2}")


            else:
                st.warning("selecione as variáveis")
            st.warning('Se quiser continuar a análise, então clica na aba 2 acima **Pressupostos da Anova**')









                    #boxplot:
                # padronização dos dados:



    escolhas = []
    if variavel == 2:
        categorica= st.selectbox('Escolha a primeira variável  categórica',['Selecione'] + chaves1, key = '3')
        if categorica != 'Selecione':
            st.success(f"Você escolheu a variável categórica: {categorica}")

        categorica_2 = st.selectbox('Escolha a segunda variável  categórica',['Selecione'] +chaves1, key = '4')
        if categorica_2 != 'Selecione':
            st.success(f"Você escolheu a variável categórica: {categorica_2}")

        continua= st.selectbox('Escolha a variável contínua',['Selecione'] +chaves1, key = '5')
        if continua != 'Selecione':
            st.success(f"Você escolheu a variável contínua: {continua}")

            escolhas.append(categorica)
            escolhas.append(categorica_2)
            escolhas.append(continua)

            data = data[escolhas]  # escolhi e armazenei as variáveis que quero trabalhar
            st.write(data)
            data_na = data.isna().sum()
            # fazer uma função aqui! def retirarana(data na , data) e chamo novamente em outro lugar
            if data_na.sum() == 0:
                st.write(f'Você Não tem **NA** nas  variáveis de seus dados  ')
                st.dataframe(data_na)
            else:
                st.write(f'Você  tem **NA** nas  variáveis de seus dados  ')
                st.dataframe(data_na)
                st.write('Você gostaria de retira  as **NAs** ou substituir por valores médios?')
                escolha_2 = st.radio("Você deseja ?", ["Substituir por Valores médios", "Retirar Na"])
                if escolha_2 == "Substituir por Valores médios":
                    data = data.fillna(data.median(numeric_only=True))
                    st.write('Dados com valores médios substituidos no lugar de NA')
                    st.dataframe(data)
                else:
                    data = data.dropna(axis=1)
                    st.dataframe(data)  # manter o mes

        if categorica !='Selecione' and categorica_2 != 'Selecione' and continua!= 'Selecione':
            cv = data.loc[:, continua].values  # transforma em array numpy  e pega os valores, para o cálculo
            # cálculo do cv
            st.write('Análise descritiva dos seus dados ')
            data_grouped = data.groupby(data.columns[0:variavel].tolist()).describe()
            st.dataframe(data_grouped)
            cv2 = np.std(cv) / np.mean(cv) * 100
            st.write(f"CV% = {cv2}")

            st.subheader('Z-score ')
            zscore = (data[continua] - np.mean(data[continua])) / np.std(data[continua])
            data2 = data.copy()
            data2['zscore'] = zscore
            st.write(data2)
            # print(zscore)

            # plotar a curva de KDE
            fig2, ax = plt.subplots()
            sns.kdeplot(data=data2, x='zscore', fill=True, alpha=0.3)
            ax.set_title("Curva de KDE para visualização de normalidade ")
            plt.axvline(0, color='red', linestyle='dashed', linewidth=1)  # Linha central em 0
            # sns.stripplot(x=zscore, color='black', jitter=True, alpha=0.5, ax=ax)
            st.pyplot(fig2)

            # Plotar o boxplot dos z-scores
            fig, ax = plt.subplots()
            sns.boxplot(x=zscore, ax=ax)
            sns.stripplot(x=zscore, color='black', jitter=True, alpha=0.5, ax=ax)
            ax.set_title("Boxplot dos Z-Scores")
            st.pyplot(fig)

            # cálculo de outliers:
            st.subheader('Outlier ')
            st.write(
                'O cálculo de outlier consiste em identificar os dados que estão acima ou abaixo  de 3 desvios padrão do Z-score e utiliza-se o método do IQR')

            Q1 = data.loc[:, continua].quantile(0.25)
            Q3 = data.loc[:, continua].quantile(0.75)
            # print(Q1)
            # print(Q3)

            IQR = Q3 - Q1
            LS = Q3 + 1.5 * IQR
            LI = Q1 - 1.5 * IQR
            print()
            linha = 70 * '='
            print(linha)

            print(linha)

            # Outliers acima e abaixo:
            st.write('Limite superior = ', LS)
            acima = data[(data.loc[:, continua] > LS)]
            if acima.empty:  # Usa-se empty, porque estamos tratando de um dataframe
                st.write('Você não tem outliers acima do limite superior  ')
                st.write(acima)
            else:
                st.write('Você tem alguns outliers acima do limite superior')
                st.write(acima)

            st.write("limite inferior = ", LI)
            abaixo = data[(data.loc[:, continua] < LI)]
            if abaixo.empty:  # Usa-se empty, porque estamos tratando de um dataframe
                st.write('Você não tem outliers abaixo do limite inferior  ')
                st.write(abaixo)
            else:
                st.write('Você tem alguns outliers abaixo  do limite inferior')
                st.write(abaixo)

            escolha_3 = st.radio("Você deseja retirar os outliers ?", ["SIM", "Não "], horizontal=True)
            if escolha_3 == 'SIM':
                data = data[(data[continua] < LS) & (data[continua] > LI)]
                st.success('os outliers foram tirados com sucesso ')
                escolha_4 = st.radio("Você gostaria de ver os dados sem outliers?", ['Sim', 'Não'])
                if escolha_4 == 'Sim':
                    st.write('Seus dados sem outliers')
                    st.dataframe(data)
                escolha_5 = st.radio('Você deseja ver os gráficos boxplot e KDE', ['Sim', 'Não'], horizontal=True)
                if escolha_5 == 'Sim':
                    st.subheader('Z-score ')
                    zscore = (data[continua] - np.mean(data[continua])) / np.std(data[continua])
                    data['zscore'] = zscore

                    # plotar a curva de KDE
                    fig2, ax = plt.subplots()
                    sns.kdeplot(data= data, x='zscore', fill=True, alpha=0.3)
                    ax.set_title("Curva de KDE para visualização de normalidade ")
                    plt.axvline(0, color='red', linestyle='dashed', linewidth=1)  # Linha central em 0
                    # sns.stripplot(x=zscore, color='black', jitter=True, alpha=0.5, ax=ax)
                    st.pyplot(fig2)

                    st.dataframe(data)
                    # Plotar o boxplot dos z-scores
                    fig, ax = plt.subplots()
                    sns.boxplot(x=zscore, ax=ax)
                    sns.stripplot(x=zscore, color='black', jitter=True, alpha=0.5, ax=ax)
                    ax.set_title("Boxplot dos Z-Scores")
                    st.pyplot(fig)

                    st.write('Análise descritiva dos seus dados ')
                    data_grouped = data.groupby(data.columns[0:variavel].tolist()).describe()
                    st.dataframe(data_grouped)
                    cv = data.loc[:, continua].values  # transforma em array numpy  e pega os valores, para o cálculo
                    # st.write(cv)
                    # cálculo do cv
                    cv2 = np.std(cv) / np.mean(cv) * 100
                    st.write(f"CV% = {cv2}")

                    with tab2:
                        st.header('Análise exploratória')
                        st.subheader('Gráfico boxplot')


                        Eixo_y = data.columns[2]
                        print(Eixo_y)
                        Axis_x = data.columns[0]

                        dentro_1 = data.columns[1]
                        #colocando gráfico um ao lado do outro
                        col1, col2 = st.columns(2)

                        with col1:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.boxplot(x=Axis_x, y=Eixo_y, hue=dentro_1, palette="Set2", data=data, ax=ax)
                            sns.despine(offset=10, trim=True)
                            st.pyplot(fig)

                        with col2:

                            st.subheader('Gráfico de barras')
                            fig3, ax = plt.subplots(figsize=(10, 6))
                            sns.barplot(x=Axis_x, y=Eixo_y, hue=dentro_1, palette="Set2",errorbar = 'sd', width = 0.5, data=data, ax=ax)
                            plt.ylim(0)
                            #sns.despine(offset=10, trim=True)
                            st.pyplot(fig3)



                        escolha_6 = st.radio('Você gostaria de alterar o gráfico boxplot?', ['Sim', 'Não '])
                        if escolha_6 ==  'Sim':



                            escolha_7 = st.radio( f"Você gostaria de alterar o nível da variável categórica: {categorica}?",['Sim', 'Não'])
                            if escolha_7 == 'Sim':
                                data_grouped = data[categorica].unique()
                                lista = list(data_grouped)
                                tamanho = len(lista)

                                if tamanho == 2:
                                    ordem = st.selectbox('Escolha a ordem do nível ', ['Selecione'] + lista,
                                                              key='15')
                                    ordem2 = st.selectbox('Escolha a ordem do nível ', ['Selecione'] + lista,

                                                        key='16')


                                    if ordem != 'Selecione' and ordem2 != "Selecione" and ordem != ordem2:
                                        ordem_desejada = [ordem , ordem2]
                                        options = ["Blues", "BuGn", "Set1","Set2","Set3","viridis","magma", "Pastel1", "Pastel2", "colorblind","Accent", "tab10","tab20","tab20b",'tab20c', "Paired"]

                                        cor_padrão = "Set2"
                                        cores = st.selectbox('Escolha a cor de interesse:', ['Cores']+ options, index = 0)
                                        st.success(f"Você escolheu: {cores}.")
                                        if  cores == 'Cores':
                                            cores = cor_padrão


                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        sns.boxplot(x=Axis_x, y=Eixo_y, hue=dentro_1, order=ordem_desejada,
                                                     palette=cores, data=data, ax=ax)
                                        sns.despine(offset=10, trim=True)
                                        st.pyplot(fig)

                                        st.subheader('Gráfico de barras')
                                        fig3, ax = plt.subplots(figsize=(10, 6))
                                        sns.barplot(x=Axis_x, y=Eixo_y, hue=dentro_1, order=ordem_desejada, palette=cores, errorbar='sd',
                                                    width=0.5, data=data, ax=ax)
                                        plt.ylim(0)
                                        # sns.despine(offset=10, trim=True)
                                        st.pyplot(fig3)

                                    else:
                                        st.warning('Escolha dois níveis diferentes')



                            escolha_8 = st.radio(f"Você gostaria de alterar os níveis da variável categórica:{categorica_2}", ['Sim', 'Não'])
                            if escolha_8 ==  'Sim':
                                data_grouped = data[categorica_2].unique()

                                lista = list(data_grouped)
                                tamanho = len(lista)

                                ordem3= st.selectbox('Escolha a ordem do nível1 ', ['Selecione'] + lista,
                                                     key='10')
                                ordem4 = st.selectbox('Escolha a ordem do nível2 ', ['Selecione'] + lista,

                                                      key='11')

                                ordem5 = st.selectbox('Escolha a ordem do nível3 ', ['Selecione'] + lista,

                                                      key='12')

                                ordem6 = st.selectbox('Escolha a ordem do nível4 ', ['Selecione'] + lista,

                                                      key='13')

                                if ordem3 != 'Selecione' and ordem4 != "Selecione" and ordem5 != "Selecione" and ordem6 != "Selecione" :
                                    ordem_desejada2 = [ordem3, ordem4, ordem5, ordem6]
                                    nome_eixo_y = st.text_input("Digite o nome que você quer para o eixo Y:",
                                                                value=Eixo_y)
                                    nome_eixo_x = st.text_input("Digite o nome que você quer para o eixo X:", value = Axis_x)
                                    with st.spinner("Por favor, aguarde..."):
                                        st.subheader(f"Gráfico de interação  {categorica} e {categorica_2}")
                                        fig, ax = plt.subplots(figsize=(14, 8))
                                        sns.boxplot(x=Axis_x, y=Eixo_y, hue=dentro_1,order=ordem_desejada,hue_order= ordem_desejada2, palette="Set2", data=data, ax=ax)
                                        ax.set_ylabel(nome_eixo_y, fontsize=14, weight='bold')
                                        ax.set_xlabel(nome_eixo_x, fontsize=14, weight='bold')
                                        sns.despine(offset=10, trim=True)

                                        st.pyplot(fig)




                                    # Salvar a figura em um arquivo PNG
                                    fig.savefig(f"Gráfico de interação {categorica} e {categorica_2}.png",dpi=300,  bbox_inches='tight')  # Salva a figura como .png

                                    # Cria um botão para download
                                    with open(f"Gráfico de interação {categorica} e {categorica_2}.png", "rb") as f:
                                        st.download_button(
                                            label="Baixar o gráfico",  # Nome do botão
                                            data=f,  # Dados do arquivo
                                            file_name=f"Gráfico de interação {categorica} e {categorica_2}.png",  # Nome do arquivo a ser baixado
                                            mime="image/png"  # Tipo MIME do arquiv
                                        )


                                        #grráfico de barras e download
                                        st.subheader(f'Gráfico de barras interação {categorica} e {categorica_2}')
                                        fig2, ax = plt.subplots(figsize=(14, 8))
                                        sns.barplot(x=Axis_x, y=Eixo_y, hue=dentro_1, order=ordem_desejada,
                                                    hue_order=ordem_desejada2, palette= cores,linewidth = 1, edgecolor = 'black', data=data,width = 0.5,  ax=ax,
                                                    errorbar='sd')
                                        ax.set_ylabel(nome_eixo_y, fontsize=14, weight='bold')
                                        ax.set_xlabel(nome_eixo_x, fontsize=14, weight='bold')
                                        plt.ylim(0)
                                        st.pyplot(fig2)

                                        # Salvar a figura em um arquivo PNG
                                        fig2.savefig(f"Gráfico de interação barras {categorica} e {categorica_2}.png", dpi=300,
                                                    bbox_inches='tight')  # Salva a figura como .png

                                        # Cria um botão para download
                                        with open(f"Gráfico de interação barras {categorica} e {categorica_2}.png", "rb") as f:
                                            st.download_button(
                                                label="Baixar o gráfico",  # Nome do botão
                                                data=f,  # Dados do arquivo
                                                file_name=f"Gráfico de interação barras {categorica} e {categorica_2}.png",
                                                # Nome do arquivo a ser baixado
                                                mime="image/png"  # Tipo MIME do arquivo

                                            )











                            else:


                                    st.warning('Escolha dois níveis diferentes')


                        escolha_10 = st.radio('Você gostaria de ver os gráfico sem interação?',['Sim', 'Não'])
                        if escolha_10 == 'Sim':
                            st.subheader(f'Gráfico {categorica_2} ')
                            fig, ax = plt.subplots(figsize=(14, 8))
                            sns.boxplot( y=Eixo_y, hue=dentro_1,
                                        hue_order=ordem_desejada2, palette=cores,  data=data, ax=ax)
                            ax.set_ylabel(nome_eixo_y, fontsize=14, weight='bold')
                            sns.despine(offset=10, trim=True)
                            st.pyplot(fig)

                            fig.savefig(f"Gráfico {categorica_2}.png",dpi=300,
                                        bbox_inches='tight')  # Sem espaço antes de .png

                            with open(f"Gráfico {categorica_2}.png", "rb") as f:
                                st.download_button(
                                    label="Baixar o gráfico",
                                    data=f,
                                    file_name=f"Gráfico {categorica_2}.png",
                                    mime="image/png"

                                )



                                #gráfico de barras:
                                st.subheader(f'Gráfico {categorica_2} ')
                                fig8, ax = plt.subplots(figsize=(14, 8))
                                sns.barplot (y=Eixo_y, hue=dentro_1,
                                            hue_order=ordem_desejada2, palette=cores, linewidth = 1, edgecolor = 'black',width = 0.4, data=data, ax=ax)
                                ax.set_ylabel(nome_eixo_y, fontsize=14, weight='bold')
                                plt.ylim(0)
                                st.pyplot(fig8)

                                fig8.savefig(f"Gráfico de barras {categorica_2}.png", dpi=300,
                                            bbox_inches='tight')  # Sem espaço antes de .png

                                with open(f"Gráfico de barras {categorica_2}.png", "rb") as f:
                                    st.download_button(
                                        label="Baixar o gráfico",
                                        data=f,
                                        file_name=f"Gráfico de barras {categorica_2}.png",
                                        mime="image/png"

                                    )






                            st.subheader(f"Gráfico {categorica}")

                            fig, ax = plt.subplots(figsize=(14, 8))
                            sns.boxplot(x=Axis_x, y=Eixo_y, order=ordem_desejada,
                                        palette= cores, data=data, ax=ax)
                            ax.set_ylabel(nome_eixo_y, fontsize=14, weight='bold')
                            ax.set_xlabel(nome_eixo_x, fontsize=14, weight='bold')
                            sns.despine(offset=10, trim=True)
                            st.pyplot(fig)

                            # Salvar a figura com nome seguro
                            fig.savefig(f"Gráfico {categorica}.png", dpi=300,bbox_inches='tight')

                            # Botão de download
                            with open(f"Gráfico {categorica}.png", "rb") as f:
                                st.download_button(
                                    label="Baixar o gráfico",
                                    data=f,
                                    file_name=f"Gráfico {categorica}.png",
                                    mime="image/png"
                                )



                                #gráfico de barras:

                                st.subheader(f"Gráfico de barras {categorica}")

                                fig11, ax = plt.subplots(figsize=(14, 8))
                                sns.barplot(x=Axis_x, y=Eixo_y, order=ordem_desejada,
                                            palette=cores,linewidth = 1, edgecolor = 'black',width = 0.4, data=data, ax=ax)
                                ax.set_ylabel(nome_eixo_y, fontsize=14, weight='bold')
                                ax.set_xlabel(nome_eixo_x, fontsize=14, weight='bold')
                                #sns.despine(offset=10, trim=True)
                                plt.ylim(0)
                                st.pyplot(fig11)

                                # Salvar a figura com nome seguro
                                fig11.savefig(f"Gráfico barras2 {categorica}.png", dpi=300, bbox_inches='tight')

                                # Botão de download
                                with open(f"Gráfico barras2 {categorica}.png", "rb") as f:
                                    st.download_button(
                                        label="Baixar o gráfico",
                                        data=f,
                                        file_name=f"Gráfico barras2 {categorica}.png",
                                        mime="image/png"
                                    )

                    with tab3:
                        st.header(f"Pressupostos da ANOVA ")
                        st.write(f'Modelo completo: {categorica}:{categorica_2}')
                        st.write(f"Parâmetro: {continua}")
                        st.subheader('Teste de normalidade de Shapiro Wilk')
                        st.write('H0: Os resíduos seguem uma distribuição normal ')
                        st.write('Se P < 0.05, então rejeita H0 : O resíduos não segue uma distribuição normal ')



                        formula = f'{continua}~{categorica_2}*{categorica}'
                        # print(formula)
                        # modelo
                        model = smf.ols(formula, data= data).fit()
                        df_resid = data.copy()
                        df_resid['Residuos2'] = model.resid
                        stat, p_valor = shapiro(df_resid['Residuos2'])
                        if p_valor > 0.05:
                            reject = 'Não rejeita a H0'
                            decisao= 'Os resíduos  seguem uma distribuição  normal '
                            st.success(f' P-valor =  {p_valor}')
                            st.success(f'Decisão {reject}')
                            st.success(decisao )
                        else:
                            reject = 'Rejeita H0 '
                            decisao = 'Os resíduos não  seguem uma distribuição  normal '
                            st.success(f' P-valor =  {p_valor}')
                            st.success(f'Decisão {reject}')
                            st.success(decisao)
                        st.subheader('Curva de distribuição KDE')
                        # plotar a curva de KDE
                        fig5, ax = plt.subplots()
                        sns.kdeplot(data=df_resid, x= 'Residuos2', fill=True, alpha=0.3)
                        ax.set_title(f"Curva de KDE para visualização de normalidade do modelo {categorica}-{categorica_2}")
                        plt.axvline(0, color='red', linestyle='dashed', linewidth=1)  # Linha central em 0
                        # sns.stripplot(x=zscore, color='black', jitter=True, alpha=0.5, ax=ax)
                        st.pyplot(fig5)

                        # Anderson Darling test
                        # Teste de normalidade de Anderson darling


                        st.header("Teste de Normalidade dos resíduos ")
                        st.subheader('Anderson Darling ')
                        st.write(f'H0: Os resíduos do modelo: {categorica}-{categorica_2} seguem distribuição normal ')
                        st.write('H0: Se valor crítico > valor estatístico, então não rejeita H0')
                        test = anderson(df_resid['Residuos2'], dist='norm')
                        critical_value = test.critical_values[2]  # O valor crítico para o nível de 5%

                        if test.statistic > critical_value:
                            reject2 = 'Rejeita H0'
                            resultado = "Os resíduos não seguem uma distribuição normal "
                        else:
                            reject2 = 'Não rejeita H0'
                            resultado = 'Os resíduos seguem uma distribuição normal '

                        # Exibindo os resultados
                        print(linha)
                        st.success(f' Valor crítico: {critical_value} ')
                        st.success(f'Estatística do teste:  {test.statistic}')
                        st.success(reject2)
                        st.success(resultado)

                        #Homogneidade da variância:
                        st.header('Homogeneidade de variância')
                        st.subheader("Teste de levene")
                        st.write('H0: A variãncia dos grupos comparados são iguais a um nível de significância de 5%')
                        st.write('Se p-valor <0.05, então rejeita H0 e os resíduos não seguem distribuição normal')
                        agrupamento = df_resid.groupby(categorica)
                        grupo = []
                        for nome, dados_grupo in agrupamento:
                            # print(dados_grupo['Residuos'].values)
                            grupo.append(dados_grupo['Residuos2'].values)
                            # print(x)
                        stat, p_value = stats.levene(*grupo)
                        if p_value < 0.05:
                            reject = 'Rejeita a H0'
                            homoge_neo = 'não são '
                            resposta = 'Os resíduos não seguem uma distribuição normal'
                        else:
                            reject = 'Não rejeita H0'
                            homoge_neo = 'são '
                            resposta = 'Os resíduos seguem uma distribuição normal '
                        st.success(
                            f' P-valor :  {p_value}' )
                        st.success(f"A variância dos níveis comparados {homoge_neo} homogêneos")
                        st.success(f'Decisão:  {reject} ')
                        st.success(resposta)




                        #teste de barlett
                        st.subheader('Teste de barlett para homogeneidade de variância')
                        st.write('H0: A variãncia dos grupos comparados são iguais a um nível de significância de 5%')
                        st.write('Se p-valor <0.05, então rejeita H0 e os resíduos não seguem distribuição normal')
                        stat, p = stats.bartlett(*grupo)
                        if p_value < 0.05:
                            reject = 'Rejeita a H0'
                            homoge_neo = 'não são '
                            decisão = 'Os resíduos não são homogênos(iguais)'
                        else:
                            reject = 'Não rejeita H0'
                            homoge_neo = 'são '
                            decisao = ' As variâncias dos resíduos são homogêneos '

                        st.success(f'P-valor :  {p_value}')
                        st.success(f'a variância dos níveis comparados {homoge_neo} homogêneos')
                        st.success(reject)
                        st.success(decisao)

                        st.header('ANOVA')
                        model = smf.ols(formula, data=data).fit()
                        anova_table = anova_lm(model)
                        st.dataframe(anova_table)
                        st.write(f"R squared adjusted: {model.rsquared_adj}")
                        p_value = anova_table['PR(>F)'][2]

                        if p_value < 0.05:
                            print(f'Análise de tukey para o moddelo {categorica}: {categorica_2}')
                            df_clean2 = data.copy()
                            df_clean2['Combinação'] = df_clean2[categorica].astype(str) + ':' + df_clean2[
                                categorica_2].astype(str)
                            # Garantindo que a coluna Combinação seja categórica
                            df_clean2['Combinação'] = pd.Categorical(df_clean2['Combinação'])

                            mc = MultiComparison(df_clean2.iloc[:, 2], df_clean2['Combinação'])
                            tukey_test = mc.tukeyhsd(alpha=0.05)
                            st.dataframe(tukey_test.summary())
                        else:
                            st.warning('O testde tukey não pode ser mostrado, pois não houve um p-valor significativo na interação')
                            st.warning(f'O p-valor foi de {p_value}')
                            st.warning('Que está acima de 0.05')
                            anova2 = st.radio('Você deseja fazer a análise dos fatores isolados?', ['Sim','Não'])

                            if anova2 == 'Sim':
                                st.header('Análise dos fatores isolados')
                                st.subheader(f'Modelo: {categorica} +{categorica_2}')
                                formula = f'{continua}~{categorica}+{categorica_2}'
                                model = smf.ols(formula, data=data).fit()
                                anova_table1 = anova_lm(model)
                                st.dataframe(anova_table1)
                                st.write(f"R squared adjusted: {model.rsquared_adj}")
                                p_value1 = anova_table['PR(>F)'][1]
                                p_value2 = anova_table['PR(>F)'][0]

                                if p_value1 < 0.05:
                                    st.subheader(f'Análise de tukey para  o fator   {categorica}')

                                    categorico1 = pd.Categorical(data.iloc[:,0]
                                       )  # transformando a primeira coluna em categórica

                                    mc = MultiComparison(data.iloc[:, 2], categorico1)
                                    tukey_test1 = mc.tukeyhsd(alpha=0.05)
                                    st.dataframe(tukey_test1.summary())
                                else:
                                    st.warning(f'O valor de p para o fator {categorica} não foi significativo')
                                    st.warning(p_value1)
                                    st.warning('Não prossegue a análise de contraste')

                                if p_value2< 0.05:

                                    st.subheader(f'Análise de tukey para  o fator  {categorica_2}')
                                    categorico2 = pd.Categorical(data.iloc[:,1]
                                       )  # transforma a segunda coluna em categórica

                                    mc2= MultiComparison(data.iloc[:, 2], categorico2)
                                    tukey_test2 = mc2.tukeyhsd(alpha=0.05)
                                    st.dataframe(tukey_test2.summary())
                                else:
                                    st.warning(f'O valor de p para o fator {categorica_2} não foi significativo')
                                    st.warning(p_value2)
                                    st.warning('Não prossegue a análise de contraste')

    escolhas = []
    if variavel == 3:
        categorica= st.selectbox('Escolha as variável categórica',['Selecione'] + chaves1,key = '6')
        if categorica != 'Selecione':
            st.success(f"Você escolheu a variável categórica: {categorica}")
        categorica_2 = st.selectbox('Escolha as variável categórica', ['Selecione'] + chaves1, key = '7')
        if categorica_2 != 'Selecione':
            st.success(f"Você escolheu a variável categórica: {categorica_2}")
        categorica_3 = st.selectbox('Escolha as variável categórica', ['Selecione'] +chaves1, key='8')
        if categorica_3 != 'Selecione':
            st.success(f"Você escolheu a variável categórica: {categorica_3}")
        continua = st.selectbox('Escolha a variável contínua', ['Selecione'] +chaves1, key = '9')
        if continua  != 'Selecione':
            st.success(f"Você escolheu a variável contínua: {continua}")

        if categorica != 'Selecione' and  categorica_2 != 'Selecione' and categorica_3  and continua  != 'Selecione':

            escolhas.append(categorica)
            escolhas.append(categorica_2)
            escolhas.append(categorica_3)
            escolhas.append(continua)

            data = data[escolhas]  # escolhi e armazenei as variáveis que quero trabalhar
            st.write(data)
            data_na = data.isna().sum()
            # fazer uma função aqui! def retirarana(data na , data) e chamo novamente em outro lugar
            if data_na.sum() == 0:
                st.write(f'Você Não tem **NA** nas  variáveis de seus dados  ')
                st.dataframe(data_na)
            else:
                st.write(f'Você  tem **NA** nas  variáveis de seus dados  ')
                st.dataframe(data_na)
                st.write('Você gostaria de retira  as **NAs** ou substituir por valores médios?')
                escolha_2 = st.radio("Você deseja ?", ["Substituir por Valores médios", "Retirar Na"])
                if escolha_2 == "Substituir por Valores médios":
                    data = data.fillna(data.median(numeric_only=True))
                    st.write('Dados com valores médios substituidos no lugar de NA')
                    st.dataframe(data)
                else:
                    data = data.dropna(axis=1)
                    st.dataframe(data)  # manter o mes



            cv = data.loc[:, continua].values  # transforma em array numpy  e pega os valores, para o cálculo
            # cálculo do cv
            cv2 = np.std(cv) / np.mean(cv) * 100
            st.write(f"CV% = {cv2}")

            st.subheader('Z-score ')
            zscore = (data[continua] - np.mean(data[continua])) / np.std(data[continua])
            data2 = data.copy()
            data2['zscore'] = zscore
            st.write(data2)
            # print(zscore)

            # plotar a curva de KDE
            fig2, ax = plt.subplots()
            sns.kdeplot(data=data2, x='zscore', fill=True, alpha=0.3)
            ax.set_title("Curva de KDE para visualização de normalidade ")
            plt.axvline(0, color='red', linestyle='dashed', linewidth=1)  # Linha central em 0
            # sns.stripplot(x=zscore, color='black', jitter=True, alpha=0.5, ax=ax)
            st.pyplot(fig2)

            # Plotar o boxplot dos z-scores
            fig, ax = plt.subplots()
            sns.boxplot(x=zscore, ax=ax)
            sns.stripplot(x=zscore, color='black', jitter=True, alpha=0.5, ax=ax)
            ax.set_title("Boxplot dos Z-Scores")
            st.pyplot(fig)

            # cálculo de outliers:
            st.subheader('Outlier ')
            st.write(
                'O cálculo de outlier consiste em identificar os dados que estão acima ou abaixo  de 3 desvios padrão do Z-score e utiliza-se o método do IQR')

            Q1 = data.loc[:, continua].quantile(0.25)
            Q3 = data.loc[:, continua].quantile(0.75)
            # print(Q1)
            # print(Q3)

            IQR = Q3 - Q1
            LS = Q3 + 1.5 * IQR
            LI = Q1 - 1.5 * IQR
            print()
            linha = 70 * '='
            print(linha)

            print(linha)

            # Outliers acima e abaixo:
            st.write('Limite superior = ', LS)
            acima = data[(data.loc[:, continua] > LS)]
            if acima.empty:  # Usa-se empty, porque estamos tratando de um dataframe
                st.write('Você não tem outliers acima do limite superior  ')
                st.write(acima)
            else:
                st.write('Você tem alguns outliers acima do limite superior')
                st.write(acima)

            st.write("limite inferior = ", LI)
            abaixo = data[(data.loc[:, continua] < LI)]
            if abaixo.empty:  # Usa-se empty, porque estamos tratando de um dataframe
                st.write('Você não tem outliers abaixo do limite inferior  ')
                st.write(abaixo)
            else:
                st.write('Você tem alguns outliers abaixo  do limite inferior')
                st.write(abaixo)

            escolha_3 = st.radio("Você deseja retirar os outliers ?", ["SIM", "Não "], horizontal=True)
            if escolha_3 == 'SIM':
                data = data[(data[continua] < LS) & (data[continua] > LI)]
                st.success('os outliers foram tirados com sucesso ')
                escolha_4 = st.radio("Você gostaria de ver os dados sem outliers?", ['Sim', 'Não'])
                if escolha_4 == 'Sim':
                    st.write('Seus dados sem outliers')
                    st.dataframe(data)
                escolha_5 = st.radio('Você deseja ver os gráficos boxplot e KDE', ['Sim', 'Não'], horizontal=True)
                if escolha_5 == 'Sim':
                    st.subheader('Z-score ')
                    zscore = (data[continua] - np.mean(data[continua])) / np.std(data[continua])
                    data['zscore'] = zscore

                    # plotar a curva de KDE
                    fig2, ax = plt.subplots()
                    sns.kdeplot(data=data, x='zscore', fill=True, alpha=0.3)
                    ax.set_title("Curva de KDE para visualização de normalidade ")
                    plt.axvline(0, color='red', linestyle='dashed', linewidth=1)  # Linha central em 0
                    # sns.stripplot(x=zscore, color='black', jitter=True, alpha=0.5, ax=ax)
                    st.pyplot(fig2)

                    st.dataframe(data)
                    # Plotar o boxplot dos z-scores
                    fig, ax = plt.subplots()
                    sns.boxplot(x=zscore, ax=ax)
                    sns.stripplot(x=zscore, color='black', jitter=True, alpha=0.5, ax=ax)
                    ax.set_title("Boxplot dos Z-Scores")
                    st.pyplot(fig)

                    st.write('Análise descritiva dos seus dados ')
                    data_grouped = data.groupby(data.columns[0:variavel].tolist()).describe()
                    st.dataframe(data_grouped)

    else:
            st.warning('Você não selecionou as suas variáveis, meu caro!!"')

                    #with tab2:
























