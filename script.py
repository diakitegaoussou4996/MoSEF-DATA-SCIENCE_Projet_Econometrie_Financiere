import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import itertools
from statsmodels.tsa.stattools import adfuller, kpss 
from arch.unitroot import PhillipsPerron
import plotly.express as px
import os
import warnings
warnings.filterwarnings("ignore")
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox
import pykalman
from pykalman import KalmanFilter

def main():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import itertools
    from statsmodels.tsa.stattools import adfuller, kpss 
    from arch.unitroot import PhillipsPerron
    import plotly.express as px
    import os
    from filterpy.kalman import KalmanFilter
    from scipy.stats import norm
    from statsmodels.stats.stattools import jarque_bera
    from statsmodels.stats.diagnostic import acorr_ljungbox
    import pykalman
    from pykalman import KalmanFilter

    import warnings
    warnings.filterwarnings("ignore")
    
    def load_financial_data(file_path):
        # Extraction du nom de l'actif
        asset_name = os.path.splitext(os.path.basename(file_path))[0].upper()

        # Lecture du fichier csv
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)

        # Préfixage des colonnes avec le nom de l'actif
        df = prefix_close(df, asset_name+'_')

        # Retourne un dictionnaire avec le nom de l'actif et son dataframe correspondant
        return {asset_name: df}

    def prefix_close(df, table_name):
        # on garde que le prix de clôture
        df = pd.DataFrame(df['Close'])
        
        # On prend le log
        df = np.log(df)
        
        # On préfixe la variable 
        df = df.add_prefix(table_name)
        return df


    # Chemin vers le dossier contenant les fichiers csv
    folder_name = 'AMD_MSFT'
    folder_path = f'Data/{folder_name}/'

    # Séparer les noms des fichiers en utilisant le caractère '_'
    file_names = folder_name.split('_')

    # Afficher les noms des fichiers séparés
    print('Les actifs sont :', file_names)

    # Liste des fichiers csv dans le dossier
    csv_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]

    # Chargement des données pour chaque fichier csv et stockage dans un dictionnaire
    assets_dict = {}
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        assets_dict.update(load_financial_data(file_path))

    assets1 = assets_dict[file_names[0]]
    assets2 = assets_dict[file_names[1]]

    titres = merged_df = pd.merge(assets1, assets2, on='Date')
    print(titres.head(4))

    # Définir les dates de l'échantillon et des données hors échantillon
    def choose_period(folder_name):
        if folder_name == 'XOM_LUV':
            start_sample = '2011-09-22'
            end_sample   = '2012-09-20'
            start_oos    = '2012-09-21'
            end_oos      = '2013-03-26'
            index_folder_name = 'index'
        else :
            start_sample = '2021-11-01'
            end_sample   = '2022-11-01'
            start_oos    = '2022-11-01'
            end_oos      = '2023-05-01'
            index_folder_name = 'index_test'
        return start_sample, end_sample, start_oos, end_oos, index_folder_name
    
    start_sample, end_sample, start_oos, end_oos, index_folder_name = choose_period(folder_name)

    # Importer les données de l'indice
    indice = pd.read_csv(f'Data/{index_folder_name}.csv', index_col='Date', parse_dates=True)
    indice = np.log(indice)
    indice.drop('Unnamed: 0', axis=1, inplace=True)
    BVSP = pd.DataFrame(indice['^BVSP'].values, index=indice.index, columns=['BVSP'])
    GSPC = pd.DataFrame(indice['^GSPC'].values, index=indice.index, columns=['GSPC'])


    # ___________________________________  Visualisation _______________________
    import plotly.express as px

    def plot_assets(df):
        for col in df.columns:
            fig = px.line(df[col], x=df.index, y=col, title=col, labels={'x': 'Date', 'y': col})
            fig.update_layout(legend_title_text=f"{col} Legend")
            fig.show()

    plot_assets(titres)

    # Créer un scatterplot coloré par la date
    def draw_date_coloured_scatterplot(etfs, prices):
        """
        Create a scatterplot of the two ETF prices, which is
        coloured by the date of the price to indicate the 
        changing relationship between the sets of prices    
        """
        # Create a yellow-to-red colourmap where yellow indicates
        # early dates and red indicates later dates
        plen = len(prices)
        colour_map = plt.cm.get_cmap('YlOrRd')    
        colours = np.linspace(0.1, 1, plen)
        
        # Create the scatterplot object
        scatterplot = plt.scatter(
            prices[etfs[0]], prices[etfs[1]], 
            s=30, c=colours, cmap=colour_map, 
            edgecolor='k', alpha=0.8
        )
        
        # Add a colour bar for the date colouring and set the 
        # corresponding axis tick labels to equal string-formatted dates
        colourbar = plt.colorbar(scatterplot)
        colourbar.ax.set_yticklabels(
            [str(p.date()) for p in prices[::plen//9].index]
        )
        plt.xlabel(prices.columns[0])
        plt.ylabel(prices.columns[1])
        plt.show()

    # Appeler la fonction pour afficher le scatterplot
    draw_date_coloured_scatterplot(titres.columns, titres)


    plt.figure(figsize=(7, 4))
    mask = np.triu(np.ones_like(titres.corr(), dtype=bool))
    sns.heatmap(titres.corr(), annot=True, vmin=-1, vmax=1, cmap='coolwarm', mask=mask, square=True)
    plt.show()


    print('______________________   Test de cointégration ')
    # Sélectionner les données de l'échantillon et des données hors échantillon
    train_data= titres.loc[start_sample:end_sample]
    test_data = titres.loc[start_oos:end_oos]

    series1 = train_data[train_data.columns[0]]
    series2 = train_data[train_data.columns[1]]

    mean_series1 = pd.Series.mean(series1)
    mean_series2 = pd.Series.mean(series2)

    if mean_series1 > mean_series2:
        max_price = train_data[train_data.columns[0]]
        min_price = train_data[train_data.columns[1]]
    else:
        max_price = train_data[train_data.columns[1]]
        min_price = train_data[train_data.columns[0]]


    # Sélectionner les colonnes pour la régression
    series1 = train_data[train_data.columns[0]]
    series2 = train_data[train_data.columns[1]]

    # Régression de series1 sur series2
    model = sm.OLS(max_price, sm.add_constant(min_price))
    results = model.fit()

    print(results.summary())

    # Récupération des résidus de la régression
    residuals = results.resid

    # Test de la stationnarité des résidus
    adf_test = adfuller(residuals)
    p_value = adf_test[1]
    print(p_value)

    if p_value < 0.05:
        print("Les séries sont cointégrées.")
    else:
        print("Les séries ne sont pas cointégrées.")


    print('Calcul de spread')
    # Extraire les paramètres de la régression
    alpha, beta = results.params

    print("Les paramètres de la régression OLS")
    print('α :', alpha, '\n' 'β :',beta)

    # Calculer le spread

    spread  = pd.DataFrame(max_price - alpha - beta*min_price, columns=['spread'])
    #train_df= train_data.concat(spread, axis=1)
    # Afficher le spread
    print("\nLes  premières lignes du Spread")
    print(spread.head(4))

    fig_title = f'Spread {folder_name}'
    fig = px.line(spread, x=spread.index, y='spread', title=fig_title,
                labels={'x': 'Date', 'y': 'Spread'})
    fig.update_layout(legend_title_text ='Spread Legend')
    fig.show()


    # Test de la stationnarité des résidus
    adf_spread_test = adfuller(spread)
    adf_spread_test_p_value = adf_spread_test[1]
    print(adf_spread_test_p_value)

    if adf_spread_test_p_value < 0.05:
        print("Le Spread est stationnaire.")
    else:
        print("Le Spread n'est pas stationnaire.")

    print('Modeles : AR(1), AR(2), ARMA(1,1)')

    def fit_models(spread):
        # Modèle AR(1)
        ar1_model = sm.tsa.AutoReg(spread, lags=1)
        ar1_results = ar1_model.fit()

        # Obtention des paramètres estimés
        alpha_ar1, beta_ar1 = ar1_results.params

        # Obtention de l'écart-type des erreurs de mesure du spread
        sigma_ar1 = np.sqrt(ar1_results.sigma2)

        # Stockage des résultats
        ar1_dict = {'model': ar1_results, 
                    'alpha': alpha_ar1, 
                    'beta': beta_ar1, 
                    'sigma': sigma_ar1}
        
        # Modèle AR(2)
        ar2_model = sm.tsa.AutoReg(spread, lags=2)
        ar2_results = ar2_model.fit()

        # Obtention des paramètres estimés
        alpha_ar2, beta_ar2, gamma_ar2 = ar2_results.params

        # Obtention de l'écart-type des erreurs de mesure du spread
        sigma_ar2 = np.sqrt(ar2_results.sigma2)

        # Stockage des résultats
        ar2_dict = {'model': ar2_results, 
                    'alpha': alpha_ar2, 
                    'beta': beta_ar2, 
                    'gamma': gamma_ar2, 
                    'sigma': sigma_ar2}

        # Modèle ARIMA(1,1,0)
        arima_model = sm.tsa.ARIMA(spread, order=(1,1,0))
        arima_results = arima_model.fit()

        # Obtention des paramètres estimés
        alpha_arima, beta_arima = arima_results.params

        # Obtention de l'écart-type des erreurs de mesure du spread
        sigma_arima = np.sqrt(arima_results.resid.var())

        # Obtention de l'écart-type des erreurs de mesure de la différence de spread
        sigma_diff_arima = np.sqrt(arima_results.resid.diff().dropna().var())

        # Stockage des résultats
        arima_dict = {'model': arima_results, 
                    'alpha': alpha_arima, 
                    'beta': beta_arima, 
                    'sigma': sigma_arima, 
                    'sigma_diff': sigma_diff_arima}
        
        # Stockage des résultats des trois modèles dans un dictionnaire
        models_dict = {'AR(1)': ar1_dict, 'AR(2)': ar2_dict, 'ARIMA(1,1,0)': arima_dict}

        return models_dict
    
    models_dict = fit_models(spread)
    print(models_dict['AR(1)']['model'].summary())
    print(models_dict['AR(2)']['model'].summary())
    print(models_dict['ARIMA(1,1,0)']['model'].summary())



    print('                                    Choix du modele')
    def compare_models(models_dict, spread):
        # Initialisation du tableau de résultats
        results_table = pd.DataFrame(columns=['Log-Likelihood', 'Pseudo R2', 'MSE', 'AIC', 'BIC', 'Mean', 'Variance', 'Ljung-Box Q', 'Ljung-Box p-value', 'Jarque-Bera', 'Jarque-Bera p-value'])
        
        # Calcul des métriques pour chaque modèle
        for name, model in models_dict.items():
            # Log-likelihood
            ll = model['model'].llf
            
            # Pseudo R2
            nobs = len(spread)
            p = len(model['model'].params)
            r2 = 1 - ((ll / nobs) * -2)**(1/nobs-p)
            
            # MSE
            resid = model['model'].resid
            mse = resid.var()
            
            # AIC
            aic = model['model'].aic
            
            # BIC
            bic = model['model'].bic
            
            # Moyenne
            mean = np.mean(spread)
            
            # Variance
            variance = np.var(spread)
            
            # Ljung-Box
            ljung_box_q, ljung_box_pvalue = acorr_ljungbox(resid, lags=[10])
            
            # Jarque-Bera
            jb, jb_pvalue, _, _ = jarque_bera(resid)
            
            # Stockage des résultats dans le tableau
            results_table.loc[name] = [ll, r2, mse, aic, bic, mean, variance, ljung_box_q[0], ljung_box_pvalue[0], jb, jb_pvalue]

        return results_table
    print(compare_models(models_dict, spread))

    # Obtention des modèles
    models_dict = fit_models(spread)

    # Initialisation des dictionnaires pour stocker les mesures de chaque modèle
    ll_dict = {}
    aic_dict = {}
    bic_dict = {}

    # Calcul des mesures pour chaque modèle
    for name, model in models_dict.items():
        # Log-vraisemblance
        ll_dict[name] = model['model'].llf
        
        # AIC
        aic_dict[name] = model['model'].aic
        
        # BIC
        bic_dict[name] = model['model'].bic

    # Affichage des mesures pour chaque modèle
    print("Log-vraisemblance: ", ll_dict)
    print("AIC: ", aic_dict)
    print("BIC: ", bic_dict)

    # Sélection du meilleur modèle
    best_model_name = min(aic_dict, key=aic_dict.get)
    print("Le meilleur modèle est : ", best_model_name)

    # 4. Estimation du spread par modèle d'espace d'état
    kf = KalmanFilter(n_dim_obs=1, n_dim_state=1,
                  initial_state_mean=spread.iloc[0],
                  initial_state_covariance=1,
                  transition_matrices=[1],
                  observation_matrices=[1],
                  observation_covariance=1,
                  transition_covariance=0.01)
    

    print('                    Stratégie de traiding')
    # Appliquer un filtre de Kalman aux données de ratio et retourner la moyenne et la covariance de ces données
    state_means, state_covs = kf.filter(spread.values)

    # Ajouter la moyenne obtenue à partir du filtre de Kalman au dataframe data_train                                 
    train_data["mean_spread"]= state_means.squeeze()

    # Ajouter la covariance obtenue à partir du filtre de Kalman au dataframe data_train
    train_data['cov_spread'] = state_covs.squeeze()

    # Ajouter l'écart-type obtenue à partir du filtre de Kalman au dataframe data_train
    train_data['std_spread'] = train_data['mean_spread'].std()

    # on ajoute le spread dans la colonne 
    train_data["spread"]= spread

    # Calculer la moyenne mobile sur 5 périodes des valeurs du spread et l'ajouter au dataframe data
    train_data['ma'] = train_data['spread'].mean()
    #.rolling(window=5)
    # Calculer le score Z et ajouter la valeur obtenue à data_train
    train_data["z_score"] = (train_data['mean_spread'] - train_data['ma']) / train_data['std_spread']


    # Initialise positions as zero
    train_data['position_1'] = np.nan
    train_data['position_2'] = np.nan


    # Generate buy, qelltrain_data and square off signals as: z<-1 buy, z>1 sell and -1<z<1 liquidate the position
    for i in range (train_data.shape[0]):
        if train_data['z_score'].iloc[i] < -1:
            train_data['position_1'].iloc[i] = 1
            train_data['position_2'].iloc[i] = -round(train_data['mean_spread'].iloc[i],0)
        if train_data['z_score'].iloc[i] > 1:
            train_data['position_1'].iloc[i] = -1
            train_data['position_2'].iloc[i] = round(train_data['mean_spread'].iloc[i],0)
        if (abs(train_data['z_score'].iloc[i]) < 1) & (abs(train_data['z_score'].iloc[i]) > 0):
            train_data['position_1'].iloc[i] = 0
            train_data['position_2'].iloc[i] = 0
        
        
    # Calculate returns
    train_data['returns'] = ((train_data[max_price.name]-train_data[max_price.name].shift(1))/train_data[max_price.name].shift(1))*train_data['position_1'].shift(1)+ ((train_data[min_price.name]-train_data[min_price.name].shift(1))/train_data[min_price.name].shift(1))*train_data['position_2'].shift(1)
    train_data['returns'].sum()

    # Calculer le ratio de Sharpe
    sharpe_ratio = np.sqrt(252) * (train_data['returns'].mean() / train_data['returns'].std())

    # Calculer la volatilité
    volatility = train_data['returns'].std()

    # Imprimer les résultats
    print("Ratio de Sharpe: ", sharpe_ratio)
    print("Volatilité: ", volatility)
    # Print the total returns
    print("Total returns: ", train_data['returns'].sum())


    print('                    Evaluation de stratégie ')
    # Sélectionner les données de l'échantillon et des données hors échantillon
    train_BVSP= BVSP.loc[start_sample:end_sample]
    train_GSPC= GSPC.loc[start_sample:end_sample]

    test_BVSP= BVSP.loc[start_oos:end_oos]
    test_GSPC= GSPC.loc[start_oos:end_oos]

    
    train_BVSP['returns_BVSP'] = (train_BVSP['BVSP']-train_BVSP['BVSP'].shift(1))/train_BVSP['BVSP'].shift(1)
    train_BVSP['returns_BVSP'].sum()

    # Calculer le ratio de Sharpe
    sharpe_ratio_BVSP = np.sqrt(252) * (train_BVSP['returns_BVSP'].mean() / train_BVSP['returns_BVSP'].std())

    # Calculer la volatilité
    volatility_BVSP = train_BVSP['returns_BVSP'].std()

    train_GSPC['returns_GSPC'] = (train_GSPC['GSPC']-train_GSPC['GSPC'].shift(1))/train_GSPC['GSPC'].shift(1)
    train_GSPC['returns_GSPC'].sum()

    # Calculer le ratio de Sharpe
    sharpe_ratio_BVSP = np.sqrt(252) * (train_GSPC['returns_GSPC'].mean() / train_GSPC['returns_GSPC'].std())

    # Calculer la volatilité
    volatility_BVSP = train_GSPC['returns_GSPC'].std()

    # Calculer le ratio de Sharpe
    sharpe_ratio = np.sqrt(252) * (train_data['returns'].mean() / train_data['returns'].std())

    # Calculer la volatilité
    volatility = train_data['returns'].std()

    # Calculer le ratio de Sharpe et la volatilité pour chaque ensemble de données
    results = {
        "train_data": {
            "Ratio de Sharpe": np.sqrt(252) * (train_data['returns'].mean() / train_data['returns'].std()),
            "Volatilité": train_data['returns'].std(),
            "Total returns": train_data['returns'].sum()
        },
        "train_BVSP": {
            "Ratio de Sharpe": np.sqrt(252) * (train_BVSP['returns_BVSP'].mean() / train_BVSP['returns_BVSP'].std()),
            "Volatilité": train_BVSP['returns_BVSP'].std(),
            "Total returns": train_BVSP['returns_BVSP'].sum()
        },
        "train_GSPC": {
            "Ratio de Sharpe": np.sqrt(252) * (train_GSPC['returns_GSPC'].mean() / train_GSPC['returns_GSPC'].std()),
            "Volatilité": train_GSPC['returns_GSPC'].std(),
            "Total returns": train_GSPC['returns_GSPC'].sum()
        }
    }

    # Créer un DataFrame à partir du dictionnaire
    results_df = pd.DataFrame(results)
    print(results_df)
    print('             Appplication sur le jeux de données test')
    spread_test  = pd.DataFrame(test_data[max_price.name] - alpha - beta*test_data[min_price.name], columns=['spread'])

    fig_title = f'Spread {folder_name}'
    fig = px.line(spread_test, x=spread_test.index, y='spread', title=fig_title,
                labels={'x': 'Date', 'y': 'Spread'})
    fig.update_layout(legend_title_text ='Spread Legend')
    fig.show()

    # Appliquer un filtre de Kalman aux données de ratio et retourner la moyenne et la covariance de ces données
    state_means, state_covs = kf.filter(spread_test.values)

    # Ajouter la moyenne obtenue à partir du filtre de Kalman au dataframe data_train                                 
    test_data["mean_spread"]= state_means.squeeze()

    # Ajouter la covariance obtenue à partir du filtre de Kalman au dataframe data_train
    test_data['cov_spread'] = state_covs.squeeze()

    # Ajouter l'écart-type obtenue à partir du filtre de Kalman au dataframe data_train
    test_data['std_spread'] = test_data['mean_spread'].std()

    # on ajoute le spread dans la colonne 
    test_data["spread"]= spread_test

    # Calculer la moyenne mobile sur 5 périodes des valeurs du spread et l'ajouter au dataframe data
    test_data['ma'] = test_data['spread'].mean()
    #.rolling(window=5)
    # Calculer le score Z et ajouter la valeur obtenue à data_train
    test_data["z_score"] = (test_data['mean_spread'] - test_data['ma']) / test_data['std_spread']


    test_data['position_1'] = np.nan
    test_data['position_2'] = np.nan


    # Generate buy, qelltrain_data and square off signals as: z<-1 buy, z>1 sell and -1<z<1 liquidate the position
    for i in range (test_data.shape[0]):
        if test_data['z_score'].iloc[i] < -1:
            test_data['position_1'].iloc[i] = 1
            test_data['position_2'].iloc[i] = -round(test_data['mean_spread'].iloc[i],0)
        if test_data['z_score'].iloc[i] > 1:
            test_data['position_1'].iloc[i] = -1
            test_data['position_2'].iloc[i] = round(test_data['mean_spread'].iloc[i],0)
        if (abs(test_data['z_score'].iloc[i]) < 1) & (abs(test_data['z_score'].iloc[i]) > 0):
            test_data['position_1'].iloc[i] = 0
            test_data['position_2'].iloc[i] = 0
        
    
        
    # Calculate returns
    test_data['returns'] = ((test_data[max_price.name]-test_data[max_price.name].shift(1))/test_data[max_price.name].shift(1))*test_data['position_1'].shift(1)+ ((test_data[min_price.name]-test_data[min_price.name].shift(1))/test_data[min_price.name].shift(1))*test_data['position_2'].shift(1)
    test_data['returns'].sum()

    print(test_data.head(4))

    
    test_BVSP['returns_BVSP'] = (test_BVSP['BVSP']-test_BVSP['BVSP'].shift(1))/test_BVSP['BVSP'].shift(1)
    test_BVSP['returns_BVSP'].sum()

    # Calculer le ratio de Sharpe
    sharpe_ratio_BVSP = np.sqrt(252) * (test_BVSP['returns_BVSP'].mean() / test_BVSP['returns_BVSP'].std())

    # Calculer la volatilité
    volatility_BVSP = test_BVSP['returns_BVSP'].std()


    test_GSPC['returns_GSPC'] = (test_GSPC['GSPC']-test_GSPC['GSPC'].shift(1))/test_GSPC['GSPC'].shift(1)
    test_GSPC['returns_GSPC'].sum()

    # Calculer le ratio de Sharpe
    sharpe_ratio_GSPC = np.sqrt(252) * (test_GSPC['returns_GSPC'].mean() / test_GSPC['returns_GSPC'].std())

    # Calculer la volatilité
    volatility_GSPC = test_GSPC['returns_GSPC'].std()

    # Calculer le ratio de Sharpe et la volatilité pour chaque ensemble de données
    results = {
        "test_data": {
            "Ratio de Sharpe": np.sqrt(252) * (test_data['returns'].mean() / test_data['returns'].std()),
            "Volatilité": test_data['returns'].std(),
            "Total returns": test_data['returns'].sum()
        },
        "test_BVSP": {
            "Ratio de Sharpe": np.sqrt(252) * (test_BVSP['returns_BVSP'].mean() / test_BVSP['returns_BVSP'].std()),
            "Volatilité": test_BVSP['returns_BVSP'].std(),
            "Total returns": test_BVSP['returns_BVSP'].sum()
        },
        "test_GSPC": {
            "Ratio de Sharpe": np.sqrt(252) * (test_GSPC['returns_GSPC'].mean() / test_GSPC['returns_GSPC'].std()),
            "Volatilité": test_GSPC['returns_GSPC'].std(),
            "Total returns": test_GSPC['returns_GSPC'].sum()
        }
    }

    # Créer un DataFrame à partir du dictionnaire
    results_df = pd.DataFrame(results)
    print(results_df)