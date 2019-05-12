'''******************************************************************************************************
Importing functions
******************************************************************************************************'''
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from bs4 import BeautifulSoup
import os
import urllib.request
'''******************************************************************************************************
Functions
******************************************************************************************************'''

#--------------------------------------------------------------------------------------------------------
#Functio to get information from wikipedia
#--------------------------------------------------------------------------------------------------------
def GettingInfo1(df):
    
    #................................................................
    #Inputs
    #df - Dataframe to store the web scraping information
    #................................................................
    
    #Code:
    
    #Urls with information about affiliated companies with Nintendo, Sony and Microsoft
    urls = ['https://en.wikipedia.org/wiki/List_of_Nintendo_development_teams',
            'https://en.wikipedia.org/wiki/SIE_Worldwide_Studios',
            'https://en.wikipedia.org/wiki/Xbox_Game_Studios']

    #Identifiers for the web scraping
    iden = ['th', 'tr', 'th']

    #Creating empty container for the data
    FirstParties = {}
    NintyComp = []
    SonyComp = []
    MicroComp = []

    #Finding first parties from the web
    for n in range(3):

        with urllib.request.urlopen(urls[n]) as response:
            html = response.read()
        soup = BeautifulSoup(html, 'lxml')

        if n==0:
            for table in soup.findAll('table', {'class':"wikitable"}):
                for item in table.find('tbody').findAll(iden[n]):
                    if item.find('a') != None:
                        NintyComp.append(item.find('a').text.replace(' ', '').lower())
                        FirstParties[item.find('a').text.replace(' ', '').lower()] = 'Nintendo'

        if n==1:
            for table in soup.findAll('table', {'class':"wikitable"}):
                for item in table.find('tbody').findAll(iden[n]):
                    if item.find('a') != None:
                        SonyComp.append(item.find('a').text.replace(' ', '').lower())
                        FirstParties[item.find('a').text.replace(' ', '').lower()] = 'Sony'

        elif n==2:                     
            for table in soup.findAll('div', class_='navbox'):
                for item in table.findAll('tr'):
                    for obj in item.findAll('th', class_='navbox-group'):
                        if obj.text == 'Current subsidiaries' or obj.text == 'Former subsidiaries':
                            for it in obj.findNext('ul'):
                                if it.find('a') != -1:
                                    MicroComp.append(it.find('a').text.replace(' ', '').lower())
                                    FirstParties[it.find('a').text.replace(' ', '').lower()] = 'Microsoft'

    FirstParties['nintendo'] = 'Nintendo'
    FirstParties['sony'] = 'Sony'
    FirstParties['microsoft'] = 'Microsoft'

    #Creating a dictionary to define which companies are first parties or not
    FirstParty = []
    for pub in df.Publisher:
        if type(pub) == float:
            pub = ''
        else:
            pub = pub.replace(' ', '').lower()
        if pub in FirstParties:
            FirstParty.append(FirstParties[pub])
        else:
            FirstParty.append('No')

    #Crating the column that defines first parties companies
    df.insert(loc=4, column='FirstParty', value=FirstParty)
    
    return df

#--------------------------------------------------------------------------------------------------------
#Function to get information from Quora
#--------------------------------------------------------------------------------------------------------

def GettingInfo2(df):
    
    #................................................................
    #Inputs
    #df - Dataframe to store the web scraping information
    #................................................................
    
    #Code:    
    
    #Creating an empty dictionary to store the console generations years
    ConsoleGens = {}

    #URL with information that defines the console generations years
    url = 'https://www.quora.com/How-are-video-game-console-generations-defined-Who-defines-them'

    #Web scraping to fill the ConsoleGens dictionary
    with urllib.request.urlopen(url) as response:
        html = response.read()
    soup = BeautifulSoup(html, 'lxml')
    Gen = 1
    for item in soup.find('ul').findAll('li'):
        for gen in item.findAll('b'):
            ConsoleGens[Gen] = gen.contents[0][5:-1].replace('Present):', '2019')
            Gen+=1

    #Creating a list to store the new column values
    ConsoleGen = []
    for Gen in df.Year:
        if type(Gen) != float:
            for key in list(ConsoleGens.keys()):
                if key != 8:
                    low = ConsoleGens[key].split('–')[0]
                    high = ConsoleGens[key].split('–')[1]
                else:
                    low = ConsoleGens[key].split('-')[0]
                    high = ConsoleGens[key].split('-')[1]            

                if Gen > low and Gen <= high:
                    llave = key
                    break

        ConsoleGen.append(llave)

    #Creating a new column containing the console generation corresponding to the given platforms
    df.insert(loc=3, column='Generation', value=ConsoleGen)
    
    return df

#--------------------------------------------------------------------------------------------------------
#Function to plot the distribution for nan values in the dataframe columns
#--------------------------------------------------------------------------------------------------------

def NaNValuesDistribution(df, null_columns):

    #................................................................
    #Inputs
    #df - Dataframe
    #null_columns - columns containing nan values
    #................................................................
    
    fig, ax = plt.subplots(2, 1, figsize=(9,5))
    ax1 = plt.subplot(2,1,1) 
    sns.barplot(df.columns,df.isnull().sum().values);
    plt.xticks(rotation=90);
    plt.ylabel('NaN values');
    plt.grid()
    ax2 = plt.subplot(2,1,2)
    ax2 = df[null_columns].isnull().sum().hist(bins=50);
    plt.ylabel('Columns')
    plt.xlabel('Amount of NaN')

    plt.subplots_adjust(top=1, bottom=0.08, left=0.10, right=0.95, hspace=0.7, wspace=0.35);
    
    
#--------------------------------------------------------------------------------------------------------
#Functio to create dummy variables from the Metacritic dataframe created
#--------------------------------------------------------------------------------------------------------    
    
def CreatingScoresGenres(df):
    
    #................................................................
    #Inputs
    #df - Dataframe created with kaggle data
    #
    #Outputs
    #dfMetas - Dataframe composed by the video genres and their scores for several video games
    #................................................................    
    
    #Creating a list with all posible game genres fine in the used dataframe
    Genres = []
    for item in df.Genre:
        itemSplitted = item.split(';')
        for genre in itemSplitted:
            Genres.append(genre)      


    #Converting users average scores to numbers
    df.Avg_Userscore = pd.to_numeric(df.Avg_Userscore)

    #Creating the dataframes
    dfMetas = df.sort_values('Metascore', ascending=False)[['Title', 'Genre', 'Metascore']].copy()
    dfMetas.reset_index( drop=True, inplace=True)
    Genres = list(set(Genres))
    #Creating columns for each of the different found video games genres 
    for genre in set(Genres):
        dfMetas[genre] = 0

    #Split the provided video games genres on each of their corresponding new generated genres columns
    for n in range(df.shape[0]):
        itemMetSplitted = dfMetas.Genre.loc[n].split(';')
        for genre in itemMetSplitted:
            dfMetas[genre].loc[n]=int(dfMetas.Metascore.loc[n])

    #Removing unused columns
    dfMetas.drop(['Metascore', 'Genre'], axis=1, inplace=True)

    #df.to_csv('Test.csv')        
    #Drop the provided video games genres column
    df.drop('Genre', axis=1, inplace=True)

    return dfMetas

#--------------------------------------------------------------------------------------------------------
#Function plot variance for obtained PCA components
#--------------------------------------------------------------------------------------------------------    

def PCAvariance(VarRat, VarRatAcum):
    
    #................................................................
    #Inputs
    #VarRat - Variance ratio for the PCA components
    #................................................................    
    
    fig, ax1 = plt.subplots(figsize=(15,5))

    plt.rcParams.update({'font.size': 14})

    ax1.bar(range(len(VarRat)),VarRat, color='c')
    ax1.set_xlabel('PCA components')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Variance ratio', color='k')
    ax1.tick_params('y', colors='k')
    plt.grid(axis='x')

    ax2 = ax1.twinx()
    ax2.plot(VarRatAcum, 'r-*')
    ax2.set_ylabel('Commulative Variance ratio', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.grid(axis='y')
    plt.show()
    
#--------------------------------------------------------------------------------------------------------
#Function to plot the weights for the PCA components
#--------------------------------------------------------------------------------------------------------      
    
def PCAcompWeights(pca, df, Columns,component, plot=False):
    
    #................................................................
    #Inputs
    #pca - Fitter pca model
    #df - dataframe used to fit the pca model
    #columns - columns for the df dataframe
    #components - component of interest to plot its weights
    #................................................................  
    
    #Creating a dataframe with the pca components
    componentsA = pd.DataFrame(np.round(pca.components_, 4), columns = Columns)

    #Generating a copy of the created dataframe
    components = componentsA.copy()
    components.head()
    #Sorting the weigths for the required pca component
    components.sort_values(by=component-1, ascending=False, axis=1, inplace=True)

    #Printing the sorted pca weights for the required component
    if plot==False:
        print(components.loc[component-1].transpose())
    
    #Plotting the sorted pca weights for the required component 
    if plot==True:
        plt.rcParams.update({'font.size': 10})
        fig, ax = plt.subplots(figsize = (19,5))
        components[components.columns[0:90]].loc[component-1].plot(ax = ax, kind = 'bar');
        plt.grid()

#--------------------------------------------------------------------------------------------------------
#Function to finds the amounts of clusters, using kmeans, that can represent the data
#--------------------------------------------------------------------------------------------------------      
        
def PCATest(df, n_scor, rango, layout=(1,1), Figsize=(15,13)):

    #................................................................
    #Inputs
    #df - dataframe
    #n_scor - Number of clusters to score, in case of subplot, needs ot be an arrrays
    #rango - Number of components to use for the evaluation
    #layout - Subplots layout
    #Figsize - Size of the figure to display the results
    #................................................................  
    
    fig, ax = plt.subplots(layout[0], layout[1], figsize=Figsize)
    
    for i, m in enumerate(n_scor):

        df = df[0:rango, :]

        interia = []

        for center in range(1,m+1):
    
            kmeans = KMeans(center, random_state=0)
    
            model = kmeans.fit(df)

            interia.append(model.inertia_)
    
        centers = list(range(1,m+1))

        plt.subplot(layout[0], layout[1], i+1) 
        plt.plot(centers,interia)
        plt.title('Kmeans (PCA components)')
        plt.xlabel('Centers');
        plt.ylabel('Average distance');
        plt.grid()
        i+=1
        
    plt.subplots_adjust(top=1, bottom=0.08, left=0.10, right=0.95, hspace=0.3, wspace=0.35);


#--------------------------------------------------------------------------------------------------------
#Funtion to plot the ratio of the total population that each found cluester represents
#--------------------------------------------------------------------------------------------------------      

def ClustersProportions(PredDf, Comp):

    #................................................................
    #Inputs
    #prefDf - Predicted clusters
    #comp - Number of components to display
    #................................................................      
    
    # Compare the proportion of data in each cluster for the customer data to the
    # proportion of data in each cluster for the general population.
    CountsMeta = []
    components = Comp

    for n in range(components):
        CountsMeta.append(sum(PredDf == n)/len(PredDf))

    d= {'Cluster':range(1,components+1), 'Clusters': CountsMeta}
    General = pd.DataFrame(d)

    General.plot(x='Cluster', y = ['Clusters'], kind='bar', figsize=(19,8), color=['steelblue', 'darkorange'], fontsize=12)
    plt.legend(prop={'size': 15});
    plt.xticks(rotation=0);
    plt.xlabel('Clusters', fontsize=15);
    plt.ylabel('Scores proportion', fontsize=15);
    plt.title('Metacritic scores', fontsize=19)
    plt.grid()
    
#--------------------------------------------------------------------------------------------------------
#Funtion to plot the ratio of the total population that each found cluester represents, for each of the
#video games consoles generations
#--------------------------------------------------------------------------------------------------------        
    
def PlottingGamesClusters(PredMetas5th, PredMetas6th, PredMetas7th, PredMetas8th, Comp):
    
    #................................................................
    #Inputs
    #PredMetas5th- Predicted clusters for the 5th gen
    #PredMetas6th - Predicted clusters for the 6th gen
    #PredMetas7th - Predicted clusters for the 7th gen
    #PredMetas8th - Predicted clusters for the 8th gen
    #comp - Number of components to display
    #................................................................    
    
    Colors = ['tomato', 'dodgerblue', 'mediumseagreen', 'darkorange']
    Years = ['1993–1998', '1999–2005', '2006–2012', '2013-Present']

    fig, axes = plt.subplots(2,2,figsize=(19,10))

    for m in range(4):

        ax = plt.subplot(2,2,m+1)

        CountsMeta = []
        components = Comp[m]

        for n in range(components):
            if m==0:
                CountsMeta.append(sum(PredMetas5th == n)/len(PredMetas5th))
            if m==1:
                CountsMeta.append(sum(PredMetas6th == n)/len(PredMetas6th))
            if m==2:
                CountsMeta.append(sum(PredMetas7th == n)/len(PredMetas7th))
            if m==3:
                CountsMeta.append(sum(PredMetas8th == n)/len(PredMetas8th))

        if m==0:
            d= {'Cluster':range(1,components+1), "PS1/N64/Saturn": CountsMeta}
            General = pd.DataFrame(d)
            General.plot(x='Cluster', y = ["PS1/N64/Saturn"], kind='bar', color=Colors[m], ax=ax, fontsize=12)
        if m==1:
            d= {'Cluster':range(1,components+1), "Dreamcast/PS2/Gamecube/Xbox": CountsMeta}
            General = pd.DataFrame(d)
            General.plot(x='Cluster', y = ["Dreamcast/PS2/Gamecube/Xbox"], kind='bar', color=Colors[m], ax=ax, fontsize=12)
        if m==2:
            d= {'Cluster':range(1,components+1), "PS3/360/Wii": CountsMeta}
            General = pd.DataFrame(d)
            General.plot(x='Cluster', y = ["PS3/360/Wii"], kind='bar', color=Colors[m], ax=ax, fontsize=12)
        if m==3:
            d= {'Cluster':range(1,components+1), "PS4/XB1/WiiU": CountsMeta}
            General = pd.DataFrame(d)
            General.plot(x='Cluster', y = ["PS4/XB1/WiiU"], kind='bar', color=Colors[m], ax=ax, fontsize=12)

        plt.xticks(rotation=90);
        plt.ylabel('Scores proportion', fontsize=15);
        plt.title('Metacritic Clusters ('+ Years[m] +')', fontsize=20)
        plt.xlabel('Clusters', fontsize=15)
        plt.legend(prop={'size': 15})
        plt.grid()

        plt.subplots_adjust(top=1, bottom=0.08, left=0.10, right=0.95, hspace=0.3, wspace=0.35);
        
#--------------------------------------------------------------------------------------------------------
#Funtion to plot the statistics of a dataframe, in the form of boxplots
#--------------------------------------------------------------------------------------------------------             
        
def DevStatistics(df, my_pal):
    
    #................................................................
    #Inputs
    #df - dataframe
    #my_pal - Color pallete 
    #................................................................       
    
    sns.set(style="whitegrid")
    sns.set(font_scale=1.2)
    fig, axes = plt.subplots(2,2,figsize=(15,10))
    ax = plt.subplot(2,2,1)
    sns.boxplot(x="FirstParty", y="Metascore", palette=my_pal,data=df[(df['FirstParty'] <= 3) & (df['FirstParty'] > 0)][df['Generation']==5],  ax=ax, showmeans=True);
    ax.set(xticklabels=['Nintendo', 'Sony', 'Microsoft']);
    plt.title('PS1/N64/Saturn', fontsize=17)
    ax.set_ylim(75,100);
    plt.xlabel('');
    ax = plt.subplot(2,2,2)
    sns.boxplot(x="FirstParty", y="Metascore", palette=my_pal, data=df[(df['FirstParty'] <= 3) & (df['FirstParty'] > 0)][df['Generation']==6], ax=ax, showmeans=True);
    ax.set(xticklabels=['Nintendo', 'Sony', 'Microsoft']);
    plt.title('Dreamcast/PS2/Gamecube/Xbox', fontsize=17)
    ax.set_ylim(75,100);
    plt.xlabel('');
    ax = plt.subplot(2,2,3)
    sns.boxplot(x="FirstParty", y="Metascore", palette=my_pal, data=df[(df['FirstParty'] <= 3) & (df['FirstParty'] > 0)][df['Generation']==7], ax=ax, showmeans=True);
    ax.set(xticklabels=['Nintendo', 'Sony', 'Microsoft']);
    plt.title('PS3/360/Wii', fontsize=17)
    ax.set_ylim(75,100);
    plt.xlabel('');
    ax = plt.subplot(2,2,4)
    sns.boxplot(x="FirstParty", y="Metascore", palette=my_pal, data=df[(df['FirstParty'] <= 3) & (df['FirstParty'] > 0)][df['Generation']==8], ax=ax, showmeans=True);
    ax.set(xticklabels=['Nintendo', 'Sony', 'Microsoft']);
    plt.title('PS4/XB1/WiiU', fontsize=17)
    ax.set_ylim(75,100);
    plt.xlabel('');

#--------------------------------------------------------------------------------------------------------
#Funtion to display developed games for each publisher for specific games consoles generations
#--------------------------------------------------------------------------------------------------------  

def GamesDeveloped(df):
    
    #................................................................
    #Inputs
    #df - dataframe
    #................................................................       
    
    FirstParties = ['Nintendo', 'Sony', 'Microsoft']
    Developers = [1,2,3]
    Generation = []
    FirstParty = []
    Games = []

    for n in range(5,9):
        for m in range(1,4):
            A = df[df['Generation']==n][df['FirstParty']==m]
            Generation.append(n)
            FirstParty.append(FirstParties[m-1])
            Games.append(A.Title.count())

    GamesPerDev = {'FirstParty':FirstParty, 'Generation':Generation, 'RatedGames':Games}
    GamesPerDev = pd.DataFrame(GamesPerDev)
    
    return GamesPerDev

'''******************************************************************************************************
Fin
******************************************************************************************************'''