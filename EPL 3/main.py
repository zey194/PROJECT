import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
from datetime import datetime as dt
import itertools
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, FloatField
from flask import Flask, render_template, redirect, url_for, request, session
from flask_wtf.csrf import CSRFProtect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import random
import requests



# * Global variables
classifier = None
dataset = None
dataset2 = None
matches = None
chosen_data_matches = []
display_data = None
players_by_team = {}
X_train1 = []
rand_index = 0


def preprocess_features(X):
    ''' Preprocesses the football data and converts catagorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    # for col, col_data in X.iteritems():
    for col, col_data in X.items():

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)
                    
        # Collect the revised columns
        output = output.join(col_data)
    
    return output


def preprocess_features2(x_df):

    res = pd.DataFrame(columns=['Unnamed: 0', 'FTHG', 'FTAG','HTP', 'ATP', 'HM1_D', 'HM1_L', 'HM1_M', 'HM1_W', 'HM2_D', 'HM2_L', 'HM2_M', 'HM2_W', 'HM3_D', 'HM3_L', 'HM3_M', 'HM3_W', 'AM1_D', 'AM1_L', 'AM1_M', 'AM1_W', 'AM2_D', 'AM2_L', 'AM2_M', 'AM2_W', 'AM3_D', 'AM3_L', 'AM3_M', 'AM3_W', 'HTGD', 'ATGD', 'DiffFormPts'], index=[0])
    res['FTHG'] = x_df['FTHG']
    res['FTAG'] = x_df['FTAG']
    res['HTP'] = x_df['HTP']
    res['ATP'] = x_df['ATP']
    res['Unnamed: 0'] = x_df['Unnamed: 0']

    for prefix in ['HM1', 'HM2', 'HM3', 'AM1', 'AM2', 'AM3']:
        for suffix in ['W', 'L', 'M', 'D']:
            column_name = f"{prefix}_{suffix}"
            if column_name in x_df.columns:
                res[column_name] = True
            else:
                res[column_name] = False
    res['FTHG'] = x_df['FTHG']
    res['FTAG'] = x_df['FTAG']
    res['HTGD'] = x_df['HTGD']
    res['ATGD'] = x_df['ATGD']
    res['DiffFormPts'] = x_df['DiffFormPts']

    return res


def make_model(X_train1=None, pred=None):
    global classifier

    dataset = pd.read_csv('final_dataset.csv')

    dataset2 = dataset.copy().drop(columns =['Date', 'HomeTeam', 'AwayTeam', 
        'HTGS', 'ATGS', 'HTGC', 'ATGC',
        'HM4', 'HM5','AM4', 'AM5', 'MW', 'HTFormPtsStr',
        'ATFormPtsStr', 'HTFormPts', 'ATFormPts', 'HTWinStreak3',
        'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5', 'ATWinStreak3',
        'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5',
        'DiffPts'] )

    X_all = dataset2.drop(['FTR'],axis=1)
    y_all = dataset2['FTR']


    # Appending new dataset
    if X_train1 != None:

        index1 = len(dataset2)
        X_train1 = [[index1] + X_train1]
        columns = ['Unnamed: 0', 'FTHG', 'FTAG','HTP', 'ATP', 'HM1', 'HM2', 'HM3', 'AM1', 'AM2', 'AM3', 'HTGD', 'ATGD', 'DiffFormPts']
        x_df = pd.DataFrame(X_train1, columns=columns)
        X_all = X_all = pd.concat([X_all, x_df], ignore_index=True)

        pred = 'H'
        pred_series = pd.Series(pred)
        y_all = y_all = pd.concat([y_all, pred_series], ignore_index=True)



    cols = [['FTHG', 'FTAG','HTGD','ATGD','HTP','ATP']]
    for col in cols:
        X_all[col] = scale(X_all[col])

    #last 3 wins for both sides
    X_all.HM1 = X_all.HM1.astype('str')
    X_all.HM2 = X_all.HM2.astype('str')
    X_all.HM3 = X_all.HM3.astype('str')
    X_all.AM1 = X_all.AM1.astype('str')
    X_all.AM2 = X_all.AM2.astype('str')
    X_all.AM3 = X_all.AM3.astype('str')


    X_all = preprocess_features(X_all)


    # Shuffle and split the dataset into training and testing set.
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, 
                                                        test_size = 0.2,
                                                        random_state = 42,
                                                        stratify = y_all)

    # fitting GausisanNB
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)


def make_pred(X_train1):

    global classifier
    
    index = 0

    X_train1 = [ [index] + X_train1]

    columns = ['Unnamed: 0','FTHG', 'FTAG', 'HTP', 'ATP', 'HM1', 'HM2', 'HM3', 'AM1', 'AM2', 'AM3', 'HTGD', 'ATGD', 'DiffFormPts']

    x_df = pd.DataFrame(X_train1, columns=columns)

    x_df = preprocess_features(x_df)
    
    ans = preprocess_features2(x_df)

    pred = classifier.predict(ans)

    return pred[0]


def work():
    global dataset, dataset2, matches, players_by_team, display_data, history_api_data
    dataset = pd.read_csv('final_dataset.csv')
    dataset2 = dataset.copy().drop(columns =['FTR', 'Date', 'HomeTeam', 'AwayTeam', 
       'HTGS', 'ATGS', 'HTGC', 'ATGC',
       'HM4', 'HM5','AM4', 'AM5', 'MW', 'HTFormPtsStr',
       'ATFormPtsStr', 'HTFormPts', 'ATFormPts', 'HTWinStreak3',
       'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5', 'ATWinStreak3',
       'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5',
       'DiffPts'] )
    display_data = dataset[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG',
       'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTP', 'ATP',
       'HTFormPts', 'ATFormPts', 'HM1', 'AM1', 'HM2', 'AM2', 'HM3', 'AM3',
       'HTWinStreak3', 'ATWinStreak3', 'HTLossStreak3', 'ATLossStreak3',
       ]].copy()
    display_data = display_data.values.tolist()
    competition_id=2
    # history_api_data=get_recent_matches(competition_id,from_date,to_date)
    matches_df = dataset[['HomeTeam', 'AwayTeam']].copy()
    matches = [tuple(x) for x in matches_df.to_records(index=False)]
    df = pd.read_csv('Premier League Player Stats.csv')
    for index, row in df.iterrows():
        team = row['TEAM'].lower()
        player = row['PLAYER']
        if team in players_by_team:
            players_by_team[team].append(player)
        else:
            players_by_team[team] = [player]
    # new_names = {'wolverhampton wanderers' : 'wolves', 'brighton and hove albion' : 'brighton', 'leicester city': 'leicester', 'manchester city': 'man city', 'tottenham hotspur':'tottenham', 'manchester united':'man united', 'norwich city':'norwich', 'west ham united': 'west ham', 'newcastle united':'newcastle'}
    # players_by_team = {new_names.get(old_key, old_key): value for old_key, value in players_by_team.items()}
    players_by_team['arsenal']= ["Aaron Ramsdale","David Raya","Karl Hein","William Saliba","Ben White","Gabriel Magalhaes","Cedric Soares","Takehiro Tomiyasu","Oleksandr Zinchenko","Jurrien Timber","Jakub Kiwior","Taylor Foran","Reuell Walters","Ayden Heaven","Thomas Partey","Martin Ødegaard","Emile Smith Rowe","Jorginho","Fabio Vieira","Mohamed Elneny","Declan Rice","Miguel Azeez","Mauro Bandeira","Amario Cozier-Duberry","Myles Lewis-Skelly","Ethan Nwaneri","Eddie Nketiah","Arthur Okonkwo","Kieran Tierney","Nuno Tavares","Albert Sambi Lokonga","Marquinhos"]
    players_by_team['aston villa']=["Emiliano Martinez", "Matty Cash", "Diego Carlos", "Ezri Konsa", "Tyrone Mings", "Lucas Digne", "Pau Torres", "Álex Moreno", "Douglas Luiz", "John McGinn", "Youri Tielemans", "Emiliano Buendía", "Moussa Diaby", "Nicolò Zaniolo", "Morgan Rogers", "Leon Bailey", "Jacob Ramsey", "Boubacar Kamara", "Tim Iroegbunam", "Ollie Watkins", "Jhon Durán"]
    players_by_team['afc bournemouth']=["Darren Randolph", "Neto", "Andrei Radu", "Ryan Fredericks", "Milos Kerkez", "Lloyd Kelly", "Chris Mepham", "Marcos Senesi", "Illia Zabarnyi", "Lewis Cook", "Joe Rothwell", "Gavin Kilkenny", "Hamed Traorè", "Marcus Tavernier", "Philip Billing", "Dominic Solanke", "Dango Ouattara", "Antoine Semenyo", "Justin Kluivert", "Kieffer Moore", "Enes Ünal"]
    players_by_team['brentford']=["Mark Flekken", "Thomas Strakosha", "Hákon Valdimarsson", "Ellery Balcombe", "Vincent Angelini", "Matthew Cox", "Aaron Hickey", "Rico Henry", "Charlie Goode", "Ethan Pinnock", "Kristoffer Ajer", "Christian Norgaard", "Josh Dasilva", "Mathias Jensen", "Sergi Canos", "Vitaly Janelt (on loan to FC Bayern Munich II)", "Frank Onyeka", "Tariqe Fosu", "Mads Bech Sørensen (on loan to Odense Boldklub)", "Yoane Wissa", "Kevin Schade", "Ivan Toney (on loan to Crewe Alexandra)", "Bryan Mbeumo"]
    players_by_team['brighton & hove albion']=["Bart Verbruggen", "Jason Steele", "Tom McGill", "Tariq Lamptey", "Igor Julio", "Adam Webster", "Lewis Dunk", "Valentín Barco", "Jan Paul van Hecke", "Pervis Estupiñán", "Joël Veltman", "James Milner", "Solly March", "Billy Gilmour", "Pascal Groß", "Adam Lallana", "Jakub Moder", "Carlos Baleba", "Kaoru Mitoma", "Facundo Buonanotte", "Jack Hinshelwood", "Cameron Peupion", "João Pedro", "Julio Enciso", "Danny Welbeck", "Simon Adingra", "Evan Ferguson", "Ansu Fati"]
    players_by_team['burnley']=["Arijanet Muric", "Bailey Peacock-Gizbett", "Nick Pope", "Matthew Lowton", "James Tarkowski", "Kevin Long", "Erik Pieters", "Taylor Harwood-Bellis", "Charlie Taylor", "Nathan Collins", "Josh Brownhill", "Ashley Westwood", "Josh Cullen", "Dwight McNeil", "Manuel Benson", "Robert Iwanyi", "Johann Berg Gudmundsson", "Jay Rodriguez", "Wout Weghorst", "Matej Vydra", "Ashley Barnes"]
    players_by_team['chelsea']=["Robert Sanchez", "Marcus Bettinelli", "Djordje Petrovic", "Axel Disasi", "Marc Cucurella", "Benoit Badiashile", "Thiago Silva", "Trevoh Chalobah", "Ben Chilwell", "Reece James", "Levi Colwill", "Malo Gusto", "Enzo Fernandez", "Conor Gallagher", "Moises Caicedo", "Cesare Casadei", "Romeo Lavia", "Raheem Sterling", "Mykhailo Mudryk", "Noni Madueke", "Nicolas Jackson", "Christopher Nkunku", "Cole Palmer"]
    players_by_team['crystal palace']=["Dean Henderson", "Joseph Whitworth", "Remi Matthews", "Sam Johnstone", "Chris Richards", "Daniel Munoz", "James Tomkins", "Joachim Andersen", "Joel Ward", "Marc Guehi", "Nathaniel Clyne", "Rob Holding", "Tyrick Mitchell", "Adam Wharton", "Cheick Oumar Doucoure", "David Ozoh", "Eberechi Eze", "Jadan Raymond", "Jairo Riedewald", "Jefferson Lerma", "Franco Umeh-Chibueze", "Jean-Philippe Mateta", "Jordan Ayew", "Matheus Franca", "Odsonne Edouard", "Roshaun Mathurin"]
    players_by_team['everton']=["Jordan Pickford", "Andy Lonergan", "João Virgínia", "Seamus Coleman", "Nathan Patterson", "Ben Godfrey", "Michael Keane", "Vitaliy Mykolenko", "Ashley Young", "Jarrad Branthwaite", "Abdoulaye Doucouré", "Idrissa Gueye", "Amadou Onana", "André Gomes", "James Garner", "Lewis Warrington", "Tyler Onyango", "Mackenzie James Hunt", "Dominic Calvert-Lewin", "Dwight McNeil", "Arnaut Danjuma", "Jack Harrison", "Beto", "Youssef Chermiti", "Lewis Dobbin"]
    players_by_team['fulham']=["Marek Rodak", "Steven Benda", "Bernd Leno", "Alex Borto", "Kenny Tete", "Calvin Bassey", "Tosin Adarabioyo", "Fode Ballo-Toure", "Tim Ream", "Timothy Castagne", "Issa Diop", "Harvey Araujo", "Antonee Robinson", "Harrison Reed", "Harry Wilson", "Tom Cairney", "Bobby De Cordova-Reid", "Andreas Pereira", "João Palhinha", "Willian", "Ollie O'Neill", "Raúl Jiménez", "Armando Broja", "Adama Traoré", "Rodrigo Muniz", "Alex Iwobi"]
    players_by_team['liverpool']=["Alisson Becker", "Adrian", "Caoimhin Kelleher", "Trent Alexander-Arnold", "Virgil van Dijk", "Ibrahima Konate", "Kostas Tsimikas", "Andrew Robertson", "Joe Gomez", "Joel Matip", "Conor Bradley", "Fabinho", "Thiago Alcantara", "Curtis Jones", "Harvey Elliott", "Naby Keita", "Mohamed Salah", "Cody Gakpo", "Luis Diaz", "Diogo Jota", "Darwin Nunez"]
    players_by_team['luton town']=["James Shea", "Tim Krul", "Thomas Kaminski", "Dan Potts", "Tom Lockyer", "Mads Andersen", "Teden Mengi", "Reece Burke", "Daiki Hashioka", "Amari'i Bell", "Gabriel Osho", "Issa Kabore", "Luke Berry", "Marvelous Nakamba", "Pelly Ruddock Mpanzu", "Albert Sambi Lokonga", "Fred Onyedinma", "Zack Nelson", "Alfie Doughty", "Jake Burger", "Dominic Martins", "Josh Allen", "Chiedozie Ogbene", "Jacob Brown", "Carlton Morris", "Cauley Woodrow"]
    players_by_team['manchester city']=["Ederson", "Stefan Ortega Moreno", "Scott Carson", "True Grant", "Kyle Walker", "Ruben Dias", "John Stones", "Nathan Ake", "Aymeric Laporte", "Sergio Gomez", "Josko Gvardiol", "Manuel Akanji", "Rico Lewis", "Max Alleyne", "Luke Mbete", "Rodri", "Kevin De Bruyne", "Bernardo Silva", "Phil Foden", "Jack Grealish", "Mateo Kovacic", "Jeremy Doku", "Matheus Nunes", "Oscar Bobb", "Micah Hamilton", "Jacob Wright", "James Mcatee", "Kalvin Phillips", "Claudio Gomes", "Mahamadou Susoho", "Erling Haaland", "Julian Alvarez", "Ben Knight"]
    players_by_team['manchester united']= ["Altay Bayindir", "Tom Heaton", "Andre Onana", "Victor Lindelof", "Harry Maguire", "Lisandro Martinez", "Tyrell Malacia", "Raphael Varane", "Diogo Dalot", "Luke Shaw", "Aaron Wan-Bissaka", "Bruno Fernandes", "Casemiro", "Christian Eriksen", "Hannibal Mejbri", "Kobbie Mainoo", "Scott McTominay", "Sofyan Amrabat", "Anthony Martial", "Marcus Rashford", "Rasmus Hojlund", "Alejandro Garnacho", "Antony", "Jadon Sancho"]
    players_by_team['newcastle united']=["Martin Dubravka", "Nick Pope", "Loris Karius", "Mark Gillespie", "Kieran Trippier", "Sven Botman", "Fabian Schär", "Jamaal Lascelles", "Matt Targett", "Dan Burn", "Paul Dummett", "Emil Krafth", "Lewis Hall", "Valentino Livramento", "Alex Murphy", "Bruno Guimarães", "Joe Willock", "Miguel Almirón", "Elliot Anderson", "Jacob Murphy", "Matt Ritchie", "Sean Longstaff", "Lucas De Bolle", "Alexander Isak", "Callum Wilson", "Joelinton", "Anthony Gordon"]
    players_by_team['nottingham forest']=["Matt Turner", "Wayne Hennessey", "Matz Sels", "Will Brook", "Aaron Bott", "Odysseas Vlachodimos", "Nuno Tavares", "Neco Williams", "Harry Toffolo", "Felipe", "Moussa Niakhaté", "Gonzalo Montiel", "Willy-Arnaud Boly", "Andrew Omobamidele", "Murillo", "Ola Aina", "Zach Abbott", "Ibrahim Sangare", "Cheikhou Kouyate", "Morgan Gibbs-White", "Callum Hudson-Odoi", "Nicolas Dominguez", "Giovanni Reyna", "Anthony Elanga", "Ryan Yates", "Divock Origi", "Danilo dos Santos de Oliveira", "Taiwo Awoniyi"]
    players_by_team['sheffield united']=["Paul Foderingham", "Mason Holgate", "Jake Eastwood", "George Baldock", "Chris Basham", "John Egan", "Anel Ahmedhodzic", "Jack Robinson", "Rhys Norrington-Davies", "Jayden Bogle", "Enda Stevens", "Oliver Norwood", "Oliver Arblaster", "Ben Osborn", "Sander Berge", "James McAtee", "Tom Davies", "John Fleck", "Iliman Ndiaye", "Rhian Brewster", "Oliver McBurnie", "William Osula", "Daniel Jebbison", "Antwoine Hackford"]
    players_by_team['tottenham hotspur']=["Guglielmo Vicario", "Fraser Forster", "Brandon Austin", "Alfie Whiteman", "Luca Gunter", "Marqes Muir", "Brooklyn Lyons-Foster", "Charlie Sayers", "Radu Dragusin", "Emerson Royal", "Cristian Romero", "Ryan Sessegnon", "Kallum Cesay", "Pedro Porro", "Ben Davies", "Micky van der Ven", "Destiny Udogie", "Tyrell Ashcroft", "Alfie Dorrington", "Oliver Skipp", "Pierre-Emile Højbjerg", "Yves Bissouma", "James Maddison", "Bryan Gil", "Timo Werner", "Giovani Lo Celso", "Dejan Kulusevski"]
    players_by_team['west ham united']=["Lukasz Fabianski", "Alphonse Areola", "Aaron Cresswell", "Kurt Zouma", "Ben Johnson", "Vladimir Coufal", "Angelo Ogbonna", "Nayef Aguerd", "Emerson Palmieri", "Konstantinos Mavropanos", "Thilo Kehrer", "Tomas Soucek", "James Ward-Prowse", "Lucas Paqueta", "Flynn Downes", "Kalvin Phillips", "Mohammed Kudus", "Maxwel Cornet", "Edson Alvarez", "Michail Antonio", "Danny Ings"]
    players_by_team['wolverhampton wanderers']=["Jose Sa", "Daniel Bentley", "Tom King", "Matt Doherty", "Rayan Ait-Nouri", "Santiago Bueno", "Craig Dawson", "Hugo Bueno", "Nelson Semedo", "Maximilian Kilman", "Toti Gomes", "Joao Gomes", "Mario Lemina", "Boubacar Traore", "Jean-Ricner Bellegarde", "Tommy Doyle", "Pablo Sarabia", "Fletcher Holman", "Harvey Griffiths", "Pedro Neto", "Hwang Hee-Chan", "Matheus Cunha", "Noha Lemina", "Enso Gonzalez", "Nathan Fraser"]



def get_player_data(match_id):
    url = "https://livescore-api.com/api-client/matches/lineups.json"
    params = {
        "match_id": match_id,
        "key": "6mod0qgqPE0saEZh",
        "secret": "HlXmn6jwSqdPRxOXUcaVCrls6mVg1UxI"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        player_data = response.json()  # Convert response to JSON format
        # Process player data here
        print(player_data)
        return player_data
    else:
        print("Failed to fetch player data:", response.status_code)
        return None

# match_id = 194137
# player_data = get_player_data(match_id)
competition_id = 2
def get_recent_matches(competition_id,from_date,to_date):
    if from_date is None:
        from_date = "2021-01-01" 
    url = "https://livescore-api.com/api-client/scores/history.json"
    params = {
        "competition_id": competition_id,
        "from":from_date,
        "to":to_date,
        "key": "6mod0qgqPE0saEZh",
        "secret": "wg0uhM9gAfbiaxUhK6DXMQCYAFOxVExs"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        recent_matches = response.json()  # Convert response to JSON format
        # Process recent matches data here
        return recent_matches 
    else:
        print("Failed to fetch recent matches:", response.status_code)
        return None

def get_scores(id):
    url = "https://livescore-api.com/api-client/scores/events.json"
    params = {
        "id": id,
        "key": "6mod0qgqPE0saEZh",
        "secret": "wg0uhM9gAfbiaxUhK6DXMQCYAFOxVExs"
    }
    print("ABCDAASNLKSANL",id)
    response = requests.get(url, params=params)
    if response.status_code == 200:
        recent_matches = response.json()  # Convert response to JSON format
        # Process recent matches data here
        return recent_matches 
    else:
        print("Failed to fetch recent matches:", response.status_code)
        return None

def get_statistics(id):
    url = "https://livescore-api.com/api-client/matches/stats.json"
    params = {
        "match_id": id,
        "key": "6mod0qgqPE0saEZh",
        "secret": "wg0uhM9gAfbiaxUhK6DXMQCYAFOxVExs"
    }
    # print("Staticses..........",id)
    response = requests.get(url, params=params)
    if response.status_code == 200:
        recent_matches = response.json()  # Convert response to JSON format
        # Process recent matches data here
        print("Staticses..........",recent_matches)
        return recent_matches 
    else:
        print("Failed to fetch recent matches:", response.status_code)
        return None


def get_live_scores(competition_id):
    url = "https://livescore-api.com/api-client/matches/live.json"
    params = {
        "competition_id": competition_id,
        "key": "6mod0qgqPE0saEZh",
        "secret": "HlXmn6jwSqdPRxOXUcaVCrls6mVg1UxI"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        live_scores = response.json()  # Convert response to JSON format
        # Process live scores data here
        return live_scores
    else:
        print("Failed to fetch live scores:", response.status_code)
        return None

def get_fixtures(competition_id):
    url = "https://livescore-api.com/api-client/fixtures/matches.json"
    params = {
        "competition_id": competition_id,
        "key": "6mod0qgqPE0saEZh",
        "secret": "HlXmn6jwSqdPRxOXUcaVCrls6mVg1UxI"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        live_scores = response.json()  # Convert response to JSON format
        # Process live scores data here
        print('upcoming matches are.....',live_scores)
        return live_scores
    else:
        print("Failed to fetch live scores:", response.status_code)
        return None

def zip_lists(list1, list2):
    return zip(list1, list2)

# ! Flask App
app = Flask(__name__)
csrf = CSRFProtect(app)
app = Flask(__name__)
app.secret_key = 'your_secret_key' 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
app.jinja_env.filters['zip_lists'] = zip_lists

matcheslive= [
            {
                "country": {
                   
                    "id": 71,
                    "uefa_code": "",
                    "name": "Australia",
                    "fifa_code": "AUS",
                    "flag": "AUS.png"
                },
                "last_changed": "2023-12-18 08:42:53",
                "home": {
                    "stadium": "Campbelltown Stadium",
                    "id": 6581,
                    "logo": "https://cdn.live-score-api.com/teams/b9d463b08c43de617618f7ec65a54cc3.png",
                    "name": "Macarthur",
                    "country_id": "71"
                },
                "added": "2023-12-18 07:45:06",
                "status": "IN PLAY",
                
                "location": "Campbelltown Stadium",
                "scheduled": "08:00",
                "competition": {
                    "tier": 1,
                    
                    "id": 67,
                    
                    "name": "Hyundai A-League",

                },
                "id": 483252,
                "time": "43",
                "fixture_id": 1657338,
                "odds": {
                    "live": {
                        "1": 5.6,
                        "2": 1.61,
                        "X": 3.55
                    },
                    "pre": {
                        "1": 2,
                        "2": 3.45,
                        "X": 3.85
                    }
                },
                "away": {
                    "stadium": "Westpac Stadium",
                    "id": 202,
                    "logo": "https://cdn.live-score-api.com/teams/004d52954bd4936cc14e30b4511073ab.png",
                    "name": "Wellington Phoenix",
                    "country_id": "71"
                },
                
                "scores": {
                    "score": "0 - 1",
                    "ht_score": "",
                    "ft_score": "",
                    "et_score": "",
                    "ps_score": ""
                },
                "urls": {
                    "events": "https:\/\/livescore-api.com\/api-client\/scores\/events.json?id=483252",
                    "statistics": "https:\/\/livescore-api.com\/api-client\/matches\/stats.json?match_id=483252",
                    "lineups": "https:\/\/livescore-api.com\/api-client\/matches\/lineups.json?match_id=483252",
                    "head2head": "https:\/\/livescore-api.com\/api-client\/teams\/head2head.json?team1_id=6581&team2_id=202"
                }
            },
            {
                "country": {
                    
                    "id": 90,
                    "uefa_code": "",
                    "name": "Indonesia",
                    "fifa_code": "IDN",
                    "flag": "IDN.png"
                },
                "last_changed": "2023-12-18 08:42:52",
                "home": {
                    "stadium": "Stadion Brawijaya",
                    "id": 5746,
                    "logo": "https:\/\/cdn.live-score-api.com\/teams\/ee2fd3a662dc351faf3b6fa4e1b57074.png",
                    "name": "Persik Kediri",
                    "country_id": "90"
                },
                "added": "2023-12-18 07:45:07",
                "status": "IN PLAY",
               
                "location": "Stadion Brawijaya",
                "scheduled": "08:00",
                "competition": {
                    "tier": 1,
                    
                    "id": 27,
                   
                    "name": "GO-JEK Liga 1",
                    
                },
                "id": 483253,
                "time": "43",
                "fixture_id": 1624810,
                "odds": {
                    "live": {
                        "1": 2.38,
                        "2": 3.6,
                        "X": 2.43
                    },
                    "pre": {
                        "1": 1.98,
                        "2": 3.3,
                        "X": 3.15
                    }
                },
                "away": {
                    "stadium": "Stadion Andi Mattalatta",
                    "id": 204,
                    "logo": "https:\/\/cdn.live-score-api.com\/teams\/d39724be7ff1da90678b25cf3be6e6f1.png",
                    "name": "PSM Makassar",
                    "country_id": "90"
                },
                
                "scores": {
                    "score": "0 - 0",
                    "ht_score": "",
                    "ft_score": "",
                    "et_score": "",
                    "ps_score": ""
                },
                "urls": {
                    "events": "https:\/\/livescore-api.com\/api-client\/scores\/events.json?id=483253",
                    "statistics": "https:\/\/livescore-api.com\/api-client\/matches\/stats.json?match_id=483253",
                    "lineups": "https:\/\/livescore-api.com\/api-client\/matches\/lineups.json?match_id=483253",
                    "head2head": "https:\/\/livescore-api.com\/api-client\/teams\/head2head.json?team1_id=5746&team2_id=204"
                }
            }
        ]

class MyForm(FlaskForm):
    data15 = FloatField('Data 15')
    data16 = FloatField('Data 16')
    data1 = FloatField('Data 1')
    data2 = FloatField('Data 2')
    data3 = StringField('Data 3')
    data4 = StringField('Data 4')
    data5 = StringField('Data 5')
    data6 = StringField('Data 6')
    data7 = StringField('Data 7')
    data8 = StringField('Data 8')
    data9 = FloatField('Data 9')
    data10 = FloatField('Data 10')
    data11 = FloatField('Data 11')
    pred = StringField('pred')
    submit = SubmitField('Submit')


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False) 
    

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    match_index = db.Column(db.Integer, nullable=False) 
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('comments', lazy=True))
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


with app.app_context():
    db.create_all()


@app.route('/', methods=['GET', 'POST'])
def index():
    global chosen_data_matches, display_data, matches, rand_index, teams_data,match_data_live,upcoming_matches,matcheslive
    if 'username' in session:
        if request.method == 'GET':
            from_date = request.args.get('from_date')
            to_date = request.args.get('to_date')
            match_data=get_recent_matches(2,from_date,to_date)
            upcoming_matches=get_fixtures(2)
            match_data_live=get_live_scores(2)
            # matches_json = matches
            # match_data = [(index, match) for index, match in enumerate( matches[ rand_index : rand_index+5 ] )]
            chosen_data_matches = display_data[ rand_index : rand_index+5 ]
            match_index = 2  # Replace 0 with the actual index value
            return render_template('index.html', match_data=match_data, match_index=match_index,match_data_live=match_data_live, upcoming_matches=upcoming_matches, matcheslive=matcheslive )
    else:
        return redirect(url_for('login'))
@app.route('/add_comment/<int:match_index>', methods=['POST'])
def add_comment(match_index):
    if 'username' in session:
        content = request.form['content']
        user = User.query.filter_by(username=session['username']).first()
        if content:
            new_comment = Comment(content=content, user=user, match_index=match_index)
            db.session.add(new_comment)
            db.session.commit()
    return redirect(url_for('show_match_data', match_index=match_index))
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']  
        if username and password and email: 
            # Check if the username already exists
            existing_user = User.query.filter_by(username=username).first()
            if existing_user:
                return render_template('register.html', error='Username already exists. Please choose a different username.')
            else:
                # Check if the email already exists
                existing_email = User.query.filter_by(email=email).first()
                if existing_email:
                    return render_template('register.html', error='Email already exists. Please choose a different email.')
                else:
                    # Create the new user
                    new_user = User(username=username, password=password, email=email)
                    db.session.add(new_user)
                    db.session.commit()
                    session['username'] = username
                    return redirect(url_for('index'))
    return render_template('register.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username_or_email = request.form['username_or_email']  # Retrieve username or email from form
        password = request.form['password']
        # Check if the input is an email or username
        if '@' in username_or_email:
            user = User.query.filter_by(email=username_or_email).first()  # Check if email exists
        else:
            user = User.query.filter_by(username=username_or_email).first()  # Check if username exists
        
        if user and user.password == password:
            session['username'] = user.username
            return redirect(url_for('index'))
    return render_template('login.html')
@app.route('/signin', methods=['GET'])
def signin():
    if request.method == 'POST':
        username_or_email = request.form['username_or_email']  # Retrieve username or email from form
        password = request.form['password']
        # Check if the input is an email or username
        if '@' in username_or_email:
            user = User.query.filter_by(email=username_or_email).first()  # Check if email exists
        else:
            user = User.query.filter_by(username=username_or_email).first()  # Check if username exists
        
        if user and user.password == password:
            session['username'] = user.username
            return redirect(url_for('index'))
    return render_template('signin.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))
@app.route('/match/<int:match_index>', methods=['GET', 'POST'])

def show_match_data(match_index):
    global chosen_data_matches, X_train1, display_data, rand_index ,team_data,statistics
    chosen_data_matches =get_scores(match_index)
    statistics=get_statistics(match_index)
    team_data=get_player_data(match_index)
    
    if chosen_data_matches == []:
        
        print(match_index)
        print(chosen_data_matches)
        # chosen_data_matches = display_data[ rand_index : rand_index+5 ]
    dataset2_values = chosen_data_matches
    comments = Comment.query.filter_by(match_index=match_index).order_by(Comment.timestamp.desc()).all()
    form = MyForm()
    if form.validate_on_submit():

        X_train = [ (form.data15.data) ,  (form.data16.data) ,(form.data1.data) ,  (form.data2.data) ,  (form.data3.data) ,  (form.data4.data) ,  (form.data5.data) ,  (form.data6.data) ,  (form.data7.data) ,  (form.data8.data) ,  (form.data9.data) ,  (form.data10.data) ,  (form.data11.data) ]
        X_train1 = X_train
        pred = make_pred(X_train)

        return render_template('match_data2.html', pred=pred, dataset2_values=dataset2_values, form=form, comments=comments, match_index=match_index, players_by_team=players_by_team, display_data=chosen_data_matches, team_data=team_data,statistics=statistics)

    # return render_template('match_data.html', dataset2_values=dataset2_values, form=form, comments=comments, match_index=match_index, players_by_team=players_by_team, display_data=chosen_data_matches)
    return render_template('match_data.html', dataset2_values=dataset2_values, form=form, comments=comments, match_index=match_index, players_by_team=players_by_team, display_data=chosen_data_matches,team_data=team_data,statistics=statistics)
@app.route('/upcoming_match_data',methods=['GET', 'POST'])
def upcoming_match_data():
    global chosen_data_matches, X_train1, display_data, rand_index ,team_data,statistics
    form = MyForm()
    home_name = request.form.get('home_name')
    away_name = request.form.get('away_name')
    if form.validate_on_submit():
        X_train = [ (form.data15.data) ,  (form.data16.data) ,(form.data1.data) ,  (form.data2.data) ,  (form.data3.data) ,  (form.data4.data) ,  (form.data5.data) ,  (form.data6.data) ,  (form.data7.data) ,  (form.data8.data) ,  (form.data9.data) ,  (form.data10.data) ,  (form.data11.data) ]
        X_train1 = X_train
        pred = make_pred(X_train)
        return render_template('upcoming_match_data2.html', pred=pred,form=form,players_by_team=players_by_team,home_name=home_name,away_name=away_name)
    # return render_template('match_data.html', dataset2_values=dataset2_values, form=form, comments=comments, match_index=match_index, players_by_team=players_by_team, display_data=chosen_data_matches)
    return render_template('upcoming_match_data.html',form=form,players_by_team=players_by_team,home_name=home_name,away_name=away_name)

@app.route('/refresh', methods=['GET', 'POST'])
def refresh():
    global rand_index
    rand_index = random.randint( 0, (len(matches)-6) )
    return redirect(url_for('index'))
@app.route('/clear_form', methods=['POST'])
def clear_form():
    global msg, X_train1
    msg = 'New Data Fitted!'
    form = MyForm()
    if form.validate_on_submit():
        if X_train1 == []:
            X_train1 = [ (form.data15.data) ,  (form.data16.data) ,(form.data1.data) ,  (form.data2.data) ,  (form.data3.data) ,  (form.data4.data) ,  (form.data5.data) ,  (form.data6.data) ,  (form.data7.data) ,  (form.data8.data) ,  (form.data9.data) ,  (form.data10.data) ,  (form.data11.data) ]
        make_model(X_train1, form.pred.data) # Retrain the model
    print('\n', msg, '\n')
    return redirect(url_for('index'))
@app.route('/clear_form2', methods=['POST'])
def clear_form2():
    return redirect(url_for('index'))
if __name__ == '__main__':

    make_model()
    work()
    # live_scores = get_live_scores(competition_id)
    # player_data = get_player_data(match_id)
    # recent_matches = get_recent_matches(competition_id)
   
    rand_index = random.randint( 0, (len(matches)-6) )

    app.run(debug=True)

