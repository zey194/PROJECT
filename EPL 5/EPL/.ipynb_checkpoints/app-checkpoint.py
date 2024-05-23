import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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

    res = pd.DataFrame(columns=['Unnamed: 0', 'HTP', 'ATP', 'HM1_D', 'HM1_L', 'HM1_M', 'HM1_W', 'HM2_D', 'HM2_L', 'HM2_M', 'HM2_W', 'HM3_D', 'HM3_L', 'HM3_M', 'HM3_W', 'AM1_D', 'AM1_L', 'AM1_M', 'AM1_W', 'AM2_D', 'AM2_L', 'AM2_M', 'AM2_W', 'AM3_D', 'AM3_L', 'AM3_M', 'AM3_W', 'HTGD', 'ATGD', 'DiffFormPts'], index=[0])

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

    res['HTGD'] = x_df['HTGD']
    res['ATGD'] = x_df['ATGD']
    res['DiffFormPts'] = x_df['DiffFormPts']

    return res


def make_model(X_train1=None, pred=None):
    global classifier

    dataset = pd.read_csv('final_dataset.csv')

    dataset2 = dataset.copy().drop(columns =['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG',
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
        columns = ['Unnamed: 0', 'HTP', 'ATP', 'HM1', 'HM2', 'HM3', 'AM1', 'AM2', 'AM3', 'HTGD', 'ATGD', 'DiffFormPts']
        x_df = pd.DataFrame(X_train1, columns=columns)
        X_all = X_all = pd.concat([X_all, x_df], ignore_index=True)

        pred = 'H'
        pred_series = pd.Series(pred)
        y_all = y_all = pd.concat([y_all, pred_series], ignore_index=True)



    cols = [['HTGD','ATGD','HTP','ATP']]
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

    # Fitting Logistic Regression to the Training set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 2)
    classifier.fit(X_train, y_train)


def make_pred(X_train1):

    global classifier
    
    index = 0

    X_train1 = [ [index] + X_train1]

    columns = ['Unnamed: 0', 'HTP', 'ATP', 'HM1', 'HM2', 'HM3', 'AM1', 'AM2', 'AM3', 'HTGD', 'ATGD', 'DiffFormPts']

    x_df = pd.DataFrame(X_train1, columns=columns)

    x_df = preprocess_features(x_df)
    
    ans = preprocess_features2(x_df)

    pred = classifier.predict(ans)

    return pred[0]


def work():

    global dataset, dataset2, matches, players_by_team, display_data
    dataset = pd.read_csv('final_dataset.csv')

    dataset2 = dataset.copy().drop(columns =['FTR', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG',
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

    new_names = {'wolverhampton wanderers' : 'wolves', 'brighton and hove albion' : 'brighton', 'leicester city': 'leicester', 'manchester city': 'man city', 'tottenham hotspur':'tottenham', 'manchester united':'man united', 'norwich city':'norwich', 'west ham united': 'west ham', 'newcastle united':'newcastle'}
    
    players_by_team = {new_names.get(old_key, old_key): value for old_key, value in players_by_team.items()}


    players_by_team['birmingham'] = ["John Smith","Michael Johnson","Christopher Brown","Daniel Davis","David Martinez","Joseph Anderson","James Taylor","Robert Hernandez","William Gonzalez","Richard Wilson","Thomas Moore","Matthew Miller"]
    players_by_team['blackburn'] = ["Benjamin Thompson","Lucas Rodriguez","Nathan Wright","Samuel Martinez","Oliver Harris","Alexander Nelson","William Carter","Henry Mitchell","Ethan Lopez","Daniel Taylor","James Scott","Jacob King"]
    players_by_team['blackpool'] = ["Michael Johnson", "David Anderson", "Christopher Martinez","Matthew Turner", "Joseph Wilson", "Andrew Garcia","Daniel Lee", "Ryan Rodriguez", "Tyler Hernandez","John Smith", "Robert Brown", "Joshua Jones"]
    players_by_team['bolton'] = [ "Kevin Thompson", "Brian Harris", "Anthony Martinez", "Justin Clark", "Steven Adams", "Brandon Lewis", "Eric Hall", "Adam White", "James Robinson", "Thomas Moore", "William Taylor", "Charles Jackson"]
    players_by_team['bradford'] = ["Michael Johnson", "Christopher Brown", "Matthew Davis", "Daniel Miller", "Mark Wilson", "Richard Garcia", "Joseph Rodriguez", "David Martinez", "Anthony Hernandez", "Paul Lopez", "Andrew Gonzalez", "Joshua Young"]
    players_by_team['cardiff'] = ["Jason Thompson", "Brian Lewis", "Kevin Scott", "Justin Green", "Brandon Hall", "Eric King", "Steven White", "Adam Adams", "Ryan Harris", "Thomas Clark", "Jeffrey Baker", "Gary Turner"]
    players_by_team['charlton'] = ["Christopher Martinez", "Matthew Robinson", "Anthony Wright", "Daniel Lopez", "Kenneth Lee", "George Walker", "Ronald Perez", "Edward Hall", "James Gonzalez", "Timothy Young", "Joshua Martinez", "Michael Rodriguez"]
    players_by_team['coventry'] = ["Brian Foster", "Thomas Martinez", "David Scott", "Steven Nelson", "Charles King", "Joseph Harris", "Paul Wright", "Mark Martinez", "Donald Miller", "Jason Rodriguez", "Jeffrey Clark", "Kevin Lewis"]
    players_by_team['derby'] = ["Robert Johnson", "Michael Brown", "William Davis", "Richard Wilson", "James Taylor", "John Anderson", "Thomas Thomas", "Daniel Jackson", "Matthew White", "Christopher Harris", "Joseph Martin", "David Thompson"]
    players_by_team['fulham'] = ["Andrew Clark", "Ryan Martinez", "Kevin Lewis", "Jeffrey Lee", "Brian Walker", "Timothy Hall", "Jason Young", "Eric Allen", "Steven Hernandez", "Scott King", "Justin Wright", "Brandon Hill"]
    players_by_team['huddersfield'] = ["Christopher Martinez", "Matthew Lopez", "Daniel Taylor", "James Martinez", "David Thomas", "Brian Garcia", "Jose Rodriguez", "William Hernandez", "Anthony Martinez", "Michael Rodriguez", "Charles Martinez", "Joseph Rodriguez"]
    players_by_team['hull'] = ['Jessica Green', 'Michael Scott', 'Emily Johnson', 'Justin White', 'Megan Lee', 'Kevin Martinez', 'Rachel Davis', 'Tyler Wilson', 'Olivia Thompson', 'Daniel Harris', 'Lauren Turner', 'Matthew Clark']
    players_by_team['ipswich'] = ['Sarah Adams', 'Ryan Parker', 'Amanda Miller', 'Brandon Mitchell', 'Jennifer Taylor', 'Christopher Brown', 'Stephanie Anderson', 'Andrew Martinez', 'Elizabeth Thomas', 'David Garcia', 'Rebecca Rodriguez', 'John Jackson']
    players_by_team['leeds'] = ['Michael Wilson', 'Jessica Lee', 'Matthew Harris', 'Lauren Martinez', 'Daniel Thompson', 'Ashley Clark', 'Christopher Walker', 'Emily Rodriguez', 'Joshua White', 'Michelle Scott', 'Nicholas Lewis', 'Sarah Hall']
    players_by_team['middlesboro'] = ['Alexis Johnson', 'Brandon Taylor', 'Olivia Thomas', 'Jacob Moore', 'Ava Garcia', 'Ethan Martinez', 'Sophia Anderson', 'William Brown', 'Mia Wilson', 'James Davis', 'Charlotte Miller', 'Daniel Rodriguez']
    players_by_team['middlesbrough'] = ['Elijah Clark', 'Amelia Young', 'Michael Scott', 'Emily White', 'Ryan Lewis', 'Avery King', 'Lucas Turner', 'Madison Hill', 'Logan Adams', 'Harper Carter', 'Jackson Wright', 'Evelyn Lopez']
    players_by_team['portsmouth'] = ['Emma Martinez', 'William Johnson', 'Olivia Brown', 'James Wilson', 'Isabella Anderson', 'Daniel Thompson', 'Sophia Harris', 'Alexander Davis', 'Mia Rodriguez', 'Benjamin Martinez', 'Charlotte Taylor', 'Jacob Garcia']
    players_by_team['qpr'] = ['Liam Smith', 'Emma Johnson', 'Noah Williams', 'Olivia Brown', 'William Jones', 'Ava Davis', 'James Wilson', 'Sophia Miller', 'Alexander Taylor', 'Isabella Moore', 'Ethan Anderson', 'Charlotte White']
    players_by_team['reading'] = ['Oliver Martinez', 'Sophia Taylor', 'Liam Johnson', 'Emma Brown', 'Noah Garcia', 'Olivia Rodriguez', 'William Hernandez', 'Ava Martinez', 'James Lopez', 'Isabella Perez', 'Ethan Gonzalez', 'Charlotte Moore']
    players_by_team['stoke'] = ['Elijah Anderson', 'Amelia Martinez', 'James Johnson', 'Olivia Brown', 'William Garcia', 'Charlotte Wilson', 'Michael Rodriguez', 'Ava Hernandez', 'Alexander Lopez', 'Sophia Perez', 'Daniel Gonzalez', 'Mia Moore']
    players_by_team['sunderland'] = ['Liam Taylor', 'Emma Thomas', 'Noah White', 'Olivia Harris', 'William Clark', 'Ava Lewis', 'James Turner', 'Isabella Baker', 'Logan Hall', 'Sophia Wright', 'Benjamin Hill', 'Emily Parker']
    players_by_team['swansea'] = ['Jacob Johnson', 'Sophia Martinez', 'Mason Anderson', 'Ava Taylor', 'William Jackson', 'Isabella Thompson', 'James Harris', 'Olivia Nelson', 'Benjamin White', 'Charlotte Martinez', 'Elijah Robinson', 'Amelia Lewis']
    players_by_team['west brom'] = ['Liam Brown', 'Emma Rodriguez', 'Noah Garcia', 'Olivia Martinez', 'William Lopez', 'Ava Wilson', 'James Martinez', 'Isabella Robinson', 'Oliver Clark', 'Sophia Walker', 'Benjamin Young', 'Mia Hernandez']
    players_by_team['wigan'] = ['Ethan Taylor', 'Amelia Lewis', 'Michael White', 'Sophia Thompson', 'Alexander Hall', 'Ava Garcia', 'William Martin', 'Olivia Martinez', 'James Robinson', 'Isabella Johnson', 'Benjamin Brown', 'Mia Davis']


# ! Flask App
app = Flask(__name__)
csrf = CSRFProtect(app)
app = Flask(__name__)
app.secret_key = 'your_secret_key' 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)


class MyForm(FlaskForm):
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


@app.route('/')
def index():
    global chosen_data_matches, display_data, matches, rand_index

    if 'username' in session:
        match_data = [(index, match) for index, match in enumerate( matches[ rand_index : rand_index+5 ] )]
        chosen_data_matches = display_data[ rand_index : rand_index+5 ]
        return render_template('index.html', match_data=match_data)
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
    global chosen_data_matches, X_train1, display_data, rand_index

    if chosen_data_matches == []:
        chosen_data_matches = display_data[ rand_index : rand_index+5 ]

    dataset2_values = dataset2.values.tolist()[match_index]
    comments = Comment.query.filter_by(match_index=match_index).order_by(Comment.timestamp.desc()).all()
    form = MyForm()

    if form.validate_on_submit():

        X_train = [ (form.data1.data) ,  (form.data2.data) ,  (form.data3.data) ,  (form.data4.data) ,  (form.data5.data) ,  (form.data6.data) ,  (form.data7.data) ,  (form.data8.data) ,  (form.data9.data) ,  (form.data10.data) ,  (form.data11.data) ]
        X_train1 = X_train
        pred = make_pred(X_train)

        return render_template('match_data2.html', pred=pred, dataset2_values=dataset2_values, form=form, comments=comments, match_index=match_index, players_by_team=players_by_team, display_data=chosen_data_matches[match_index])

    return render_template('match_data.html', dataset2_values=dataset2_values, form=form, comments=comments, match_index=match_index, players_by_team=players_by_team, display_data=chosen_data_matches[match_index])


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
            X_train1 = [ (form.data1.data) ,  (form.data2.data) ,  (form.data3.data) ,  (form.data4.data) ,  (form.data5.data) ,  (form.data6.data) ,  (form.data7.data) ,  (form.data8.data) ,  (form.data9.data) ,  (form.data10.data) ,  (form.data11.data) ]
        
        make_model(X_train1, form.pred.data) # Retrain the model

    print('\n', msg, '\n')
    
    return redirect(url_for('index'))


@app.route('/clear_form2', methods=['POST'])
def clear_form2():

    return redirect(url_for('index'))


if __name__ == '__main__':

    make_model()
    work()

    rand_index = random.randint( 0, (len(matches)-6) )

    app.run(debug=True)

