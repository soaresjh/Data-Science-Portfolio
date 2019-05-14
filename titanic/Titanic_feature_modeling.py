def modeling(file_path):
    import pandas as pd
    import numpy as np

    from pandas import get_dummies as dummies 

    ##

    def read_csv_col(col_list):
        df = pd.read_csv(file_path,
                         usecols = col_list
                        )
        return df

    ## Age

    x_age = read_csv_col(['Age']);

    age_mean = np.mean(x_age['Age']);
    x_age.fillna(age_mean, inplace = True);

    n_bins = 8
    _, age_bins = pd.cut(x_age['Age'], bins = n_bins, retbins=True)

    def helper_Age(value):
        is_in_bin = 0;
        for bound in age_bins:
            if value <= bound:
                return is_in_bin
            is_in_bin += 1

    x_age['Age'] = x_age['Age'].apply(helper_Age)/n_bins

    ## Embarked

    x_embarked = read_csv_col(['Embarked']);

    embarked_top = list(x_embarked['Embarked'].value_counts().axes[0])[0]
    x_embarked.fillna(embarked_top, inplace = True)
    x_embarked.replace({'C':'Cherbourg', 'S':'Southampton', 'Q':'Queenstown'}, inplace = True)

    x_embarked = dummies(x_embarked['Embarked'], prefix = 'Embark')

    ## Cabin

    x_cabin = read_csv_col(['Cabin'])

    flours = ('A','B','C','D','E','F','G');

    def helper_cabin(element):
        for level in flours:
            if level in element:
                return level
        return element

    x_cabin['Cabin'].replace({'T':'NaN'}, inplace = True)
    x_cabin['Cabin'].fillna('NaN', inplace = True)
    x_cabin['Cabin'] = x_cabin['Cabin'].apply(helper_cabin)

    x_cabin = dummies(x_cabin['Cabin'], prefix = 'Cabin')
    x_cabin.drop('Cabin_NaN', axis = 1, inplace = True)

    def helper_cabin_allocat(pdSerie):

        if pdSerie['Pclass'] == 1:
            floors = ['A','B','C','D','E']
            if pdSerie['Related'] > 2:
                floors = ['B', 'C']
            elif pdSerie['Related'] > 4:
                floors = ['C']

        elif pdSerie['Pclass'] == 2:
            floors = ['D','E','F']
            if pdSerie['Related'] > 1:
                floors = ['F']
        else:
            floors = ['E','F','G']
            if pdSerie['Related'] > 1:
                floors = ['G']

        col_floors = []
        for f in floors:
            col_floors += ['Cabin_' + f]

        returnSerie = pdSerie.copy(deep = True)

        for c in col_floors:
            returnSerie[c] = 1

        return returnSerie

    data = read_csv_col(['SibSp', 'Parch', 'Pclass']) 

    x_cabin['Related'] = data['SibSp']+data['Parch'];
    x_cabin['Pclass']  = data['Pclass']

    x_cabin = x_cabin.apply(helper_cabin_allocat, axis = 1)
    x_cabin.drop(['Related','Pclass'],axis = 1, inplace = True)

    x_name = read_csv_col(['Name'])

    name_title = ('Mr', 'Mrs', 'Ms', 'Mme', 'Miss', 'Mlle', 'Master', 'Dr', 'Don', 'Countess', 'Major', 'Cap', 'Col', 'Rev')

    def helper_name(elemente):
        for title in name_title:
            if title in elemente:
                return title
        return elemente

    x_name['Name'] = x_name['Name'].apply(helper_name)
    x_name['Name'].replace({'Mme':'Ms',
                            'Mlle':'Miss',
                            'Major':'Arm',
                            'Cap':'Arm',
                            'Col':'Arm',
                            'Countess':'Nobil',
                            'Don':'Nobil',
                            'Reuchlin, Jonkheer. John George':'Mr'
                           }, inplace = True)

    x_name = dummies(x_name, prefix = 'Title')

    ## Sex

    x_sex = read_csv_col(['Sex'])

    def helper_sex(element):
        if element == 'male':
            return 1
        elif element == 'female':
            return -1
        return 0

    x_sex['Sex'] = x_sex['Sex'].apply(helper_sex)

    ## Data

    drop_col = ['Age', 'Cabin', 'Embarked', 'Name', 'Ticket', 'Sex','PassengerId']

    x = pd.read_csv(file_path, usecols = lambda x: x not in drop_col)

    ## Normalize 

    def helper_normalize(pdColumn):
        maximun = pdColumn.max()
        return pdColumn/maximun

    x = x.apply(helper_normalize)

    x = pd.concat([x,
                   x_age,
                   x_cabin,
                   x_embarked,
                   x_name,
                   x_sex
                  ], axis = 1)

    columns = list(x.columns)
    selected =  ['SibSp', 'Fare', 'Age', 'Cabin_A', 'Cabin_C', 'Cabin_F', 'Cabin_G',
       'Title_Master', 'Sex']

    drop_col = list(set(columns)-set(selected))

    x.drop(drop_col, axis = 1, inplace = True)
    #x.to_csv('x_test.csv', index = False)
    return x