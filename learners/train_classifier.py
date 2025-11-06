import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
import joblib

def train_flood_classifier(data_path, model_path_voting, model_path_lr):
    df = pd.read_csv(data_path)

    features = [
        'Rainfall_today', 'DrainLevel_today', 'RoadLevel_today',
        'SoilMoisture_today', 'Rainfall_tomorrow', 'FloodProbability'
    ]
    targets = ['FloodRiskLevel'] 
    
    ## 'EventSeverity', 'AlertLevel'
    ## df['AlertLevel'] = df['AlertLevel'].fillna('None')

    X = df[features]
    y = df[targets]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler(with_mean=False, with_std=True)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # LogisticRegression as before
    base_clf = LogisticRegression(
        C=75.4312,
        class_weight=None,
        penalty='l2',
        solver='newton-cg',
        multi_class='ovr',
        max_iter=1000
    )
    lr_clf = MultiOutputClassifier(base_clf)
    lr_clf.fit(X_train_scaled, y_train)
    y_pred_lr = lr_clf.predict(X_test_scaled)

    print("\n=== Logistic Regression Classification Results ===")
    for i, col in enumerate(targets):
        acc = accuracy_score(y_test.iloc[:, i], y_pred_lr[:, i])
        print(f"{col}: {acc:.4f}")
        print(classification_report(y_test.iloc[:, i], y_pred_lr[:, i]))

    joblib.dump({'scaler': scaler, 'classifier': lr_clf}, model_path_lr)
    print("\nLogisticRegression model saved to flood_classifier_lr.joblib")

    # VotingEnsemble (Soft voting, can add/remove base models as needed)
    voting_estimator = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(
                C=75.4312, class_weight=None, penalty='l2',
                solver='newton-cg', multi_class='ovr', max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('dt', DecisionTreeClassifier(random_state=42))
        ],
        voting='soft'
    )
    voting_clf = MultiOutputClassifier(voting_estimator)
    voting_clf.fit(X_train_scaled, y_train)
    y_pred_vote = voting_clf.predict(X_test_scaled)

    print("\n=== VotingEnsemble Classification Results ===")
    for i, col in enumerate(targets):
        acc = accuracy_score(y_test.iloc[:, i], y_pred_vote[:, i])
        print(f"{col}: {acc:.4f}")
        print(classification_report(y_test.iloc[:, i], y_pred_vote[:, i]))

    joblib.dump({'scaler': scaler, 'classifier': voting_clf}, model_path_voting)
    print("\nVotingEnsemble model saved to flood_classifier_voting.joblib")

    return scaler, lr_clf, voting_clf

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    transformed_data_path = project_root / 'data' / 'Flood-Data-Transformed.csv'
    model_path_voting = project_root / 'models' / 'flood_classifier_voting.joblib'
    model_path_lr = project_root / 'models' / 'flood_classifier_lr.joblib'

    scaler, lr_clf, voting_clf = train_flood_classifier(transformed_data_path, model_path_voting, model_path_lr)

