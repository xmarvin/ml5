y_pred1 = np.array([int(y>=0.5) for y in y_pred])
accuracy_score(y_pred1, y_valid)
zip1 = [a for a in zip(y_pred1, y_valid, range(0,len(y_pred1))) ]
true_neg_idx = [a[2] for a in zip1 if a[0]!=a[1] and a[0]==1]
false_neg_idx = [a[2] for a in zip1 if a[0]!=a[1] and a[0]==0]

x_valid1 = x_valid.reset_index()

true_negative_df = x_valid1.loc[true_neg_idx]
false_negative_df = x_valid1.loc[false_neg_idx]

false_negative_df.to_csv('false_negative.csv')
true_negative_df.to_csv('true_negative.csv')


false_pos_idx = [a[2] for a in zip1 if a[0]==a[1] and a[0]==0]
false_pos_df = x_valid1.loc[false_pos_idx]


(Pdb) accuracy_score(y_pred1, y_valid)
0.74042857142857144