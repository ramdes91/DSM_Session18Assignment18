
# coding: utf-8

# In[11]:


# import libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
from ipykernel import kernelapp as app
import matplotlib.pyplot as plt
import math


# Problem 1

# In[12]:


# Create DataFrame from the given Data
lst_qualification = ['High School','Bachelors','Masters','PHD']
lst_female = [60,54,46,41]
lst_male = [40,44,53,57]
df=pd.DataFrame({'Qualification':lst_qualification,'Count_F': lst_female ,'Count_M': lst_male})
df


# In[13]:


##USING Z SCORE AND p VALUE

# Add column in the Dataframe for Mean, Standard Deviation, Z Score
# and P Values for Female(F) and Male (M)

df['Mean_F']=df['Count_F'].mean()
df['Mean_M']=df['Count_M'].mean()

df['Std_Dev_F']=df['Count_F'].std()
df['Std_Dev_M']=df['Count_M'].std()

df['Z_F']=stats.zscore(df['Count_F'])
df['Z_M']=stats.zscore(df['Count_M'])

df['p_F']=[stats.norm.cdf(pval) for pval in stats.zscore(df['Count_F'])]
df['p_M']=[stats.norm.cdf(pval) for pval in stats.zscore(df['Count_M'])]
df.head()


# In[14]:


print('Conclutions from the above table: \npvalue of Male and Female (more than 5%, there is a relationship \n''between the gender of an individual and the level of education that they have obtained.\n')

print('Female populations is more at High School and Bachelors')
print('Female populations is less at Masters and PHD\n')

print('Male populations is less at High School and Bachelors')
print('Male populations is more at Masters and PHD')


# In[15]:


##Using Chi-square test

# redefine the dataset
df=df[['Qualification','Count_F','Count_M']]

N = 395          # Sample Size
df['Count_Total']=df.Count_F+df.Count_M

# Expected frequency = ((row total×column)/total sample size
df['ef_F']=(df.Count_F.sum()*df.Count_Total)/N
df['ef_M']=df.Count_Total-df.ef_F

# Chi Sqaure value χ2=∑(Observe freq−Expected Freq)2/Expected Freq 
df['chi_F']=[(math.pow((df.Count_F.values[i]-df.ef_F.values[i]),2))/df.ef_F.values[i] for i in range(df.Count_F.count())]
df['chi_M']=[(math.pow((df.Count_M.values[i]-df.ef_M.values[i]),2))/df.ef_M.values[i] for i in range(df.Count_M.count())]
df


# In[17]:


chi_sq_stat =df.chi_F.sum() + df.chi_M.sum()
print("Chi-Square Test Statstic value:\t", chi_sq_stat)
dof = 3       # Degree of Freedom - here dof =3 

# Calculate P value from chi_square_stat and degree of freedom using cdf function
p_val = 1 - stats.chi2.cdf(chi_sq_stat,dof) 
print("Chi-Square P value\t\t", p_val)

α =0.05  # significance level, confidence level 95%

#Calculate chi-square crtical value
chi_critical= stats.chi2.ppf(0.95,dof)
print("Chi-Square Test Critical value:\t", chi_critical)

print('\nAs Chi-Square Test Statstic value (8.006) greater than Chi-Square Test Critical value (7.815)'       '\nby Null hypothesis, it can be concluded Education level depends on gender (at 5% significance level)')


# Problem 2

# In[20]:


# Create DataFrame from the given Data
lst_group1 = [51, 45, 33, 45, 67]
lst_group2 = [23, 43, 23, 43, 45]
lst_group3 = [56, 76, 74, 87, 56]
df=pd.DataFrame({'Group1':lst_group1,'Group2': lst_group2 ,'Group3': lst_group3})
df


# In[22]:


p_Val=stats.f_oneway(df['Group1'],df['Group2'],df['Group3']).pvalue
F_Val=stats.f_oneway(df['Group1'],df['Group2'],df['Group3']).statistic

α = 0.05                    # Significance level, confidence level 95%

print('Null Hypothesis: \t Group1=Group2=Group3')

print('\nHypothesis testing with 5% significance')

print('\nHere p Value greater than α , so Null Hypothesis(Group1=Group2=Group3) can be Accepted. ')

print('\nWriting up the results in APA format:')

print('\t Significance level:\t', round(α,4))
print('\t F Value:\t\t', round(F_Val,4))
print('\t p Value:\t\t', round(p_Val,4), ' <', round(α,4) , '(Significance level)' )
print('\t So, Accept Null Hypothesis: \t Group1=Group2=Group3' )


# Problem 3

# In[24]:


# Create DataFrame from the given Data
lst_group1 = [10,20,30,40,50]
lst_group2 = [5,10,15, 20, 25]

df=pd.DataFrame({'Group1':lst_group1,'Group2': lst_group2})
df


# In[26]:


# Add column in the Dataframe for Mean, Standard Deviation and Variance

df['Mean_Group1']=df['Group1'].mean()
df['Mean_Group2']=df['Group2'].mean()

df['Std_Dev_Group1']=df['Group1'].std()
df['Std_Dev_Group2']=df['Group2'].std()

df['Var_Group1']=df['Group1'].var()
df['Var_Group2']=df['Group2'].var()
df


# In[27]:


# Calculate the P Values
# Hypothesis Test
print('Null Hypothesis Group1 = Group2') 

α =0.05  # significance level, confidence level 95%
print('\nSignificance level:\t', round(α,4))

# F test
# F-Test Formula:\t (Varience of Group 1)/(Varience of Group 1)
F_Val=df['Group1'].var()/df['Group2'].var()
print('F Test Results:\t\t',F_Val)

p_Val = stats.f.cdf(F_Val, len(df['Group1'])-1,len(df['Group1'])-1)

print('p Values is:\t\t',p_Val)

print('\nHere:\t p Value:\t', round(p_Val,4), ' >', round(α,4) , '(Significance level)' )
print('\t So, Reject Null Hypothesis: \t Group1=Group2' )

