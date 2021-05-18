import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
from statsmodels.stats.contingency_tables import SquareTable
import seaborn as sns
from PIL import Image
from scipy.stats import chi2

###########################################################################################
# done done 

def Khi2_conformity_count(df):     
    st.title('KHI² conformity test for sample counts')
    st.subheader('For Qk variable and a theoretical expectation')
    st.write('H0 : The sample counts in the k different classes of Qk \
                are consistent with the theoretical expectation.')
    st.text('Example : Are the sample cunts in the different corn "Color" as expected by theory ? \n\
             - 20 for "Yellow" \n\
             - 50 for "Red" \n\
             - 29 for "Yellow.red"')
    st.write('')
    obs1=df['Color']
    obs1=obs1.dropna()
    #st.dataframe(obs1,width=900) 
    nber=obs1.value_counts()
    #st.write(nber)
    fig=plt.figure()
    plt.style.use('ggplot')
    x = obs1.unique()
    #st.write(x)
    y= nber
    x_pos = [i for i, _ in enumerate(y)]
    plt.bar(x_pos, y, color='green')
    plt.xlabel("Color")
    plt.ylabel("Proportion")
    plt.title("")
    #st.write(y)
    plt.xticks(x_pos, y.index)
    col1,col2=st.beta_columns([0.5,1])
    with col1:
        st.pyplot(fig)
    with col2:
        st.text('Observed sample count for "Yellow" : {:.1f}'.format(nber[0]))
        st.text('Observed sample count for "Red" : {:.1f}'.format(nber[1]))
        st.text('Observed sample count for "Yellow.red" : {:.1f}'.format(nber[2]))
        st.text('Total observed sample count : {:.1f}'.format(nber.sum()))
    f_obs=y
    
    with st.echo():
        from scipy.stats import chisquare
        chi2,pvalue=chisquare(f_obs, f_exp=[20,50,29])
        # inputs :
        #    - f_obs :        observed sample counts
        #    - f_exp :        expected sample counts
        # returns : 
        #    - statistics :   Khi2 statistic value
        #    - pvalue :       p-value
    dof=3
    st.write("Chi2 = ",chi2)
    st.write("p-value = ",pvalue)
    st.text('In this example, the conformity of the observed proportion to theory [50%,20%,30%] can be rejected at a confidence level of 5%')
    st.subheader('Effect Size :')
    st.write('Theory is expected not being assessed from data.')
    st.write('Thus, degree of freedom is k, number of classes of Qk',3)
    st.write("Number of observations : ",obs1.shape[0])
    st.write("Proposed effect size is Cramer's V equal to ") 
    st.latex(r'V = \sqrt{\frac{\chi^2}{n.dof}}\newline where\ n\ is\ number\ of\ observations \newline and\ dof\ is\ degrees\ of\ freedom')
    CramerV=np.sqrt(chi2/obs1.shape[0])
    st.write("Cramer's V criteria is 0.1 for small ; 0.3 for medium ; 0.5 for large")
    st.text('In example, effect size is {:.2f} corresponding to a large effect size.'.format(CramerV))
    st.subheader('<<< See Khi2_conformity_count() function >>>')

###########################################################################################
#done done

def Khi2_homogeneity_count(df):
    st.title('Chi² homogeneity for sample counts')
    st.subheader('For 2 x Qk variables')
    st.write('H0 : The sample counts in classes of both Qk variables are independent.')
    st.text('Example : Are the sample counts in the "Color" and "Rooting" categories of corn dataset independent from each other ? (i.e. compatible with random sampling)') 
    obs1=df[['Rooting','Color']]
    obs1=obs1.dropna()
    fig=plt.figure()
    props = {}
    for yname in ["High","Very.high","Low","Middle"]:
        props[('Red', yname)] = {'color': 'tomato'}
        props[('Yellow', yname)] = {'color': 'goldenrod'}
        props[('Yellow.red', yname)] = {'color': 'peru'}
    mosaic(obs1, ['Color', 'Rooting'],properties=props,gap=0.015)
    plt.xlabel("Rooting vs Color") 
    plt.ylabel("Color")
    plt.title("")
    plt.tight_layout()
    plt.savefig('output.png')
    img=Image.open('output.png')
    col1,col2=st.beta_columns([1,1])
    with col1:
        st.image('output.png')
    conting=pd.crosstab(index=obs1['Rooting'], columns=obs1['Color'],margins=True)
    with col2:
        st.write(conting) 
    conting=pd.crosstab(index=obs1['Rooting'], columns=obs1['Color'],margins=False)
    
    with st.echo():
        from scipy.stats import chi2_contingency
        chi2,pvalue,dof,expected=chi2_contingency(observed=conting) 
        # inputs :
        #    - observed :     contingency table 
        # returns : 
        #    - chi2 :         Khi2 statistic value
        #    - pvalue :       p-value
        #    - dof :          degree of freedom
        #    - expected :     expected observations
    st.write("Chi2 = ",chi2)
    st.write("p-value = ",pvalue)
    st.write("DoF = ",dof)
    st.text("In this example, the two classes can be considered not to be independent with a confidence level of 5%.")
    expected=pd.DataFrame(data=expected,columns=conting.columns.tolist(),index=conting.index.tolist())
    st.write("Expected values = \n",expected)
    khi2=(conting-expected)**2/expected
    st.write("Splitted khi2 = \n",khi2)
    khi2pct=(conting-expected)**2/expected/chi2*100
    st.write("Splitted khi2 in % = \n",khi2pct)
    st.write("This last table allows assessing the correlated levels.")

    st.subheader('Effect Size :')
    st.write("Proposed effect size is Cramer's W equal to ")
    st.latex(r'W = \sqrt{\sum_{i=1}^{m}\frac{(P^o_i-P^e_i)^2}{P^e_i}}\newline where\ m\ is\ number\ of\ cells\ (c.k)\newline \ P^o_i\ are\ proportions\ \newline and \ P^e_i\ are\ expected\ proportions')
    st.latex(r'corresponding\ to\ W^2 = \frac{\chi^2}{m} ')
    st.write("Cohen's W criteria is 0.1 for small ; 0.3 for medium ; 0.5 for large")
    st.write()
    CramerW=np.sqrt(chi2/conting.shape[0]/conting.shape[1])  
    st.text('In example, effect size is {:.2f} corresponding to a large effect size.'.format(CramerW))
    st.subheader('<<< See Khi2_homogeneity_count() function >>>')

###########################################################################################
# done done

def G_count(df,Graph=False):
    st.title('-----------------------------------------------------------------------------------')
    st.title('G test for sample counts')
    st.subheader('For 2 x Qk variables')
    st.write('H0 : The sample counts in classes of both Qk variables are independent.')
    st.text('Example : Are the sample counts in the "Color" and "Rooting" categories of corn dataset independent from each other ? (i.e. compatible with random sampling)') 
    st.write('G test is based on neperien logartihmic formula')
            
    st.subheader('<<< See G_count() function >>>')
    obs1=df[['Rooting','Color']]
    obs1=obs1.dropna()
    conting=pd.crosstab(index=obs1['Color'], columns=obs1['Rooting'],margins=False)
    if Graph==True:
        #st.dataframe(obs1,width=900) 
        #st.write(nber)
        fig=plt.figure()
        #plt.style.use('ggplot')
        props = {}
        for yname in ["High","Very.high","Low","Middle"]:
            props[('Red', yname)] = {'color': 'tomato'}
            props[('Yellow', yname)] = {'color': 'goldenrod'}
            props[('Yellow.red', yname)] = {'color': 'peru'}

        mosaic(obs1, ['Color', 'Rooting'],properties=props,gap=0.015)
        plt.xlabel("Rooting vs Color") 
        plt.ylabel("Color")
        plt.title("")
        plt.tight_layout()
        #st.pyplot(fig)
        plt.savefig('output.png')
        img=Image.open('output.png')
        st.image('output.png')
        st.write(conting) 
    
    with st.echo():
        from scipy.stats import chi2_contingency
        G,pvalue,dof,expected=chi2_contingency(observed=conting,lambda_="log-likelihood") 
        # inputs :
        #    - observed :     contingency table 
        # returns : 
        #    - G :            G statistic value
        #    - pvalue :       p-value
        #    - dof :          degree of freedom
        #    - expected :     expected observations
    st.write("G = ",G)
    st.write("p-value = ",pvalue)
    st.write("DoF = ",dof)
    expected=pd.DataFrame(data=expected,columns=conting.columns.tolist(),index=conting.index.tolist())
    st.write("Expected values = \n",expected)
    Gsplit=2*conting*np.log(conting/expected)
    st.write("Splitted G = \n",Gsplit)
    st.write("table sum = ",Gsplit.sum().sum())
    Gsplit=2*conting*np.log(conting/expected)/G*100
    st.write("Splitted G in % = \n",Gsplit) 
    #st.write(Gsplit.sum().sum())
    
###########################################################################################
# done done librairies à trouver 

def Fisher_test_cxk_Count(df):
    st.title('Fisher exact test for independencies between sample counts on two classes')
    st.subheader('For 2 x Qk variables')
    st.write('H0 : The sample counts in classes of both Qk variables are independent.')
    st.text('Example : Are the sample counts in the "Prouting" and "Rooting" categories of corn dataset independent from each other ? (i.e. compatible with random sampling)') 
    obs1=df[['Sprouting.epi',"Rooting"]]
    obs1=obs1.dropna()
    fig=plt.figure()
    #plt.style.use('ggplot')
    col1,col2=st.beta_columns([1,1])
    props = {}
    for yname in ["Yes","No"]:
        props[('Low', yname)] = {'color': 'tomato'}
        props[('High', yname)] = {'color': 'goldenrod'}
        props[('Middle', yname)] = {'color': 'peru'}
        props[('Very.high', yname)] = {'color': 'olivedrab'}
    
    with col1:
        mosaic(obs1.sort_values('Sprouting.epi',ascending=False).sort_values('Rooting')  
           ,['Rooting','Sprouting.epi'],properties=props,gap=0.015)
        plt.xlabel("Rooting vs Sprouting.epi") 
        plt.ylabel("Sprouting.epi")
        plt.title("")
        plt.tight_layout()
        #st.pyplot(fig)
        plt.savefig('output.png')
        img=Image.open('output.png')
        st.image('output.png') 
    with col2:
        conting=pd.crosstab(index=obs1['Rooting'], columns=obs1['Sprouting.epi'],margins=False)
        st.write(conting.transpose())

    #with st.echo():
        # from scipy.stats import fisher_exact # only on 2x2 table !
        #from FisherExact import fisher_exact
        #res=fisher_exact(conting)    
    st.subheader('Python library to perform cxk Fisher Exact Test not found')
    st.subheader('Potential solution is to use R existing librairies as described in reference [TBD]')    
    st.subheader('<<< See Fisher_test_cxk_Count() function >>>\n\n\n')

###########################################################################################
# done done

def Mantel_Haenszel_Count(df):    
    st.title('Mantel-Haenszel test') 
    st.subheader('to check for independencies between two binary variables, within a third variable stratification ')
    st.subheader('For two Q2 variables and one Qk variable (for stratification)')
    st.write('H0 : The sample counts in classes of both Q2 variables are independent in each stratification of Qk variable')
    st.text("Example : Are binary variables 'Verse' and 'Attacked' independent for each 'Color' levels ? (i.e. compatible with random sampling)")     
    obs1=df[['Verse','Color',"Attacked"]]
    obs1=obs1.dropna()
    color_label=obs1.Color.unique()
    crosstab_list=[]
    for lab in color_label:
        st.write('For "Color" label = ',lab)
        tmp=obs1[obs1['Color']==lab]
        tmp=pd.crosstab(index=tmp['Verse'], columns=tmp['Attacked'],margins=True,\
                        rownames=['Verse'], colnames=['Attacked']) 
        tmp['Proportions of Yes']=tmp.apply(lambda row: row["Yes"]/(row['Yes']+row['No']),axis=1)
        col1,col2=st.beta_columns([0.5,1]) 
        #st.write(tmp)
        with col2 : 
            st.write ('row="Verse", col="Attacked"')
            st.dataframe(tmp)
        crosstab_list.append(tmp.loc[['Yes','No'],['Yes','No']])
        tmp=tmp.loc[['Yes','No'],['Proportions of Yes']]
        fig=plt.figure() 
        
        plt.bar(tmp.index.tolist(),tmp['Proportions of Yes'])
        plt.xlabel('Verse' )
        plt.title('for "Color" = '+lab)
        plt.ylabel('Proportions of "Attacked" = "Yes"') 
        with col1:
            st.pyplot(fig)  

    #for i in range(3):
    #    st.write(crosstab_list[i])
    
    with st.echo():
        from statsmodels.stats.contingency_tables import StratifiedTable
        # Input variable :
        #    - tables : list of contingency table for all k classes
        res=StratifiedTable(crosstab_list)
    st.write('.summary gives a complete summary of the analysis for a given alpha confidence')
    st.write("Result summary for alpha=5% : \n")
    with st.echo():
        summary=res.summary(alpha=0.05)
    st.dataframe(summary,height=500)
    st.write('\n\n\n')
    st.write('.test_null_odds gives the statistics and p-value ' + \
            'for alpha=5% whether all tables have an odd ratio of 1')
    st.write('An odd ratio of 1 means that the exposure (row variable)' + \
            'has no effect on the outcome (columns variable)')
    st.write('correction parameter is whether a "continuity correction" has to be applied' \
             ,'(to be set True in most cases)') 
    with st.echo():
        res2=res.test_null_odds(correction=False)
    st.write('null odds result without correction :', res2)
    with st.echo():
        res3=res.test_null_odds(correction=True)
    st.write('null odds result with correction :',res3)
    st.text('for this example, the independency H0 hypothesis can be rejected only if correction is taken into account (see reference [1] for more discussions).')
    st.text('Effect size analysis may give more information.')
    st.subheader('Effect Size :')
    st.write('Odd ratio can be taken as an effect size measurement.')
    st.write('Odd ratio effect analysis : 1.5 for small, 2.5 for medium and 4.3 for large ')
    st.text('In this example, an estimated odd ratio of 2.31 indicates a medium effect')   
    
    st.subheader('<<< See Mantel_Haenszel_Count() function >>>\n\n\n')
    
###########################################################################################
# DONE DONE
def Binomial_vs_Theory_Proportions(df):
    st.title('Binomial test for proportions')
    st.subheader('For Q2 variable and a theoretical expectation')
    st.write('H0 : The proportion of the total population splitted in the two Q2 classes is \
            as expected by theory.')
    st.text('Example : Is the proportion of attacked corn as expected by theory ? \n\
             - 60% of the corn is attacked.')
    obs1=df['Attacked'] 
    obs1=obs1.dropna()
    conting=obs1.value_counts()
    col1,col2=st.beta_columns([0.5,1]) 
    with col1 : 
        fig=plt.figure()
        plt.bar(conting.index.tolist(),conting.values.tolist(),color='blue')
        plt.xlabel("Attacked ?")
        plt.ylabel("Proportion")
        plt.title("")
        st.pyplot(fig)
    with col2 : 
        st.write('Count : \n',conting)
    total=conting.sum()
    with st.echo():    
        from scipy.stats import binom_test
        Ptheo=60 # theoretical proportions assumed to be 60%
        pvalue=binom_test(x=conting['Yes'],n=100,p=Ptheo/100,alternative='two-sided')
    st.write('p-value for Probability of "Yes" is 60% : ',pvalue)
    st.text('Example : the proportion of "Yes" is not consistent with 60% theoretical expectation (rejected H0)')
    st.subheader('Effect Size : ')
    st.write(' Proposed effect size : distance between observed proportion and theoretical proportion :')
    st.text('In example, this effect size is {:.1f}%'.format(abs(conting['Yes']-Ptheo)))
    st.write("Note : Other possible effect size = Cohen's g = Pobserved - 50%")
    
    st.subheader('Other theoretical assumption (Theory : 60% for "No") : ')
    with st.echo():    
        from scipy.stats import binom_test
        Ptheo=60 # theoretical proportions assumed to be 60%
        pvalue=binom_test(x=conting['No'],n=100,p=Ptheo/100,alternative='two-sided')
    st.write('p-value for Probability of "No" is 60% : ',pvalue)
    st.text('Example : the proportion of "No" seems consistent with 60% theoretical expectation.(H0 cannot be rejected)')
    st.subheader('Effect Size : ')
    st.text('In example, effect size is {:.1f}%'.format(abs(conting['No']-Ptheo)))

    st.subheader('\n\n\n <<< See Binomial_vs_Theory_Proportions() function >>>\n\n\n')

###########################################################################################
# done done

def Khi2_conformity_proportion(df):
    st.title('KHI² conformity test for proportions')
    st.subheader('For Qk variable and a theoretical expectation')
    st.write('H0 : The proportions of the total population in the k different classes of Qk \
                are consistent with the theoretical expectation.')
    st.text('Example : Are the proportions of the different corn "Color" as expected by theory ? \n\
             - 50% for "Yellow" \n\
             - 20% for "Red" \n\
             - 30% for "Yellow.red"')
    st.write('')
    obs1=df['Color']
    obs1=obs1.dropna()
    #st.dataframe(obs1,width=900) 
    nber=obs1.value_counts()
    nber=nber/nber.sum()*100
    #st.write(nber)
    fig=plt.figure()
    plt.style.use('ggplot')
    x = obs1.unique()
    #st.write(x)
    y= nber
    x_pos = [i for i, _ in enumerate(y)]
    plt.bar(x_pos, y, color='green')
    plt.xlabel("Color")
    plt.ylabel("Proportion")
    plt.title("")
    #st.write(y)
    plt.xticks(x_pos, y.index)
    col1,col2=st.beta_columns([0.5,1])
    with col1:
        st.pyplot(fig)
    with col2:
        st.text('Observed proportion for "Yellow" : {:.1f}%'.format(nber[0]))
        st.text('Observed proportion for "Red" : {:.1f}%'.format(nber[1]))
        st.text('Observed proportion for "Yellow.red" : {:.1f}%'.format(nber[2]))
    f_obs=y
    
    with st.echo():
        from scipy.stats import chisquare
        chi2,pvalue=chisquare(f_obs, f_exp=[50,20,30])
        # inputs :
        #    - f_obs :        observed proportions (in %)
        #    - f_exp :        expected proportinos (in %)
        # returns : 
        #    - statistics :   Khi2 statistic value
        #    - pvalue :       p-value
    dof=3
    st.write("Chi2 = ",chi2)
    st.write("p-value = ",pvalue)
    st.text('In this example, the conformity of the observed proportion to theory [50%,20%,30%] can be rejected at a confidence level of 5%')
    st.subheader('Effect Size :')
    st.write('Theory is expected not being assessed from data.')
    st.write('Thus, degree of freedom is k, number of classes of Qk',3)
    st.write("Number of observations : ",obs1.shape[0])
    st.write("Proposed effect size is Cramer's V equal to ")
    st.latex(r'V = \sqrt{\frac{\chi^2}{n.dof}}\newline where\ n\ is\ number\ of\ observations \newline and\ dof\ is\ degrees\ of\ freedom')
    CramerV=np.sqrt(chi2/obs1.shape[0]/dof)
    st.write("Cohen's V criteria is 0.1 for small ; 0.3 for medium ; 0.5 for large")
    st.text('In example, effect size is {:.2f} corresponding to a small effect size.'.format(CramerV))
    st.subheader('<<< See Khi2_conformity_proportion() function >>>')
    


###########################################################################################
# A FINIR A FINIR 
def Khi2_homogeneity_proportion(df):
    st.title('KHI² homogeneity test for proportions')
    st.subheader('For 2 Q2 variables')
    st.write('H0 : The proportions in one of the two classes is the same in both categories of the other classe.')
    st.text('Example : Are the proportions of corn "Attacked" by bugs the same whether the corn is "Verse" or not ? \n')
    st.title('A FAIRE § 5.9 livre')
    st.subheader('<<< See Khi2_homogeneity_proportion() function >>>\n\n\n')

###########################################################################################
# done done  
def Fisher_test_2x2_Proportions(df):
    st.title('Fisher Exact Test for proportions (2x2 contingency table)')
    st.subheader('For 2 Q2 variables')
    st.write('H0 : The proportions in one of the two classes is the same in both categories of the other classe.')
    st.text('Example : Are the proportions of "Verse" corns the same whether the corn is "Sprouting" or not ? \n')
    obs1=df[['Sprouting.epi',"Verse"]]
    obs1=obs1.dropna()
    col1,col2=st.beta_columns([1.,1.])
    with col1 :
        fig=plt.figure()
        props = {}
        for yname in ["Yes","No"]:
            props[('Yes', yname)] = {'color': 'tomato'}
            props[('No', yname)] = {'color': 'goldenrod'}
            #props[('Middle', yname)] = {'color': 'peru'}
            #props[('Very.high', yname)] = {'color': 'olivedrab'}

        mosaic(obs1.sort_values('Sprouting.epi',ascending=False).sort_values('Verse',ascending=False)  
               ,['Verse','Sprouting.epi'],properties=props,gap=0.015)
        plt.xlabel("Verse vs Sprouting.epi") 
        plt.ylabel("Sprouting.epi")
        plt.title("")
        plt.tight_layout()
        #st.pyplot(fig)
        plt.savefig('output.png')
        img=Image.open('output.png')
        st.image('output.png') 
    with col2 :
        conting=pd.crosstab(index=obs1['Verse'], columns=obs1['Sprouting.epi'],margins=False)
        #conting=pd.DataFrame([[40,60],[40,60]],index=conting.index,columns=conting.columns)
        st.write(conting)
        st.write("'Verse' in rows, 'Sprouting' in columns")
        st.write('Total :',conting.sum().sum())
    
    with st.echo():
        from scipy.stats import fisher_exact # only on 2x2 table !
        #from FisherExact import fisher_exact
        odds_ratio,pvalue=fisher_exact(conting) 
    st.write("Odds ratio : ",odds_ratio)
    st.write("if odds ratio is 1, both proportions are equal")
    st.write("p-value regarding 'both proportions are equal' : ",pvalue)
    st.text("In this example, it is not possible to reject that the proportions are the same, with a 5% condidence level")
    st.subheader('Effect Size :')
    st.write('Odd ratio can be taken as an effect size measurement.')
    st.write('Odd ratio effect analysis : 1.5 for small, 2.5 for medium and 4.3 for large ')
    st.text('In this example, an estimated odd ratio of {:0.2f} indicates a small effect'.format(odds_ratio))

    st.subheader('<<< See Fisher_test_2x2_Proportions() function >>>\n\n\n')
    
###########################################################################################    
# done done

def Binomial_Paired_Proportions(df):
    st.title('Binomial Paired Proportions')
    st.subheader('For 1 Q2 variable and 1 QA variable')
    st.subheader('Equivalent to two Q2 variables not independent')
    st.write('H0 : The proportion for a binary variable is the same for two paired measurements (successive or on paired objects) ')
    st.text('Example : Are the proportions of "Verse" corns the same before ("Verse") and after a specific treatment ("Verse.Treatment") ? \n')
    obs1=df[['Verse','Verse.Treatment']]
    obs1=obs1.dropna()
    st.write("In this example, two measures are performed 'Verse' and 'Verse.Treatment' (class G)"  + \
            "with two outcomes 'Yes' or 'No' (Class F), both being not independent")
    st.text("\n\n\n")
    col1,col2=st.beta_columns([1.,1.])
    
    with col1:
        st.write("Each corn is tested twice before / after treatment as shown in following table :")
        st.write(obs1)
    with col2:
        conting=pd.crosstab(index=obs1['Verse'], columns=obs1['Verse.Treatment'],margins=False,\
                        rownames=['Verse'], colnames=['Verse.Treatment']) 
        st.write("Contingency table ('verse' in rows, 'verse.Treatment' in columns) : \n",conting)
        total=conting.sum().sum()
        st.write('Total : ',total)
    
    Ns=conting.loc['Yes','No']+conting.loc['No','Yes']
    st.write('Total number of inconsistent measurements Ns = : ', Ns)
    p12=conting.loc['Yes','No']/Ns*100
    p21=conting.loc['No','Yes']/Ns*100
    st.write('Probability of "Yes/No" inconsistency (given inconsistency) is {:0.1f}'.format(p12))
    st.write('Probability of "No/Yes" inconsistency (given inconsistency) is {:0.1f}'.format(p21))
    st.write('The purpose of the test is to check whether both these probabilities are identical, thus 0.5%')
    st.write('This corresponds to a Binomial test B(Nmax=Ns,p=0.5) applied to the number of ' +\
            'inconsistencies "Yes/No"')

    #st.write(conting)
    #conting=conting/total*100
    #st.write('Proportions (%) :\n',conting)
    with st.echo():    
        from scipy.stats import binom_test
        pvalue=binom_test(x=conting.loc['Yes','No'],n=Ns,p=0.5,alternative='two-sided')
    st.write('p-value for Probability of proportion equality : ',pvalue)
    st.text('In our example, both proportions can be considered as different with a confidence level more than 95%, i.e. the treatment seems to have an effect (H0 rejected)')

    st.subheader('Effect Size :')
    st.write('The effect size can be assessed by an odds ratio.')
    st.write('The odds ratio formula is different from independent case.')
    st.write('The odd ratio is given by :')
    st.latex(r'OR\ =\ max(\frac{P10}{P01},\ \frac{P01}{P10})\newline with\ P10 \ and \ P10\ being\ the\ discordant\ proportions.') 
    st.write('Odd ratio effect analysis : 1.5 for small, 2.5 for medium and 4.3 for large ')
    OR=max(conting.loc['Yes','No']/conting.loc['No','Yes'],conting.loc['No','Yes']/conting.loc['Yes','No'])
    st.text('In our example, the OR is {:0.2f} corresponding to a relatively large effect'.format(OR))
    
    st.subheader('<<< See Binomial_Paired_Proportions() function >>>\n\n\n')

###########################################################################################
# done done

def MacNemar_Paired_Proportions(df):
    st.title('MacNemar Paired Proportions')
    st.subheader('For 1 Q2 variable and 1 QA variable')
    st.subheader('Equivalent to two Q2 variables not independent')
    st.write('H0 : The proportion for a binary variable is the same for two paired measurements (successive or on paired objects) ')
    st.text('Example : Are the proportions of "Verse" corns the same before ("Verse") and after a specific treatment ("Verse.Treatment") ? \n')
    obs1=df[['Verse','Verse.Treatment']]
    col1,col2=st.beta_columns([1.,1.])
    obs1=obs1.dropna()
    with col1:
        st.write("Each corn is tested twice before / after treatment as shown in following table :")
        st.write(obs1)
    with col2:
        conting=pd.crosstab(index=obs1['Verse'], columns=obs1['Verse.Treatment'],margins=False,\
                    rownames=['Verse'], colnames=['Verse.Treatment']) 
        st.write("Contingency table ('verse' in rows, 'verse.Treatment' in columns) : \n",conting)
        total=conting.sum().sum()
        st.write('Total : ',total)    
    Ns=conting.loc['Yes','No']+conting.loc['No','Yes']
    st.write('Total number of inconsistent measurements Ns = : ', Ns)
    p12=conting.loc['Yes','No']/Ns*100
    p21=conting.loc['No','Yes']/Ns*100
    st.write('Probability of "Yes/No" inconsistency (given inconsistency) is {:0.1f}%'.format(p12))
    st.write('Probability of "No/Yes" inconsistency (given inconsistency) is {:0.1f}%'.format(p21))
    st.write('The purpose of the test is to check whether both these probabilities are identical, thus 0.5%')
    st.write('This corresponds to a Binomial test B(Nmax=Ns,p=0.5) applied to the number of ' +\
            'inconsistencies "Yes/No"')    
    #st.write(conting)
    #conting=conting/total*100
    #st.write('Proportions (%) :\n',conting)
    with st.echo():    
        from statsmodels.stats.contingency_tables import mcnemar
        # INPUTS :
        #     - table :       2x2 contingency table
        #     - exact :       boolean
        #                     if exact=True : binomial distribution is used
        #                     if exact=False : chi² approximation is used (valid for large sample only)
        #     - correction :  boolean
        #                     if correction=True and exact = False, apply the continuity correction
        #                     else no correction applied
        # OUTPUTS : 
        #     - statistics :  statistic result
        #                     chi² if exact is False
        #                     min(n12,n21) if exact is True (binary test) to be compared  with (n12+n21) / 2
        #                         where n12 and n21 are the number of inconsistency cases in table 
        #     - pvalue :     p-value
    st.write('with binomial exact test :')
    with st.echo():  
        res1=mcnemar(table=conting,exact=True)
    st.write('    statistics : min of n12 and n21 : ', res1.statistic)
    st.write('    to be compared with (n12+n21)/2 : ',(conting.loc['Yes','No']+conting.loc['No','Yes'])/2)
    #st.write('    min of p12 and p21 : {:0.2f}%'.format( \
    #         res1.statistic/(conting.loc['Yes','No']+conting.loc['No','Yes'])*100))         
    st.write('p-value : ',res1.pvalue)
    st.write('\n\n\n')
    st.write('with chi2 approximative test without continuity correction :')
    with st.echo():  
        res2=mcnemar(table=conting,exact=False,correction=False)
    st.write('    chi2 : ', res2.statistic)
    st.write('p-value : ',res2.pvalue)           
    st.write('\n\n\n')
    st.write('with chi2 approximative test with continuity correction :')
    with st.echo(): 
        res3=mcnemar(table=conting,exact=False,correction=True)
    st.write('    chi2 : ', res3.statistic)
    st.write('p-value : ',res3.pvalue)    
    st.text('In our example, both proportions can be considered as different with a confidence level of more than 95%, i.e. the treatment seems to have an effect (H0 rejected)')

    st.subheader('Effect Size :')
    st.write('The effect size can be assessed by an odds ratio.')
    st.write('The odds ratio formula is different from independent case.')
    st.write('The odd ratio is given by :')
    st.latex(r'OR\ =\ max(\frac{P10}{P01},\ \frac{P01}{P10})\newline with\ P10 \ and \ P10\ being\ the\ discordant\ proportions.') 
    st.write('Odd ratio effect analysis : 1.5 for small, 2.5 for medium and 4.3 for large ')
    OR=max(conting.loc['Yes','No']/conting.loc['No','Yes'],conting.loc['No','Yes']/conting.loc['Yes','No'])
    st.text('In our example, the OR is {:0.2f} corresponding to a relatively large effect'.format(OR))
    st.subheader('<<< See MacNemar_Paired_Proportions() function >>>\n\n\n')
        
###########################################################################################
# done done

def Mantel_Haenszel_Proportion(df):    
    st.title('Mantel Haesnzel Proportions')
    st.subheader('For 2 Q2 variables and 1 Qk variable that stratifies the data')
    st.write('H0 : The proportions between the two binary variables are identical in all categories of the thid variable that stratifies the data ')
    st.text('Example : Are the proportions between the binary variables "Attacked" and "Verse" the same for the different "Colors" ? \n')
    obs1=df[['Verse','Color',"Attacked"]]
    obs1=obs1.dropna()
    color_label=obs1.Color.unique()
    st.write('The proportions of "Attacked" = "Yes" given "Verse" for the different "Color" are :')
    crosstab_list=[]
    for lab in color_label:
        st.write('For "Color" label = ',lab)
        tmp=obs1[obs1['Color']==lab]
        tmp=pd.crosstab(index=tmp['Verse'], columns=tmp['Attacked'],margins=True,\
                        rownames=['Verse'], colnames=['Attacked']) 
        tmp['Proportions of Yes']=tmp.apply(lambda row: row["Yes"]/(row['Yes']+row['No']),axis=1)
        col1,col2=st.beta_columns([0.5,1]) 
        #st.write(tmp)
        with col2 : 
            st.write ('row="Verse", col="Attacked"')
            st.dataframe(tmp)
        crosstab_list.append(tmp.loc[['Yes','No'],['Yes','No']])
        tmp=tmp.loc[['Yes','No'],['Proportions of Yes']]
        fig=plt.figure() 
        
        plt.bar(tmp.index.tolist(),tmp['Proportions of Yes'])
        plt.xlabel('Verse' )
        plt.title('for "Color" = '+lab)
        plt.ylabel('Proportions of "Attacked" = "Yes"') 
        with col1:
            st.pyplot(fig)   
    with st.echo():
        from statsmodels.stats.contingency_tables import StratifiedTable
        # Input variable :
        #    - tables : list of contingency table for all k classes
        res=StratifiedTable(crosstab_list)
    st.write('.summary gives a complete summary of the analysis for a given alpha confidence')
    st.write("Result summary for alpha=5% : \n")
    with st.echo():
        summary=res.summary(alpha=0.05)
    st.dataframe(summary,height=500)
    st.write('\n\n\n')
    st.write('.test_equal_odds gives the statistics and p-value \
            for alpha=5% whether all tables have similar odd ratios')
    with st.echo():
        res4=res.test_equal_odds()
    st.write('equal odds result :',res4)   
    st.text('For this example, we do not have enough proof that we can reject the hypothesis of constant proportions between "Attacked" and "Verse" for each class "Color".')
    st.text('Effect size analysis may give more information.')
    st.subheader('Effect Size :')
    st.write('          Method to be defined')
    st.write('           Probably based on odd ratio varaibility between the different "Color".')
    #st.write('Odd ratio can be taken as an effect size measurement.')
    #st.write('Odd ratio effect analysis : 1.5 for small, 2.5 for medium and 4.3 for large ')
    #st.text('In this example, an estimated odd ratio of 2.31 indicates a medium effect')    
    st.subheader('<<< See Mantel_Haenszel_Proportion() function >>>\n\n\n')    
        
###########################################################################################
# done done
def Khi2_homogeneity_proportion_several(df):
    st.title('Chi² homogeneity for several proportions')
    st.subheader('For 1 Q2 variables and 1 Qk variable that stratifies the data.')
    st.write('H0 : The proportion of the binary variable is constant for all classes of the variable that stratifies the data. ')
    st.text('Example : Are the proportions in the binary "Verse" class the same for the different "Colors" ? \n')
    obs1=df[['Verse','Color']]
    obs1=obs1.dropna()
    conting=pd.crosstab(index=obs1['Color'], columns=obs1['Verse'],margins=False)
    st.write('with proportions of "No" in the different classes')
    conting['Proportions']=conting.apply(lambda row: row["No"]/(row['No']+row["Yes"])*100,axis=1) 
    conting['N by colour']=conting.apply(lambda row: (row['No']+row["Yes"]),axis=1) 
    st.dataframe(conting)
    Ntot=conting["N by colour"].sum()
    st.write('total =',Ntot )
    Ntot1=conting["No"].sum() 
    st.write('total for "No" =',Ntot1 )
    Ptot1=Ntot1/Ntot
    st.write("Proportion of 'No' =",Ptot1*100)
    with st.echo():
        from scipy.stats import chi2
        # Khi2 computed based on formula at page 398 of reference [TBD]
        conting['tmp']=conting.apply(lambda row: \
                    row["N by colour"]*(row["Proportions"]/100-Ptot1)**2/Ptot1/(1-Ptot1),axis=1)
        khi2=conting['tmp'].sum()
        # dof is the number of k levels minus 1 (used to compute the total)
        dof=conting.shape[0]-1
        # pvalue is the upper probability associated to khi2 for (1-cdf)
        # sf function is (1-cdf) where cdf is cumulative distribution function
        pvalue=chi2.sf(khi2,dof)    
    st.write("Chi2 = ",khi2)
    st.write("Degrees of Freedom (k-1) = ",dof)
    st.write("p-value = ",pvalue)
    st.write('As p-value is below 5%, the proportions of "No" in the different k classes are different \
            (H0 rejected)')
    st.subheader('Effect Size :')
    st.write("Number of observations : ",obs1.shape[0])
    st.write("Proposed effect size is Cramer's V equal to ")
    st.latex(r'V = \sqrt{\frac{\chi^2}{n.dof}}\newline where\ n\ is\ number\ of\ observations \newline and\ dof\ is\ degrees\ of\ freedom')
    CohenD=np.sqrt(khi2/obs1.shape[0]/dof)
    st.write("Cohen's V criteria is 0.1 for small ; 0.3 for medium ; 0.5 for large")
    st.text('In example, effect size is {:.2f} corresponding to a small/medium effect size.'.format(CohenD))
    st.subheader('<<< See Khi2_homogeneity_proportion_several() function >>>\n\n\n')

###########################################################################################    
# done done

def Cochran_Description():
    st.subheader('Cochran condition test is as follows :')
    st.write('   - a minimum of 5 elements in each classes')
    st.write('   - it is allowed to have classes with between 1 and 5 elements' +\
             'only if 80% of all classes have more than 6 elements')    
    
###########################################################################################
# done done
def Khi2_proportion_several(df):
    st.title('Chi² test for several proportions to be compared with theoretical expectations')
    st.subheader('For 1 Q2 variable and 1 Qk variable that stratifies the data, and a theoretical hypothesis.')
    st.subheader("WARNING : Cochran's Rule has to be respected !!!!!!")
    st.write('H0 : The proportion of the binary variable follows the theoretical expectation defined for all categories of the variable that stratifies the data. ')
    st.text('Example : Are the proportions in the binary varable "Verse" consistent with the following theory: ')
    st.text(' - 50% for "color : red"')
    st.text(' - 60% for "color : Yellow"')
    st.text(' - 40% for "color : Yellow.red"')  
    obs1=df[['Verse','Color']]
    obs1=obs1.dropna()
    conting=pd.crosstab(index=obs1['Color'], columns=obs1['Verse'],margins=False)
    st.write('with proportions of "No" in the different classes')
    conting['Proportions']=conting.apply(lambda row: row["No"]/(row['No']+row["Yes"])*100,axis=1) 
    conting['N by level']=conting.apply(lambda row: (row['No']+row["Yes"]),axis=1) 
    st.dataframe(conting)
    Ntot=conting["N by level"].sum()
    st.write('total =',Ntot )   
    with st.echo():       
        from statsmodels.stats.proportion import proportions_chisquare
        # INPUTS :
        #     - count :     number of success in each level (vector size k)
        #     - nobs :      number of trials in each level (vector size k)
        #     - value :     expected success probabilities for each level (vector size k)
        # OUTPUTS :
        #     - Chi²
        #     - p-value
        Ptheo=np.array([0.50,0.60,0.40])
        khi2,pvalue,_=proportions_chisquare(count=conting["No"], nobs=conting["N by level"], value=Ptheo)
    st.write("Chi2 = ",khi2)
    st.write('Theoretical expectations are supposed not to be estimated from data => degrees of freedom')
    st.write("Degrees of Freedom = ",len(Ptheo))
    st.write("p-value = {:0.3f}".format(pvalue))
    st.write('As p-value is below 5%, the proportions of "No"  in the different levels are not consistent ',
             'with theoretical expectations')     
    st.subheader('Effect Size :')
    st.write("Proposed effect size is Cramer's W equal to ")
    st.latex(r'W = \sqrt{\sum_{i=1}^{k}\frac{(P^o_i-P^e_i)^2}{P^e_i}}\newline where\ P^o_i\ are\ observed\ proportions\ \newline and \ P^e_i\ are\ expected\ proportions')
    st.latex(r'corresponding\ to\ W^2 = \frac{\chi^2}{k} ')
    st.write("Cramer's W criteria is 0.1 for small ; 0.3 for medium ; 0.5 for large")
    
    CramerW=np.sqrt(khi2/conting.shape[0])
    st.text('In example, effect size is {:.2f} corresponding to a large effect size.'.format(CramerW))
    st.subheader('<<< See Khi2_proportion_several() function >>>\n\n\n')
    
###########################################################################################
# done done    
def Tstudent_vs_Theoretical_Means(df):
    st.title('T-Student test : sample mean versus theory')
    st.subheader('For 1 QT variable and 1 theoretical expected mean.')
    st.write('H0 : The population mean is equal to a given therory')
    st.text('Example : Is the mean value of corn "Height" in the "East" "Plot" equal to 265 ? ')
    obs1=df[df['Plot']=='East'].Height
    obs1=obs1.dropna()  
    col0,col1,col2,col3=st.beta_columns([0.2,0.8,0.6,0.2])
    with col1:
        st.write("Observed data :\n")
        st.dataframe(obs1,height=200)
        st.write('mean observed value : ',obs1.mean())    
    with col2:
        fig=plt.figure()
        plt.hist(obs1,bins=20)
        plt.xlabel("Height")
        plt.ylabel("Count")
        plt.title("In East Plot")
        st.pyplot(fig)
    with st.echo():
        from scipy.stats import ttest_1samp
        Mtheo=265 # Hypothesis
        # INPUTS : 
        #    - a :           observed values
        #    - popmean :     expected population mean
        #    - nan_policy :  how to treat nan values
        # OUTPUTS :
        #    - statistic :   t observed value
        #    - pvalue    :   p-value
        # note : alternative option not working with my library
        statistic, pvalue=ttest_1samp(a=obs1,popmean=Mtheo,nan_policy="omit")
    st.write('statistic t-value : ',statistic)
    st.write(' p-value : ',pvalue)
    st.text("In this example, and based on p-value, we cannot reject the hypothesis that the mean corn height in east plot is equal to 265, with a confidence level of 95%")
    st.subheader("Effet Size :")
    st.write("it is proposed to use Cohen's d to quantify the effect size")
    st.write("The unbiased standard deviation is given by :")
    st.latex(r"s=\sqrt{\frac{\displaystyle\sum_{i=1}^{n}(x_i-\overline{x})²}{n-1}}\ =\ \sqrt{\frac{\displaystyle\sum_{i=1}^{n}(x_i)²-\overline{x}²}{n-1}} \newline where\ n\ is\ the\ number\ of\ observations, \newline\ x_i\ the\ observations \newline\ and\ \overline{x}\ is\ the\ mean\ of\ the\ observations")
    st.write("And Cohen's d parameter is :")    
    st.latex(r"d=\frac{\overline{x}-M_{theo}}{s}\newline where\ Mtheo\ is\ the\ theoretical\ mean\ expectation.")
    st.write("The cohen's d interpretation is : 0.2 for small effect, 0.55 for medium effect and 0.8 for large effect.")
    with st.echo():
        sigma=obs1.std(ddof=1)
        # ddof = 1 to compute the unbiased standard deviation
    st.text("For the example, the unbiased standard deviation equals : {:0.2f}".format(sigma))
    with st.echo():
        CohenD=(obs1.mean()-Mtheo)/sigma
    st.text("and thus the Cohen's d coefficient is : {:0.2f} corresponding to a small effect.".format(CohenD))
    st.write('\n\n\n\n')
    st.subheader ('Z-test for large samples')
    st.write('As far as the sample size is higher than 30, a Z-test can also be performed')
    with st.echo():
        from scipy.stats import norm
        Z=(obs1.mean()-Mtheo)/(obs1.std()/obs1.shape[0]**0.5)
        st.write('Observed Z statistic value : ',Z)
        # SF = 1 - CDF ; survival function
        pvalue = norm.sf(Z) # one-tail
        st.write('p-value : ',pvalue)
    st.write('t-test has however to be preferred in all cases')
    st.write('Z test was used in the past when using written tables.')
    st.subheader('<<< See Tstudent_vs_Theoretical_Means() function >>>\n\n\n')
    
###########################################################################################
# done done
def Two_Samples_Tstudent_Same_Variances_Means(df) :
    st.title('T-Student test : sample mean comparison between two samples, assuming same variance in population')
    st.subheader('For 1 QT variable and 1 Q2 variable to stratify both categories.')
    st.subheader('or 2 QT variables => to be considered in "by dependent variables !!!!!!"')
    st.write('H0 : The two population means are equal')
    st.text('Example : Are the mean values of corn "Height" equal in the "North" and "South" Plots ? ')
    col0,col1,col2,col3,col4=st.beta_columns([0.2,1.,0.5,1.,0.2]) 
    with col1 :
        obs1=df[df['Plot']=='North'].Height
        obs1=obs1.dropna()  
        #st.write("Observed data for 'North':\n")
        #st.dataframe(obs1,height=200)
        fig=plt.figure()
        plt.hist(obs1,bins=20,color='green')
        plt.xlabel("Height")
        plt.ylabel("Count")
        plt.title("In North Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'North' : ",obs1.shape[0])
        st.write("Sample mean observed value for 'North' : {:0.2f}".format(obs1.mean()))
        st.write("Sample standard deviation for 'North' : {:0.2f}".format(obs1.std(ddof=0)))
    with col3:
        obs2=df[df['Plot']=='South'].Height
        obs2=obs2.dropna()  
        #st.write("Observed data for 'South':\n")
        #st.dataframe(obs2,height=200)
        fig=plt.figure()
        plt.hist(obs2,bins=20,color='blue')
        plt.xlabel("Height")
        plt.ylabel("Count")
        plt.title("In South Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'South' : ",obs2.shape[0])
        st.write("Sample mean observed value for 'South': {:0.2f}".format(obs2.mean()))
        st.write("sample standard deviation for 'North' : {:0.2f}".format(obs2.std(ddof=0)))
    with st.echo():
        from scipy.stats import ttest_ind
        Mtheo=265 # Hypothesis
        # INPUTS : 
        #    - a :           observed values 1
        #    - b :           observed values 2
        #    - equal_var :   boolean
        #                    True if equal variances is supposed, else False
        #    - nan_policy :  how to treat nan values
        # OUTPUTS :
        #    - statistic :   t observed value
        #    - pvalue    :   p-value
        # note : alternative option not working with my library
        statistic, pvalue=ttest_ind(a=obs1,b=obs2,equal_var=True,nan_policy="omit")
    st.write('statistic t-value : {:0.2f}'.format(statistic))
    st.write(' p-value : {:0.6f}'.format(pvalue))
    st.text('Based on p-value, it is not possible to reject the hypothesis of an equal mean in the two populations, with a confidence level of 95%.')
    st.subheader("Effect Size :")
    st.write("As effect size, it is proposed to use the Hedge's g coefficient with a pooled standard deviation")
    st.write("The unbiased standard deviation for both populations is given by :")
    st.latex(r"s_k=\sqrt{\frac{\displaystyle\sum_{i=1}^{n_k}(x_i^k-\overline{x_k})²}{n_k-1}}\ =\ \sqrt{\frac{\displaystyle\sum_{i=1}^{n_k}(x_i^k)²-(\overline{x_k})²}{n_k-1}} \newline where\ k=1,2\ represents\ both\ samples\newline\ n_k\ the\ number\ of\ observations, \newline\ x_i^k\ the\ observations\ \newline\ and\ \overline{x_k}\ the\ means\ of\ the\ observations")
    with st.echo():
        s1=obs1.std(ddof=1)
        s2=obs2.std(ddof=2)
    st.write('Unbiased standard deviations : s1={:0.2f} and s2={:0.2f}'.format(s1,s2))
        # ddof=1 for unbiased standard deviation estimates
    st.write('The pooled standard deviation is given by (variance supposed to have close values, n1 and n2 potentially different) :')
    st.latex(r'\hat{s}_{pooled}\ =\ \sqrt{\frac{(n_1-1)s_1²+(n_2-1)s_2²}{n_1+n_2-2}}')
    with st.echo():
        n1=obs1.shape[0]
        n2=obs2.shape[0]
        spooled=np.sqrt(((n1-1)*s1*s1+(n2-1)*s2*s2)/(n1+n2-2))
    st.write('Pooled standard deviations : s_pooled={:0.2f} '.format(spooled))
    st.write("And Hedge's g parameter is :")    
    st.latex(r"g=\frac{|\overline{x_1}-\overline{x_2}|}{\hat{s}_{pooled}}\newline")
    st.write("The Hedge's g interpretation is : 0.2 for small effect, 0.5 for medium effect and 0.8 for large effect.")
    with st.echo():
        HedgeG=(np.abs(obs1.mean()-obs2.mean()))/spooled
    st.text("For our example, the Hedge's g coefficient is {:0.2f} corresponding to a medium effect (even if H0 cannot be rejected).".format(HedgeG))
    st.subheader('<<< See Two_Samples_Tstudent_Same_Variances_Means function >>>\n\n\n')
    
    
###########################################################################################   
# done done
def Two_Samples_Tstudent_Different_Variances_Means(df) :
    st.title('T-Student test : sample mean comparison between two samples, assuming different variances in population')
    st.subheader('For 1 QT variable and 1 Q2 variable to stratify both categories.')
    st.subheader('or 2 QT variables => to be considered in "by dependent variables !!!!!!"')
    st.write('H0 : The two population means are equal')
    st.text('Example : Are the mean values of corn "Mass" equal in the "North" and "South" Plots ? ')
    col0,col1,col2,col3,col4=st.beta_columns([0.2,1.,0.5,1.,0.2]) 
    with col1 :
        obs1=df[df['Plot']=='North'].Mass
        obs1=obs1.dropna()  
        fig=plt.figure()
        plt.hist(obs1,bins=20,color='green')
        plt.xlabel("Mass")
        plt.ylabel("Count")
        plt.title("In North Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'North' : ",obs1.shape[0])
        st.write("Sample mean observed value for 'North' : {:0.2f}".format(obs1.mean()))
        st.write("Sample standard deviation for 'North' : {:0.2f}".format(obs1.std(ddof=0)))
    with col3:
        obs2=df[df['Plot']=='South'].Mass
        obs2=obs2.dropna()  
        fig=plt.figure()
        plt.hist(obs2,bins=20,color='blue')
        plt.xlabel("Mass")
        plt.ylabel("Count")
        plt.title("In South Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'South' : ",obs2.shape[0])
        st.write("Sample mean observed value for 'South': {:0.2f}".format(obs2.mean()))
        st.write("sample standard deviation for 'North' : {:0.2f}".format(obs2.std(ddof=0)))
    with st.echo():
        from scipy.stats import ttest_ind
        Mtheo=265 # Hypothesis
        # INPUTS : 
        #    - a :           observed values 1
        #    - b :           observed values 2
        #    - equal_var :   boolean
        #                    True if equal variances is supposed, else False
        #    - nan_policy :  how to treat nan values
        # OUTPUTS :
        #    - statistic :   t observed value
        #    - pvalue    :   p-value
        # note : alternative option not working with my library
        statistic, pvalue=ttest_ind(a=obs1,b=obs2,equal_var=False,nan_policy="omit")
    st.write('statistic t-value : {:0.4f}'.format(statistic))
    st.write(' p-value : {:0.6f}'.format(pvalue))
    st.text('Based on p-value, it is not possible to reject the hypothesis of an equal mean in the two populations, with a confidence level of 95%.')
    st.subheader("Effect Size :")
    st.write("As effect size, it is proposed to use the Hedge's g coefficient with a pooled standard deviation")
    st.write("The unbiased standard deviation for both populations is given by :")
    st.latex(r"s_k=\sqrt{\frac{\displaystyle\sum_{i=1}^{n_k}(x_i^k-\overline{x_k})²}{n_k-1}}\ =\ \sqrt{\frac{\displaystyle\sum_{i=1}^{n_k}(x_i^k)²-(\overline{x_k})²}{n_k-1}} \newline where\ k=1,2\ represents\ both\ samples\newline\ n_k\ the\ number\ of\ observations, \newline\ x_i^k\ the\ observations\ \newline\ and\ \overline{x_k}\ the\ means\ of\ the\ observations")
    with st.echo():
        s1=obs1.std(ddof=1)
        s2=obs2.std(ddof=2)
    st.write('Unbiased standard deviations : s1={:0.2f} and s2={:0.2f}'.format(s1,s2))
        # ddof=1 for unbiased standard deviation estimates
    st.write('The pooled standard deviation is given by (variance supposed to have close values, n1 and n2 potentially different) :')
    st.latex(r'\hat{s}_{pooled}\ =\ \sqrt{\frac{(n_1-1)s_1²+(n_2-1)s_2²}{n_1+n_2-2}}')
    with st.echo():
        n1=obs1.shape[0]
        n2=obs2.shape[0]
        spooled=np.sqrt(((n1-1)*s1*s1+(n2-1)*s2*s2)/(n1+n2-2))
    st.write('Pooled standard deviations : s_pooled={:0.2f} '.format(spooled))
    st.write("And Hedge's g parameter is :")    
    st.latex(r"g=\frac{|\overline{x_1}-\overline{x_2}|}{\hat{s}_{pooled}}\newline")
    st.write("The Hedge's g interpretation is : 0.2 for small effect, 0.5 for medium effect and 0.8 for large effect.")
    with st.echo():
        HedgeG=(np.abs(obs1.mean()-obs2.mean()))/spooled
    st.text("For our example, the Hedge's g coefficient is {:0.2f} corresponding to a medium effect (even if H0 cannot be rejected).".format(HedgeG))
    st.subheader("Note : Because of the strong difference between the two sample variances, it is probably best to use Glass's Delta coefficient as an effect size (not presented here).")
    st.subheader('<<< See Two_Samples_Tstudent_Different_Variances_Means function >>>\n\n\n')

###########################################################################################
# done done    
def Two_Paired_Samples_Tstudent_Means(df) :        
    st.title('T-Student test : sample mean comparison between two distributions corresponding to paired (not independent) measurements.')
    st.subheader('For 1 QT variable and 1 QA variable (i.e. 2 QT variables that are paired)')
    st.write('This test consists to perform a one-sample t-test \
            on observed differences between both paired measurements, with a theoretical mean of 0')
    st.write('H0 : The two population means are equal')
    st.text('Example : Are the mean values of corn "Height" (measured at day=D) and "Height.D7" (measured at Day= D+7) equal in the "North" plot ? ')
    col0,col1,col2,col3,col4,col5,col6=st.beta_columns([0.2,1.,0.2,1.,0.2,1.,0.2]) 
    obs1=df[df['Plot']=='North'].Height
    obs1=obs1.dropna()  
    obs2=df[df['Plot']=='North']
    obs2=obs2["Height.D7"]
    obs2=obs2.dropna()  
    for i in obs2.index.tolist():
        if i not in obs1.index.tolist():
            obs2.drop([i],inplace=True)
    with col1 :
        fig=plt.figure()
        plt.hist(obs1,bins=20,color='green')
        plt.xlabel("Height at Day=D")
        plt.ylabel("Count")
        plt.title("In North Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'Height' : ",obs1.shape[0])
        st.write("Sample mean observed value for 'Height' : {:0.2f}".format(obs1.mean()))
        st.write("Sample standard deviation for 'Height' : {:0.2f}".format(obs1.std(ddof=0)))
    with col3:
        fig=plt.figure()
        plt.hist(obs2,bins=20,color='blue')
        plt.xlabel("Height at Day=D+7")
        plt.ylabel("Count")
        plt.title("In North Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'Height.D7' : ",obs2.shape[0])
        st.write("Sample mean observed value for 'Height.D7': {:0.2f}".format(obs2.mean()))
        st.write("sample standard deviation for 'Height.D7' : {:0.2f}".format(obs2.std(ddof=0))) 
    with col5:
        fig=plt.figure()
        plt.scatter(obs1,obs2) 
        plt.xlabel("Height at Day=D")
        plt.ylabel("Height at Day=D+7")
        plt.title("In North Plot")
        st.pyplot(fig)      
    with st.echo():
        from scipy.stats import ttest_1samp
        diff=obs1-obs2
        # INPUTS : 
        #    - a :           observed values
        #    - popmean :     expected population mean
        #    - nan_policy :  how to treat nan values
        # OUTPUTS :
        #    - statistic :   t observed value
        #    - pvalue    :   p-value
        # note : alternative option not working with my library
        statistic, pvalue=ttest_1samp(a=diff,popmean=0,nan_policy="omit")
    st.write('statistic t-value : ',statistic)
    st.write(' p-value : ',pvalue)  
    st.write('Based on p-value, the hypothesis of an equal mean for both measurements cannot be rejected with a confidence level of 95%.')
    st.subheader("Effet Size :")
    st.write("it is proposed to use Cohen's d to quantify the effect size")
    st.write("The unbiased standard deviation is given by :")
    st.latex(r"s=\sqrt{\frac{\displaystyle\sum_{i=1}^{n}(d_i-\overline{d})²}{n-1}}\ =\ \sqrt{\frac{\displaystyle\sum_{i=1}^{n}(d_i)²-\overline{d}²}{n-1}} \newline where\ n\ is\ the\ number\ of\ paired\ measurements, \newline\ d_i\ the\ difference\ between\ paired\ measurements \newline\ and\ \overline{d}\ is\ the\ mean\ of\ this\ difference")
    st.write("And Cohen's d parameter is :")    
    st.latex(r"d=\frac{\overline{d}-0}{s}\newline where\ 0\ is\ the\ theoretical\ mean\ expectation\ for\ the\ difference.")
    st.write("The cohen's d interpretation is : 0.2 for small effect, 0.55 for medium effect and 0.8 for large effect.")
    with st.echo():
        obs=obs1-obs2
        sigma=obs.std(ddof=1)
        # ddof = 1 to compute the unbiased standard deviation
    st.text("For the example, the unbiased standard deviation for the difference equals : {:0.2f}".format(sigma))
    with st.echo():
        CohenD=(obs.mean()-0)/sigma
    st.text("and thus the Cohen's d coefficient is : {:0.2f} corresponding to a small effect.".format(CohenD))
    st.write('\n\n\n\n')
    st.subheader('<<< See Two_Paired_Samples_Tstudent_Means() function >>>\n\n\n')
    
###########################################################################################
# done done 

def One_Factor_Anova_Same_Variances_Means(df) : 
    st.title('One Factor ANOVA test : mean comparison between more than 2 independent distributions, with supposed same variances.')
    st.subheader('For 1 QT variable and 1 Qk variable to stratify the data')
    st.write('H0 : The means for each populations are all equal')
    st.text('Example : Are the mean values of corn "Height" equal in the 4 plots (N,S,E,O) ? ')
    obs=df[['Height','Plot']]
    obs=obs.dropna()  
    st.write("Observed data for 'Height' in the different plots:\n")
    #st.dataframe(obs,height=200)
    obs1=obs[obs['Plot']=='East'].Height
    obs2=obs[obs['Plot']=='North'].Height
    obs3=obs[obs['Plot']=='West'].Height
    obs4=obs[obs['Plot']=='South'].Height
    col0,col1,col2,col3,col4=st.beta_columns([0.5,1.,0.5,1.,0.5]) 
    with col1 :
        fig=plt.figure()
        plt.hist(obs1,bins=9,color='green',rwidth=0.9)
        plt.xlabel("Height")
        plt.ylabel("Count")
        plt.title("In East Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'East' : ",obs1.shape[0])
        st.write("Sample mean observed value for 'East' : {:0.2f}".format(obs1.mean()))
        st.write("Sample standard deviation for 'East' : {:0.2f}".format(obs1.std(ddof=0)))
    with col3:
        fig=plt.figure()
        plt.hist(obs2,bins=6,color='blue',rwidth=0.9)
        plt.xlabel("Height")
        plt.ylabel("Count")
        plt.title("In North Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'North' : ",obs2.shape[0])
        st.write("Sample mean observed value for 'North': {:0.2f}".format(obs2.mean()))
        st.write("sample standard deviation for 'North' : {:0.2f}".format(obs2.std(ddof=0)))     
    with col1 :
        fig=plt.figure()
        plt.hist(obs3,bins=6,color='orange',rwidth=0.9)
        plt.xlabel("Height")
        plt.ylabel("Count")
        plt.title("In West Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'West' : ",obs3.shape[0])
        st.write("Sample mean observed value for 'West' : {:0.2f}".format(obs3.mean()))
        st.write("Sample standard deviation for 'West' : {:0.2f}".format(obs3.std(ddof=0)))
    with col3:
        fig=plt.figure()
        plt.hist(obs4,bins=6,color='purple',rwidth=0.9)
        plt.xlabel("Height")
        plt.ylabel("Count")
        plt.title("In South Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'South' : ",obs4.shape[0])
        st.write("Sample mean observed value for 'South': {:0.2f}".format(obs4.mean()))
        st.write("sample standard deviation for 'South' : {:0.2f}".format(obs4.std(ddof=0)))     
    with st.echo():
        from scipy.stats import f_oneway
        # INPUTS :
        #     - sample1, sample2, sample3... : input arrays
        # OUTPUTS :
        #     - Statistic :   F value
        #     - pvalue :      p-value
        statistic,pvalue=f_oneway(obs1,obs2,obs3,obs4)
    st.write('statistic F-value : ',statistic)
    st.write(' p-value : ',pvalue) 
    st.text('For this example, the hypothesis of an equal height for the four plots can be rejected with a confidence level higher than 95%.')
    st.subheader('Effect Size :')
    st.write('The effect size can be estimated by partial eta² parameter')
    st.write('The partial eta² parameter is : ')
    st.latex(r"\eta^2\ =\ \frac{SSbetween}{SSbetween+SSwithin}\newline where\ SSbetween\ is\ the\ 'between'\ sum\ of\ squares\newline and\ SSwithin\ is\ the\ 'within'\ sum\ of\ squares")
    st.write('See a ANOVA lecture for more description')
    st.write('The partial eta² parameter can be written as a function of the F statistic  as follows : ')
    st.latex(r'\eta^2\ =\ \frac{F.(k-1)}{F.(k-1)+(N-k)}\newline where\ N\ is\ the\ number\ of\ observations\newline and\ k\ is\ the\ number\ of\ classes.')
    st.text('(exact formula still to be confirmed)')
    st.write('The eta² can be interpreted as :  0.001 for small effect, 0.06 for medium effect and 0.14 for large effect.')
    with st.echo():
        eta2=(statistic*(4-1))/(statistic*(4-1)+(obs.shape[0]-4)) # 4 classes
    st.text("For this example, eta² is {:0.2f} corresponding to a very large effect.".format(eta2) )
    st.write("Other effect size can be used like Cohen's f or omega² parameters (not presented here).")
    st.subheader('<<< See One_Factor_Anova_Same_Variances_Means() function >>>\n\n\n')
    
###########################################################################################
# done done
def One_Factor_Anova_Different_Variances_Means(df) :      
    st.title('One Factor ANOVA test : mean comparison between more than 2 independent distributions, with different variances.')
    st.subheader('For 1 QT variable and 1 Qk variable to stratify the data')
    st.write('H0 : The means for each populations are all equal')
    st.text('Example : Are the mean values of corn "Mass" equal in the 4 plots (N,S,E,O) ? ')
    obs=df[['Mass','Plot']]
    obs=obs.dropna()  
    st.write("Observed data for 'Mass' in the different plots:\n")
    obs1=obs[obs['Plot']=='East'].Mass
    obs2=obs[obs['Plot']=='North'].Mass
    obs3=obs[obs['Plot']=='West'].Mass
    obs4=obs[obs['Plot']=='South'].Mass
    col0,col1,col2,col3,col4=st.beta_columns([0.5,1.,0.5,1.,0.5]) 
    with col1 :
        fig=plt.figure()
        plt.hist(obs1,bins=9,color='green',rwidth=0.9)
        plt.xlabel("Mass")
        plt.ylabel("Count")
        plt.title("In East Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'East' : ",obs1.shape[0])
        st.write("Sample mean observed value for 'East' : {:0.2f}".format(obs1.mean()))
        st.write("Sample standard deviation for 'East' : {:0.2f}".format(obs1.std(ddof=0)))
    with col3:
        fig=plt.figure()
        plt.hist(obs2,bins=6,color='blue',rwidth=0.9)
        plt.xlabel("Mass")
        plt.ylabel("Count")
        plt.title("In North Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'North' : ",obs2.shape[0])
        st.write("Sample mean observed value for 'North': {:0.2f}".format(obs2.mean()))
        st.write("sample standard deviation for 'North' : {:0.2f}".format(obs2.std(ddof=0)))     
    with col1 :
        fig=plt.figure()
        plt.hist(obs3,bins=6,color='orange',rwidth=0.9)
        plt.xlabel("Mass")
        plt.ylabel("Count")
        plt.title("In West Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'West' : ",obs3.shape[0])
        st.write("Sample mean observed value for 'West' : {:0.2f}".format(obs3.mean()))
        st.write("Sample standard deviation for 'West' : {:0.2f}".format(obs3.std(ddof=0)))
    with col3:
        fig=plt.figure()
        plt.hist(obs4,bins=6,color='purple',rwidth=0.9)
        plt.xlabel("Mass")
        plt.ylabel("Count")
        plt.title("In South Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'South' : ",obs4.shape[0])
        st.write("Sample mean observed value for 'South': {:0.2f}".format(obs4.mean()))
        st.write("sample standard deviation for 'South' : {:0.2f}".format(obs4.std(ddof=0)))     
    with st.echo():
        from statsmodels.stats.oneway import anova_oneway
        data=[obs1,obs2,obs3,obs4]
        # INPUTS :
        #     - data :             input data (tuple of array, dataframe...)
        #     - use_var :          assumptions regarding variance, expecially :
        #                              unequal : for unequal variances
        #                              equal : for equal variances
        #     - welch_correction : boolean 
        # OUTPUTS :
        #     - Statistic :   F value
        #     - pvalue :      p-value
        statistic,pvalue=anova_oneway(data=data,use_var='unequal',welch_correction=True)
    st.write('statistic F-value : ',statistic)
    st.write(' p-value : ',pvalue) 
    st.text('For this example, the hypothesis of an equal Mass for the four plots can be rejected with a confidence level higher than 95%.')
    st.subheader('Effect Size :')
    st.write('The effect size can be estimated by partial eta² parameter')
    st.write('The partial eta² parameter is : ')
    st.latex(r"\eta^2\ =\ \frac{SSbetween}{SSbetween+SSwithin}\newline where\ SSbetween\ is\ the\ 'between'\ sum\ of\ squares\newline and\ SSwithin\ is\ the\ 'within'\ sum\ of\ squares")
    st.write('See a ANOVA lecture for more description')
    st.write('The partial eta² parameter can be written as a function of the F statistic  as follows : ')
    st.latex(r'\eta^2\ =\ \frac{F.(k-1)}{F.(k-1)+(N-k)}\newline where\ N\ is\ the\ number\ of\ observations\newline and\ k\ is\ the\ number\ of\ classes.')
    st.text('(exact formula still to be confirmed)')
    st.write('The eta² can be interpreted as :  0.001 for small effect, 0.06 for medium effect and 0.14 for large effect.')
    with st.echo():
        eta2=(statistic*(4-1))/(statistic*(4-1)+(obs.shape[0]-4)) # 4 classes
    st.text("For this example, eta² is {:0.2f} corresponding to a very large effect.".format(eta2) )
    st.write("Other effect size can be used like Cohen's f or omega² parameters (not presented here).")
    st.subheader('<<< See One_Factor_Anova_Different_Variances_Means() function >>>\n\n\n')
    
###########################################################################################
#done done 
def Median_vs_Theory_Wilcoxon_Sign(df):
    st.title('Wilcoxon Sign Test : median comparison between one sample and a theory (non parametric test).')
    st.subheader('For 1 QT variable and 1 theoretical data')
    st.write('H0 : The median for the population is as expected by theory')
    st.text('Example : Is the median "Mass.grains" values of corn equal to 80 in "South" plot ? ')
    col1,col2,col3,col4,col5=st.beta_columns([0.2,0.8,0.2,0.8,0.2])
    with col2:
        obs=df[df['Plot']=='South']
        obs=obs[['Mass.grains']]
        obs=obs.dropna()  
    #st.write("Observed data for 'Mass.grains':\n")
    #st.write(obs)
        st.write("Median value : ",obs.median())
    with col4:
        fig=plt.figure()
        plt.hist(obs,bins=6)
        plt.xlabel("Mass.grains")
        plt.ylabel("Count")
        plt.title("In South Plot")
        st.pyplot(fig)   
    Me_theo=80
    Theo=np.array([Me_theo]*obs.shape[0])
    #st.write(Theo)
    with st.echo():
        from scipy.stats import wilcoxon
        import scipy
        #st.write(scipy.__version__)
        # INPUTS :
        #     - x :            first observation
        #     - y :            second observation (a constant vector of value Me_theo)
        #     - zero_method :  management of zero differences : “pratt”, “wilcox”, “zsplit”
        #     - correction :   boolean ; for continuity correction
        #     - alternative :  alternative method H1
        #     - mode :         pvalue calculation method : “exact”, “approx”
        # OUTPUTS :
        #     - Statistic :    F value
        #     - pvalue :       p-value     
        statistic,pvalue=wilcoxon(x=obs["Mass.grains"],y=Theo,zero_method='wilcox',
                                  correction=True,alternative="two-sided",mode='exact') 
    st.write('statistic : ',statistic)
    st.write(' p-value : ',pvalue)     
    st.write('Based on p-value, we cannot reject the hypothesis that the mean value of "Mass.grains" for corns in "South" plot is equal to 80, with a confidence level of 95%')
    st.text('To be analyzed why the statistic value is different from reference [1] (66 isntead of 39), but p-value is exactly the same.\nI assume it depends on whether the positive or negative rank sum is used for the statistic calculation.')
    st.subheader('Effect Size')
    st.write('To be completed')
    st.text('It seems that rank biserial correlation may be one solution...')
    st.subheader('<<< See Median_vs_Theory_Wilcoxon_Sign() function >>>\n\n\n')

###########################################################################################
# done done
def Two_Independent_Samples_Mann_Whitney_Median(df):
    st.title('Mann Whitney Wilcoxon : sample median comparison between two samples (non-parametric test) :')
    st.subheader('For 1 QT variable and 1 Q2 variable to stratify both categories.')
    st.subheader('or 2 QT variables => to be considered in "by dependent variables !!!!!!"')
    st.write('H0 : The two population medians are equal')
    st.text('Example : Are the median values of corn "Mass.grains" equal in the "North" and "South" Plots ? ')
    col0,col1,col2,col3,col4=st.beta_columns([0.2,1.,0.5,1.,0.2]) 
    with col1 :
        obs1=df[df['Plot']=='North']
        obs1=obs1['Mass.grains']
        obs1=obs1.dropna()  
        #st.write("Observed data for 'North':\n")
        #st.dataframe(obs1,height=200)
        fig=plt.figure()
        plt.hist(obs1,bins=20,color='green')
        plt.xlabel("Mass.grains")
        plt.ylabel("Count")
        plt.title("In North Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'North' : ",obs1.shape[0])
        st.write("Sample median observed value for 'North' : {:0.2f}".format(obs1.median()))
    with col3:
        obs2=df[df['Plot']=='South']
        obs2=obs2['Mass.grains']
        obs2=obs2.dropna()  
        #st.write("Observed data for 'South':\n")
        #st.dataframe(obs2,height=200)
        fig=plt.figure()
        plt.hist(obs2,bins=20,color='blue')
        plt.xlabel("Mass.grains")
        plt.ylabel("Count")
        plt.title("In South Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'South' : ",obs2.shape[0])
        st.write("Sample median observed value for 'South': {:0.2f}".format(obs2.median()))
    with st.echo():
        from scipy.stats import mannwhitneyu
        # INPUTS :
        #     - x :                first observation
        #     - y :                second observation (a constant vector of value Me_theo)
        #     - use_continuity :   boolean ; for continuity correction
        #     - alternative :      alternative method H1
        # OUTPUTS :
        #     - Statistic :        F value
        #     - pvalue :           p-value 
        statistic,pvalue=mannwhitneyu(x=obs1,y=obs2,
                                  use_continuity=True,alternative="two-sided")
    st.write('statistic : ',statistic)
    st.write(' p-value : ',pvalue)      
    st.text('On our example, the hypothesis of equal median values cannot be rejected, with a confidence level of 95%')
    st.subheader('Effect Size :')
    st.write('to be completed')
    st.subheader('<<< See Two_Independent_Samples_Mann_Whitney_Median() function >>>\n\n\n')    
    
###########################################################################################
# done done
def Two_Paired_Samples_Wilcoxon_Sign_Median(df):
    st.title('Wilcoson Test for sample median comparison between two paired (not independent) measurements.')
    st.subheader('For 1 QT variable and 1 QA variable (i.e. 2 QT variables that are paired)')
    st.write('This test consists to perform a wilcoxon sign test \
            on observed differences between both paired measurements, with a theoretical median of 0')
    st.write('H0 : The two population means are equal')
    st.text('Example : Are the mean values of corn "Height" (measured at day=D) and "Height.D7" (measured at Day= D+7) equal in the "South" plot ? ')
    col0,col1,col2,col3,col4,col5,col6=st.beta_columns([0.2,1.,0.2,1.,0.2,1.,0.2]) 
    obs1=df[df['Plot']=='South'].Height
    obs1=obs1.dropna()  
    obs2=df[df['Plot']=='South']
    obs2=obs2["Height.D7"]
    obs2=obs2.dropna()  
    for i in obs2.index.tolist():
        if i not in obs1.index.tolist():
            obs2.drop([i],inplace=True)
    with col1 :
        fig=plt.figure()
        plt.hist(obs1,bins=20,color='green')
        plt.xlabel("Height at Day=D")
        plt.ylabel("Count")
        plt.title("In South Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'Height' : ",obs1.shape[0])
        st.write("Sample median observed value for 'Height' : {:0.2f}".format(obs1.median()))
    with col3:
        fig=plt.figure()
        plt.hist(obs2,bins=20,color='blue')
        plt.xlabel("Height at Day=D+7")
        plt.ylabel("Count")
        plt.title("In South Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'Height.D7' : ",obs2.shape[0])
        st.write("Sample median observed value for 'Height.D7': {:0.2f}".format(obs2.median()))
    with col5:
        fig=plt.figure()
        plt.scatter(obs1,obs2) 
        plt.xlabel("Height at Day=D")
        plt.ylabel("Height at Day=D+7")
        plt.title("In South Plot")
        st.pyplot(fig)
    with st.echo():
        from scipy.stats import wilcoxon
        import scipy
        #st.write(scipy.__version__)
        # INPUTS :
        #     - x :            first observation
        #     - y :            second observation (a constant vector of value Me_theo)
        #     - zero_method :  management of zero differences
        #     - correction :   boolean ; for continuity correction
        #     - alternative :  alternative method H1
        #     - mode :         pvalue calculation method
        # OUTPUTS :
        #     - Statistic :    F value
        #     - pvalue :       p-value     
        statistic,pvalue=wilcoxon(x=obs1,y=obs2,zero_method='wilcox',
                                  correction=False,alternative="two-sided",mode='exact')
    st.write('statistic : ',statistic)
    st.write(' p-value : ',pvalue)   
    st.text('Based on this example, the hypothesis of equal median values at days D and D+7 cannot be rejected with a confidence level of 95%')
    st.subheader('Effect Size :')
    st.write('to be completed')
    st.subheader('<<< See Two_Paired_Samples_Wilcoxon_Sign_Median() function >>>\n\n\n')


###########################################################################################
# done done    
def Several_Samples_Kruskal_Wallis_Median(df):
    st.title('Kruskal Wallis : median comparison between more than 2 independent distributions (non-parametric test)')
    st.subheader('For 1 QT variable and 1 Qk variable to stratify the data')
    st.write('H0 : The medians for each populations are all equal')
    st.text('Example : Are the median values of corn "Mass.grains" equal in the 4 plots (N,S,E,O) ? ')
    obs=df[['Mass.grains','Plot']]
    obs=obs.dropna()  
    st.write("Observed data for 'Mass.grains' in the different plots:\n")
    #st.dataframe(obs,height=200)
    obs1=obs[obs['Plot']=='East']
    obs1=obs1['Mass.grains']
    obs2=obs[obs['Plot']=='North']
    obs2=obs2['Mass.grains']
    obs3=obs[obs['Plot']=='West']
    obs3=obs3['Mass.grains']
    obs4=obs[obs['Plot']=='South']
    obs4=obs4['Mass.grains']
    col0,col1,col2,col3,col4=st.beta_columns([0.5,1.,0.5,1.,0.5]) 
    with col1 :
        fig=plt.figure()
        plt.hist(obs1,bins=6,color='green',rwidth=0.9)
        plt.xlabel("Mass.grains")
        plt.ylabel("Count")
        plt.title("In East Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'East' : ",obs1.shape[0])
        st.write("Sample median observed value for 'East' : {:0.2f}".format(obs1.median()))
    with col3:
        fig=plt.figure()
        plt.hist(obs2,bins=6,color='blue',rwidth=0.9)
        plt.xlabel("Mass.grains")
        plt.ylabel("Count")
        plt.title("In North Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'North' : ",obs2.shape[0])
        st.write("Sample median observed value for 'North': {:0.2f}".format(obs2.median()))    
    with col1 :
        fig=plt.figure()
        plt.hist(obs3,bins=6,color='orange',rwidth=0.9)
        plt.xlabel("Mass.grains")
        plt.ylabel("Count")
        plt.title("In West Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'West' : ",obs3.shape[0])
        st.write("Sample median observed value for 'West' : {:0.2f}".format(obs3.median()))
    with col3:
        fig=plt.figure()
        plt.hist(obs4,bins=6,color='purple',rwidth=0.9)
        plt.xlabel("Mass.grains")
        plt.ylabel("Count")
        plt.title("In South Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'South' : ",obs4.shape[0])
        st.write("Sample median observed value for 'South': {:0.2f}".format(obs4.median()))        
    with st.echo():
        from scipy.stats import kruskal
        # INPUTS :
        #     - liste of input arrays as observations
        #     - nan_policy :       policy regarding nan values (propagate’, ‘raise’, ‘omit’)
        # OUTPUTS :
        #     - Statistic :        F value
        #     - pvalue :           p-value 
        statistic,pvalue=kruskal(obs1,obs2,obs3,obs4,nan_policy='omit')
    st.write('statistic : ',statistic)
    st.write(' p-value : ',pvalue)   
    st.write('Degrees of freedom (k-1) : ',3)

    st.write('This test can also be performed as a 50% quantile test, but for "Median" this particular \
            test is often more effective.')
    st.text('For this example, the hypothesis that all medians are equal can be rejected, with a confidence level of moe than 95%')
    st.subheader('Effect Size :')
    st.write('to be completed')
    st.subheader('<<< See Several_Samples_Kruskal_Wallis_Median() function >>>\n\n\n')    
    
    
###########################################################################################
    
def Quantiles(df):
    st.title("Test for quantiles comparison")
    st.write("Currently, I didn't found a 'scipy' or 'statsmodel' method allowing to perform a quantile test \
            comparison between several observations.")
    st.write("The existing test, is the median test corresponding to 50% quantile test.")
    st.write("As detailed in reference [1], the median test method can be generalized to either quantiles.")
    st.write("Potentially a future work to be done.")
    st.write("Below, the test median for information (oten less effective than Kruskal Wallis test)")
    st.write('H0 : The quantiles (median) for each populations are all equal')
    st.text('Example : Are the median values of corn "Mass.grains" equal in the 4 plots (N,S,E,O) ? ')
    obs=df[['Mass.grains','Plot']]
    obs=obs.dropna()      
    obs1=obs[obs['Plot']=='East']
    obs1=obs1['Mass.grains']
    obs2=obs[obs['Plot']=='North']
    obs2=obs2['Mass.grains']
    obs3=obs[obs['Plot']=='West']
    obs3=obs3['Mass.grains']
    obs4=obs[obs['Plot']=='South']
    obs4=obs4['Mass.grains']
    col0,col1,col2,col3,col4=st.beta_columns([0.5,1.,0.5,1.,0.5]) 
    with col1 :
        fig=plt.figure()
        plt.hist(obs1,bins=6,color='green',rwidth=0.9)
        plt.xlabel("Mass.grains")
        plt.ylabel("Count")
        plt.title("In East Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'East' : ",obs1.shape[0])
        st.write("Sample median observed value for 'East' : {:0.2f}".format(obs1.median()))
    with col3:
        fig=plt.figure()
        plt.hist(obs2,bins=6,color='blue',rwidth=0.9)
        plt.xlabel("Mass.grains")
        plt.ylabel("Count")
        plt.title("In North Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'North' : ",obs2.shape[0])
        st.write("Sample median observed value for 'North': {:0.2f}".format(obs2.median()))    
    with col1 :
        fig=plt.figure()
        plt.hist(obs3,bins=6,color='orange',rwidth=0.9)
        plt.xlabel("Mass.grains")
        plt.ylabel("Count")
        plt.title("In West Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'West' : ",obs3.shape[0])
        st.write("Sample median observed value for 'West' : {:0.2f}".format(obs3.median()))
    with col3:
        fig=plt.figure()
        plt.hist(obs4,bins=6,color='purple',rwidth=0.9)
        plt.xlabel("Mass.grains")
        plt.ylabel("Count")
        plt.title("In South Plot")
        st.pyplot(fig)
        st.write("Number of observations for 'South' : ",obs4.shape[0])
        st.write("Sample median observed value for 'South': {:0.2f}".format(obs4.median()))  
    with st.echo():
        from scipy.stats import median_test
        # INPUTS :
        #     - liste of input arrays as observations
        #     - ties :             strategy how values equal to Grand Median are treated (below,abive,ignore)
        #     - correction :       boolean ; whether Yates continuous correction has to be applied
        #     - nan_policy :       policy regarding nan values (propagate’, ‘raise’, ‘omit’)
        # OUTPUTS :
        #     - stat :             F value
        #     - p :                p-value 
        #     - m :                Grand Median
        #     - table :            contingency table 
        statistic,pvalue,m,table=median_test(obs1,obs2,obs3,obs4, ties='ignore',correction=True,
                                  nan_policy='omit')
    st.write('statistic : ',statistic)
    st.write(' p-value : ',pvalue) 
    st.text('Note : to be assessed why the result is slightly different from resut given in [1].')
    st.text('For this example, the assumptions that all medians are equal can be rejected with a confidence level of 95%')
    #st.write('Degrees of freedom (k-1) : ',3)
    st.write('Grand median',m)
    table=pd.DataFrame(data=table.transpose(),index=['a','b','c','d'],columns=['False','True'])
    st.write("Contingency table :")
    st.dataframe(table)
    st.write("""This is the contingency table. The shape of the table is (2, n), where n is the number of samples. 
            The first row holds the counts of the values above the grand median, and the second row holds
            the counts of the values below the grand median. """)
    st.subheader('Effect Size :')
    st.write('to be completed')
    st.subheader('<<< See Quantiles() function >>>\n\n\n') 
    
###########################################################################################

def Two_Samples_Fisher_Snedecor_Parametric_Variances(df):
    st.title("Fisher-Snedecor Parametric Variance Test for 2 observations")
    st.subheader('<<< See Two_Samples_Fisher_Snedecor_Parametric_Variances() function >>>\n\n\n') 
        
    obs1=df[df['Plot']=='North']
    obs1=obs1['Height']
    obs1=obs1.dropna()  
    obs2=df[df['Plot']=='South']
    obs2=obs2['Height']
    obs2=obs2.dropna()     
    st.markdown('<u>obs1 : "Mass.grains" for "Plot"="North" : </u>',unsafe_allow_html=True) 
    st.dataframe(obs1,height=200)
    st.write("Variance : ",obs1.std()**2.)
    st.markdown('<u>obs2 : "Mass.grains" for "Plot"="South" : </u>',unsafe_allow_html=True)    
    st.dataframe(obs2,height=200)
    st.write("Variance : ",obs2.std()**2.)    
    
    with st.echo():
        from scipy.stats import f
        Var1=obs1.std()**2.
        Var2=obs2.std()**2.
        F_stat=Var1/Var2 # test statistic
        # Beware Var1 is numerator and Var2 denominator
        dof1=obs1.shape[0]-1 # degrees of freedom
        dof2=obs2.shape[0]-1
        # INPUTS :
        #     - F statistic
        #     - degree of freedom for observation 1
        #     - degree of freedom for observation 2
        # OUTPUTS :
        #     - p-value
        # Note : f.cdf is cumulative distribution function of Fisher-Snedecor statistic
        # Note : f.sf is survival function (1-cdf) of Fisher-Snedecor statistic
        p_value1 = f.sf(F_stat, dof1, dof2) 
        p_value2 = f.cdf(F_stat, dof1, dof2) 
        p_value = 2*min(p_value1,p_value2) # for Bilateral test        
        # Beware : first dof to be input is dof1, then dof2 ==> to be consistent with F definition
    st.write('Degrees of freedom :',dof1,dof2)
    st.write('statistic F : ',F_stat)
    st.write('p-value : ',p_value)   
       
###########################################################################################

def Two_Samples_Ansary_Bradley_Non_Parametric_Variances(df):
    st.title("Ansary_Bradley non Parametric Variance Test for two observations")
    st.subheader('<<< See Two_Samples_Ansary_Bradley_Non_Parametric_Variances() function >>>\n\n\n') 
    
    obs1=df[df['Plot']=='East']
    obs1=obs1['Mass.grains']
    obs1=obs1.dropna()  
    obs2=df[df['Plot']=='West']
    obs2=obs2['Mass.grains']
    obs2=obs2.dropna()     
    st.markdown('<u>obs1 : "Mass.grains" for "Plot"="West" : </u>',unsafe_allow_html=True) 
    st.dataframe(obs1,height=200)
    st.write("Variance : ",obs1.std()**2.)
    st.markdown('<u>obs2 : "Mass.grains" for "Plot"="East" : </u>',unsafe_allow_html=True)    
    st.dataframe(obs2,height=200)
    st.write("Variance : ",obs2.std()**2.)    
    
    with st.echo():
        from scipy.stats import ansari
        # INPUTS :
        #     - F statistic
        #     - degree of freedom for observation 1
        #     - degree of freedom for observation 2
        # OUTPUTS :
        #     - Statistic
        #     - p-value
        Statistic,p_value = ansari(obs1,obs2) # for Bilateral test        
        # Beware : first dof to be input is dof1, then dof2 ==> to be consistent with F definition
    #st.write('Degrees of freedom :',dof1,dof2)
    st.write('statistic : ',Statistic)
    st.write('p-value : ',p_value)   

########################################################################################### 

def Several_Samples_Bartlett_Parametric_Variances(df):
    st.title("Bartlett Parametric Variance Test for several observations")
    st.subheader('<<< See Several_Samples_Bartlett_Parametric_Variances() function >>>\n\n\n') 
    
    obs1=df[df['Plot']=='East']
    obs1=obs1['Mass']
    obs1=obs1.dropna()  
    obs2=df[df['Plot']=='North']
    obs2=obs2['Mass']
    obs2=obs2.dropna()  
    obs3=df[df['Plot']=='West']
    obs3=obs3['Mass']
    obs3=obs3.dropna()  
    obs4=df[df['Plot']=='South']
    obs4=obs4['Mass']
    obs4=obs4.dropna() 
    st.markdown('<u>obs1 : "Mass" for "Plot"="East" : </u>',unsafe_allow_html=True) 
    st.dataframe(obs1,height=200)
    st.write("Variance : ",obs1.std()**2.)
    st.markdown('<u>obs2 : "Mass" for "Plot"="North" : </u>',unsafe_allow_html=True)    
    st.dataframe(obs2,height=200)
    st.write("Variance : ",obs2.std()**2.)    
    st.markdown('<u>obs1 : "Mass" for "Plot"="West" : </u>',unsafe_allow_html=True) 
    st.dataframe(obs3,height=200)
    st.write("Variance : ",obs3.std()**2.)
    st.markdown('<u>obs2 : "Mass" for "Plot"="South" : </u>',unsafe_allow_html=True)    
    st.dataframe(obs4,height=200)
    st.write("Variance : ",obs4.std()**2.) 
              
    with st.echo():
        from scipy.stats import bartlett
        # INPUTS :
        #     - several successive arrays for each observation
        # OUTPUTS :
        #     - Statistic
        #     - p-value
        Statistic,p_value = bartlett(obs1,obs2,obs3,obs4) 
    st.write('statistic : ',Statistic)
    st.write('p-value : ',p_value)   

###########################################################################################
    
def Several_Samples_Fligner_Killeen_Non_Parametric_Variances(df):
    st.title("Fligner-Killeen Non Parametric Variance Test for several observations")
    st.subheader('<<< See Several_Samples_Fligner_Killeen_Non_Parametric_Variances() function >>>\n\n\n') 
    
    obs1=df[df['Plot']=='East']
    obs1=obs1['Mass.grains']
    obs1=obs1.dropna()  
    obs2=df[df['Plot']=='North']
    obs2=obs2['Mass.grains']
    obs2=obs2.dropna()  
    obs3=df[df['Plot']=='West']
    obs3=obs3['Mass.grains']
    obs3=obs3.dropna()  
    obs4=df[df['Plot']=='South']
    obs4=obs4['Mass.grains']
    obs4=obs4.dropna() 
    st.markdown('<u>obs1 : "Mass.grains" for "Plot"="East" : </u>',unsafe_allow_html=True) 
    st.dataframe(obs1,height=200)
    st.write("Variance : ",obs1.std()**2.)
    st.markdown('<u>obs2 : "Mass.grains" for "Plot"="North" : </u>',unsafe_allow_html=True)    
    st.dataframe(obs2,height=200)
    st.write("Variance : ",obs2.std()**2.)    
    st.markdown('<u>obs1 : "Mass.grains" for "Plot"="West" : </u>',unsafe_allow_html=True) 
    st.dataframe(obs3,height=200)
    st.write("Variance : ",obs3.std()**2.)
    st.markdown('<u>obs2 : "Mass.grains" for "Plot"="South" : </u>',unsafe_allow_html=True)    
    st.dataframe(obs4,height=200)
    st.write("Variance : ",obs4.std()**2.) 
              
    with st.echo():
        from scipy.stats import fligner
        # INPUTS :
        #     - several successive arrays for each observation
        # OUTPUTS :
        #     - Statistic
        #     - p-value
        Statistic,p_value = fligner(obs1,obs2,obs3,obs4) 
              
    st.write('Degree of freedom (k-1) :',3)
    st.write('statistic : ',Statistic)
    st.write('p-value : ',p_value)   

###########################################################################################
    
def Input_Correlation_Text():

    st.markdown('<u> Considerations when chosing a correlation test :</u>',unsafe_allow_html=True)
    st.write("""For the Pearson r correlation, both variables should be normally distributed 
            (normally distributed variables have a bell-shaped curve). 
            Other assumptions include linearity and homoscedasticity.  
            Linearity assumes a straight line relationship between each of the two variables 
            and homoscedasticity assumes that data is equally distributed about the regression line. """)
    
    st.write("""     
            The distribution of Kendall’s tau has better statistical properties.
            The interpretation of Kendall’s tau in terms of the probabilities of observing the agreeable
            (concordant) and non-agreeable (discordant) pairs is very direct.
            In most of the situations, the interpretations of Kendall’s tau and Spearman’s rank 
            correlation coefficient are very similar and thus invariably lead to the same inferences.""")
    st.write("(Informations from www.statisticssolutions.com)")
    
    
###########################################################################################               
    
def Pearson_Linear_Parametric_Correlation(df):
    st.title("Pearson Parametric Linear Correlation")
    st.subheader('<<< See Pearson_Linear_Parametric_Correlation() function >>>\n\n\n') 
    
    st.subheader('H0: both oservations are NOT correlated')

    obs=df[df['Plot']=='East']
    obs=obs[['Height','Mass']]
    obs=obs.dropna()  
    st.dataframe(obs)
    
    st.write('Histogram :')
    
    with st.echo():
        from scipy.stats import pearsonr
        # INPUTS :
        #     - x : first observation
        #     - y : second observation
        # OUTPUTS :
        #     - Statistic r
        #     - p-value
        r,p_value = pearsonr(obs.iloc[:,0],obs.iloc[:,1]) 
    st.write('statistic R (Pearsons correlation coefficient): ',r)
    st.write('p-value : ',p_value) 

###########################################################################################
    
def Spearman_Monotonous__Non_Parametric_Correlation(df):
    st.title("Spearman Non Parametric Monotonous Correlation")
    st.subheader('<<< See Spearman_Monotonous__Non_Parametric_Correlation() function >>>\n\n\n') 
    
    st.subheader('H0: both oservations are NOT correlated')

    obs=df[df['Plot']=='East']
    obs=obs[['Height','Mass.grains']]
    obs=obs.dropna()  
    st.dataframe(obs)
    
    st.write('Histogram :')
    
    with st.echo():
        from scipy.stats import spearmanr
        # INPUTS :
        #     - x :            first observation
        #     - y :            second observation
        #     - nan_policy :   'propagate’, ‘raise’, ‘omit’}
        # OUTPUTS :
        #     - Spearman correlation matrix, or correlation coefficient
        #     - p-value
        r,p_value = spearmanr(obs.iloc[:,0],obs.iloc[:,1],nan_policy='omit') 
    st.write('statistic rho (Spearman correlation coefficient): ',r)
    st.write('p-value : ',p_value) 

###########################################################################################

def Kendall_Monotonous__Non_Parametric_Correlation(df):
    st.title("Kendall Non Parametric Monotonous Correlation")
    st.subheader('<<< See Kendall_Monotonous__Non_Parametric_Correlation() function >>>\n\n\n') 
    
    st.subheader('H0: both oservations are NOT correlated')

    obs=df[df['Plot']=='East']
    obs=obs[['Height','Mass.grains']]
    obs=obs.dropna()  
    st.dataframe(obs)
    
    st.write('Histogram :')
    
    with st.echo():
        from scipy.stats import kendalltau
        # INPUTS :
        #     - x :            first observation
        #     - y :            second observation
        #     - nan_policy :   'propagate’, ‘raise’, ‘omit’}
        #     - method :       method to compute p-value : 'auto’, ‘asymptotic’, ‘exact’
        #     - variant :      which kendall's tau is returned ("b" or "c")
        # OUTPUTS :
        #     - Spearman correlation matrix, or correlation coefficient
        #     - p-value
        r,p_value = kendalltau(obs.iloc[:,1],obs.iloc[:,0],nan_policy='omit',method='exact',variant='b') 
    st.write('statistic tau (kendall correlation coefficient): ',r)
    st.write('p-value : ',p_value) 

###########################################################################################

def Input_Distribution_Text():

    st.markdown('<u> Considerations when chosing a distribution conformity test :</u>',unsafe_allow_html=True)
    st.text("""1/ Tests based on empirical distribution tests : compare data to cumulative distribution function.
They are non parametric test, applicable to any distributions (not only gaussian)
=> Anderson-darling 
=> Kolmogorov-Smirnov =>  seems less powerful than Anderson-Darling test (not proposed in this tool)""")
    st.text("""2/ Tests based on regression and correlation tests :    
=> Shapiro-Wilk
does not work well if many identical values in the distribution
            """)
    st.text("""3/ Tests based on moment (skewness and kurtosis) calculations :
=> D'Agostino-Pearson normality test (K² statistics)
""")
    st.write("""Monte Carlo simulations have found that Shapiro–Wilk has the best power for a given significance, followed closely by Anderson–Darling. (Razali, Nornadiah; Wah, Yap Bee (2011). "Power comparisons of Shapiro–Wilk, Kolmogorov–Smirnov, Lilliefors and Anderson–Darling tests". Journal of Statistical Modeling and Analytics. 2 (1): 21–33. Retrieved 30 March 2017.)    """)
    
    st.text("""Another potential solution is to check all these tests:
- if all tests reject Normality, it can be considered as 'Hard rejection'
- is only some of the tests reject Normality, it can be considered as a 'soft rejection' """)
    st.text("""Two additional comments:
=> a QQ Plot diagram helps to check visually for normality
=> the validity of all these tests is often questionable for small samples""")
    st.subheader("H0 : distribution is as expected")

###########################################################################################

def Anderson_Darling(df):
    st.title("Anderson-Darling")
    st.subheader('<<< See Anderson_Darling() function >>>\n\n\n') 
    
    st.subheader('H0: ................')

    obs=df[df['Plot']=='East']
    obs=obs['Height']
    obs=obs.dropna()  
    st.dataframe(obs)
    
    st.subheader('H0: data is normally distributed')
         
    with st.echo():
        from scipy.stats import anderson
        # INPUTS :
        #     - x :            first observation
        #     - dist :         expected distribution (norm, expon, logistic...)
        # OUTPUTS :
        #     - Anderson-Darling test statistic
        #     - critical values (list)
        #     - significance_level (list)
        stat,crit_value,alpha = anderson(x=obs,dist="norm") 
    st.write('statistic : ',stat)
    st.write('crit-values : ',crit_value) 
    st.write('associted with the following significance level : ',alpha)
    st.write('If the returned statistic is larger than these critical values then for the corresponding significance level, the null hypothesis that the data come from the chosen distribution can be rejected.')
    st.write('At alpha=5%, compare stat=',stat,' with crit_value=',crit_value[2])
    st.write('If stat is higher than crit_value, reject H0 at alpha=5%')
       
###########################################################################################        
    
def Shapiro_Wilk(df):
    st.title("Shapiro-Wilk")
    st.subheader('<<< See Shapiro-Wilk() function >>>\n\n\n') 
    
    st.subheader('H0: data is normally distributed')

    obs=df[df['Plot']=='East']
    obs=obs['Mass.grains']
    obs=obs.dropna()  
    st.dataframe(obs)
    
    st.write('Histogram :')
         
    with st.echo():
        from scipy.stats import shapiro
        # INPUTS :
        #     - x :            first observation
        #     - dist :         expected distribution (norm, expon, logistic...)
        # OUTPUTS :
        #     - test statistic
        #     - p-value
        stat,p_value = shapiro(x=obs) 
    st.write('statistic W : ',stat)
    st.write('p-values : ',p_value) 

###########################################################################################

def Dagostino_Pearson(df):
    st.title("D'Agostino-Pearson")
    st.subheader('<<< See Dagostino_Pearson() function >>>\n\n\n') 
    
    st.subheader('H0: data is normally distributed')

    obs=df[df['Plot']=='East']
    obs=obs['Mass.grains']
    obs=obs.dropna()  
    st.dataframe(obs)
    
    st.write('Histogram :')
         
    with st.echo():
        from scipy.stats import normaltest
        # Test based on d'Agostino and Pearson's test 
        # INPUTS :
        #     - a :            first observation
        #     - nan_policy :   how nan values are treated (propagate, raise, omit)
        # OUTPUTS :
        #     - test statistic (s^2 + k^2) where s and k are z-score returned by skewness and kurtosis
        #     - p-value
        stat,p_value = normaltest(a=obs,nan_policy='omit') 
    st.write('statistic : ',stat)
    st.write('p-values : ',p_value)            

###########################################################################################
    
def References():
    st.subheader("THIS TOOL IS BASED ON FOLLOWING REFERENCES :")
    st.text('\n\n')
    st.text("[1]    Comprendre et réaliser les tests statistiques à l'aide de R - Manuel de Biostatistique - Gael Millot - 3rd Edition - 2016 - Edition deboek Superieur")
    st.text("[2]    Statistical Test Chear Sheet from Dr. Rajeev Pandey, Lucknow university (https://dacg.in/2018/11/17/statistical-test-cheat-sheet/)")
    st.subheader(" And with following Python Module :")
    st.text("""
    numpy-1.20.2
    seaborn-0.11.0
    matplotlib-3.3.2
    streamlit-0.81.1
    pandas-1.1.3
    statsmodels-0.12.0
    scipy-1.6.3
    Pillow-8.2.0
    """)
    st.text('\n\n')
    st.subheader("This tool, the input data and all examples are based on the following book [1] that is a real Bible to me. ")
    st.subheader("Many Thanks to the Author Gaël Millot.")

###########################################################################################
    
def Notations():
        st.subheader("Notations :")
        st.write('Q2  : qualitative variable with 2 independent classes')
        st.write('Qk  : qualitative variable with 2 independent classes or more')
        st.write('QA  : qualitative variable with 2 paired classes (for example repeated measures)')
        st.write('QT  : quantitiave variable (eventually qualitative ordinal variable)')
        st.write('NP  : Non-Parametric test (test that does not suppose normal distributions)')
        st.write('\n\n\n')
        st.subheader('In this tool, all alternatives H1 are considered to be Bilateral')

###########################################################################################    
    
def main():
    st.set_page_config(page_title="Statistic Test Choice Tool", layout="wide") 
    #      page_icon="🧊",
    #      initial_sidebar_state="expanded",
    plt.style.use('ggplot')
    
    
    # reading datafile
    df=pd.read_csv("corn.txt",sep='\t')
    #st.dataframe(df,width=900) 
    #st.write(df.columns.tolist())
    
    st.title("STATISTICAL TEST CHOICE TOOL")
      
    menu=["Make your choice","By type of measure","By Dependent variable", "Other",
          "Notations/Hypothesis", "References"]
    
    choice=st.sidebar.selectbox("MENU",menu)
    st.write('\n\n\n\n\n')
    
    if choice==menu[0]:
        #st.write('MAKE YOUR CHOICE')
        st.text('\n\n\n')
        st.text('\n\n\n')
        st.header("<<<< MAKE YOUR CHOICE ON LEFT SIDEBAR >>>>")
        st.text('\n\n\n')
        st.text('\n\n\n')
        st.write("Feel free to comment this tool (constructively !), to help me to correct or to improve it")
        st.text('\n\n\n')
        st.text('\n\n\n')
        st.text('\n\n\n')
        Notations()
        st.text('\n\n\n')
        st.text('\n\n\n')
        References()
        
    elif choice==menu[1]: # BY MLEASURE
        menu2=["Make your choice","Count","Proportion","Mean","Median","Quantiles"
               ,"Variance","Correlation","Distribution"]
        choice2=st.sidebar.selectbox("TYPE OF MEASURE",menu2)
        
        if choice2==menu2[0]: 
            #st.write('MAKE YOUR CHOICE')
            pass
        elif choice2==menu2[1]: # BY MEASURE / COUNT
            menu3=["Make your choice","Conformity between 1 observation vs Theory",
                   "Homogeneity between 2 observations",
                   "Independance between 2 binary outcomes for matched / stratified data"]
            
            #st.subheader('CHOSEN ANALYSIS')
            choice3=st.selectbox("",menu3) 
            st.write('\n\n\n\n\n')
            
            if choice3==menu3[0]: 
                #st.subheader('MAKE YOUR CHOICE')
                pass
            elif choice3==menu3[1]: 
                Khi2_conformity_count(df)
                
            elif choice3==menu3[2]:
                menu4=["Make your choice","YES","NO"]
                Cochran_Description()
                st.subheader("Is Cochran condition respected ?")
                choice4=st.selectbox("",menu4)
                if choice4==menu4[0]:
                    pass
                elif choice4==menu4[2]:
                    Fisher_test_cxk_Count(df)
                elif choice4==menu4[1]:
                    Khi2_homogeneity_count(df)
                    G_count(df)
                else:
                    st.write('coding error')
                    
            elif choice3==menu3[3]:
                Mantel_Haenszel_Count(df)
            else:
                st.subheader('Coding error')
        
        elif choice2==menu2[2]: # BY MEASURE / PROPORTION
            menu3=["Make your choice","Conformity between 1 observation vs Theory",
                   "Homogeneity between 2 observations",
                   "Between several observations","Between several observations and Theory"]
            
            #st.subheader('CHOSEN ANALYSIS')
            choice3=st.selectbox("",menu3)
            
            st.write('\n\n\n\n\n')
            
            if choice3==menu3[0]: 
                pass
            elif choice3==menu3[1]: 
                menu4=["Make your choice","YES","NO"]
                st.subheader("Binomial ?")
                choice4=st.selectbox("",menu4)
                if choice4==menu4[0]:
                    pass
                elif choice4==menu4[1]:
                    Binomial_vs_Theory_Proportions(df)
                elif choice4==menu4[2]:
                    Khi2_conformity_proportion(df)
                else:
                    st.write('coding error') 
                    
            elif choice3==menu3[2]:
                menu4=["Make your choice","YES","NO"]
                st.subheader("Paired (matched) observations ?")
                choice4=st.selectbox("",menu4)
                if choice4==menu4[0]:
                    pass
                elif choice4==menu4[2]:
                    menu5=["Make your choice","YES","NO"]
                    Cochran_Description()
                    st.subheader("is Cochran condition respected ?")
                    choice5=st.selectbox("",menu5,key=1)
                    if choice5==menu5[0]:
                        pass
                    elif choice5==menu5[1]:
                        Khi2_homogeneity_proportion(df)
                    elif choice5==menu5[2]:
                        Fisher_test_2x2_Proportions(df)
                    else:
                        st.write('coding error')
                elif choice4==menu4[1]:
                    st.write('Binomial test is required for small sample (n < 20)')
                    st.write('MacNemar Python proposed test can also performed the exact binomial test')
                    menu5=["Make your choice","Binomial","MacNemar"]
                    st.subheader("Use of Binomial test Or MacNemar's Test?")
                    choice5=st.selectbox("",menu5,key=2)
                    if choice5==menu5[0]:
                        pass
                    elif choice5==menu5[1]:
                        Binomial_Paired_Proportions(df)
                    elif choice5==menu5[2]:
                        MacNemar_Paired_Proportions(df)
                    else:
                        st.write('coding error')
                else:
                    st.write('Coding error')
                        
            elif choice3==menu3[3]:
                    menu5=["Make your choice",
                           "Constant proportions between two binary outcomes, for matched / stratified data",
                           "Homogeneity between 2 observations"]
                    st.subheader("What kind of analysis ?")
                    choice5=st.selectbox("",menu5)
                    if choice5==menu5[0]:
                        pass
                    elif choice5==menu5[1]:
                        Mantel_Haenszel_Proportion(df)
                    elif choice5==menu5[2]:
                        Khi2_homogeneity_proportion_several(df)
                    else:
                        st.write('coding error')
                        
            elif choice3==menu3[4]:
                Khi2_proportion_several(df)
            else:
                st.subheader('Coding error')

                
        elif choice2==menu2[3]: # BY MEASURE / MEAN
            menu3=["Make your choice","Mean between 1 observation and Theory",
                   "Mean between 2 independent observations",
                   "Mean between paired observations","Mean between several independent observations"]

            st.write('\n\n\n\n\n')
            st.subheader("ALL VARIABLES ARE CONSIDERED AS ISSUED FROM NORMAL DISTRIBUTIONS")
            st.write ('(else Median Measurement Tests have to be considered)')
            st.write('\n\n\n\n\n')
            
            #st.subheader('CHOSEN ANALYSIS')
            choice3=st.selectbox("",menu3)
                
            if choice3==menu3[0]:
                pass
            
            elif choice3==menu3[1]: 
                Tstudent_vs_Theoretical_Means(df)
                   
            elif choice3==menu3[2]: 
                menu4=["Make your choice","YES","NO"]
                st.subheader("Same Variance in both populations ?")
                choice4=st.selectbox("",menu4)
                if choice4==menu4[0]:
                    pass
                elif choice4==menu4[1]:
                    #st.subheader("t-student test")
                    Two_Samples_Tstudent_Same_Variances_Means(df)
                elif choice4==menu4[2]:
                    #st.subheader("Welch corrected t-student test")
                    Two_Samples_Tstudent_Different_Variances_Means(df)
                else:
                    st.write('coding error')
                
            elif choice3==menu3[3]: 
                Two_Paired_Samples_Tstudent_Means(df) 
                
            elif choice3==menu3[4]:                 
                menu4=["Make your choice","YES","NO"]
                st.subheader("Same Variance for population of each observation ?")
                choice4=st.selectbox("",menu4)
                if choice4==menu4[0]:
                    pass
                elif choice4==menu4[1]:
                    One_Factor_Anova_Same_Variances_Means(df) 
                elif choice4==menu4[2]:    
                    One_Factor_Anova_Different_Variances_Means(df) 
                    
                else:
                    st.write("coding error")
                
        elif choice2==menu2[4]: # BY MEASURE / MEDIAN
            menu3=["Make your choice","Median between 1 observation and Theory",
                   "Median between 2 independent observations",
                   "Median between paired observations","Median between several independent observations"]
            
            #st.subheader('CHOSEN ANALYSIS')
            choice3=st.selectbox("",menu3)
            
            st.write('\n\n\n\n\n')
            st.subheader("NON PARAMETRIC TEST")
            st.write('\n\n\n\n\n')
                
            if choice3==menu3[0]: 
                pass
            elif choice3==menu3[1]: 
                #st.subheader("Wilcoxon sign test")
                Median_vs_Theory_Wilcoxon_Sign(df)
                
            elif choice3==menu3[2]: 
                #st.subheader("Mann Whitney Wilcoxon Test")
                Two_Independent_Samples_Mann_Whitney_Median(df)
                
            elif choice3==menu3[3]: 
                #st.subheader("Wilcoxon sign test on Median differences")            
                Two_Paired_Samples_Wilcoxon_Sign_Median(df)
                
            elif choice3==menu3[4]:                 
                #st.subheader("Kruskal Wallis Test")
                Several_Samples_Kruskal_Wallis_Median(df)
            else:
                st.write('coding error')
                
        elif choice2==menu2[5]: # BY MEASURE / QUANTILES
            #st.subheader("Median Test")  
            Quantiles(df)
        
        elif choice2==menu2[6]: # BY MEASURE / VARIANCES
            menu3=["Make your choice","Variances between 2 observations",
                   "Variances between several observations"]
                        
            #st.subheader('CHOSEN ANALYSIS')
            choice3=st.selectbox("",menu3)
            
            if choice3==menu3[0]:                 
                pass
            elif choice3==menu3[1]:
                menu4=["Make your choice","YES","NO"]
                st.subheader("Normal Distribution ?")
                choice4=st.selectbox("",menu4)
                if choice4==menu4[0]:
                    pass
                elif choice4==menu4[1]:
                    #st.subheader("Fisher Snedecor Test")
                    Two_Samples_Fisher_Snedecor_Parametric_Variances(df)
                elif choice4==menu4[2]:
                    #st.subheader("Ansari Bradley Test (non-parametric test)")  
                    Two_Samples_Ansary_Bradley_Non_Parametric_Variances(df)
                else:
                    st.write('coding error')
                
            elif choice3==menu3[2]:                 
                menu4=["Make your choice","YES","NO"]
                st.subheader("Normal Distribution ?")
                choice4=st.selectbox("",menu4)
                if choice4==menu4[0]:
                    pass
                elif choice4==menu4[1]:
                    #t.subheader("Bartlett Test")
                    Several_Samples_Bartlett_Parametric_Variances(df)
                elif choice4==menu4[2]:
                    #st.subheader("Fligner-Killeen Test (non-parametric test)")     
                    Several_Samples_Fligner_Killeen_Non_Parametric_Variances(df)
                else: 
                    st.write('coding error')
            else:
                st.write('coding error')
                    
        elif choice2==menu2[7]: # BY MEASURE / CORRELATIONS
            Input_Correlation_Text()
            st.write('\n\n')
            menu3=["Make your choice","Pearson Parametric Linear Correlation Test",
                   "Spearman Non Parametric Monotonous Correlation Test",
                   "Kendall Non Parametric Monotous Correlation Test"]
            
            choice3=st.selectbox("",menu3)
            
            if choice3==menu3[0]:                 
                pass
            elif choice3==menu3[1]:
                Pearson_Linear_Parametric_Correlation(df)
            elif choice3==menu3[2]:
                Spearman_Monotonous__Non_Parametric_Correlation(df)
            elif choice3==menu3[3]:
                Kendall_Monotonous__Non_Parametric_Correlation(df)        
            else:
                st.write('coding error')
            
        elif choice2==menu2[8]: # BY MEASURE / DISTRIBUTIONS
            Input_Distribution_Text()
            menu3=["Make your choice","Anderson-Darling",
                  "Shapiro-Wilk","D'agostino-Pearson"]
            st.write('Which test ?')
            choice3=st.selectbox("",menu3)
            
            if choice3==menu3[0]:                 
                pass
            elif choice3==menu3[1]:
                Anderson_Darling(df)
            elif choice3==menu3[2]:
                Shapiro_Wilk(df)
            elif choice3==menu3[3]:
                Dagostino_Pearson(df)
            else:
                st.write('coding error')
                                
        
        else:
            st.write("coding error")
                
    elif choice==menu[2]:
        menu2=["Make your choice",
               "Q2 + Theory", "Qk + Theory", "QT + Theory", 
               "Q2 + Qk", "Q2 + Qk + Theory","Q2 + QA (= 2xQ2 paired)",
               'QT + Q2', 'QT + Qk', 'QT + QA (=2xQT paired)', 
               "2xQ2","2xQk", "2xQT", "2xQ2 + Qk"]
        choice2=st.sidebar.selectbox("TYPE OF MEASURE",menu2)

        if choice2==menu2[0]: 
            #st.write('MAKE YOUR CHOICE')
            pass
        elif choice2==menu2[1]: # Q2 + Theory
            st.markdown('<u>Q2 + Theory </u>',unsafe_allow_html=True)
            st.write(' - Binomial proportions : binary proportions consistent with theory')
            st.write(' - Chi2 conformity : binary proportions consistent with theory')
            menu3=["Make your choice","Binomial Proportions",
                  "Chi2 conformity"]
            choice3=st.selectbox("",menu3)
            if choice3==menu3[0]:                 
                pass
            elif choice3==menu3[1]:
                Khi2_conformity_proportion(df)
            elif choice3==menu3[2]:
                st.write('????????????')
            else:
                st.write('coding error')
        
        elif choice2==menu2[2]: # Qk + Theory
            st.markdown('<u>Qk + Theory </u>',unsafe_allow_html=True)
            st.write(' - Chi2 conformity : quantity (proportions) in each classes are as expected by theory') 
            menu3=["Make your choice","Chi2 conformity"]
            choice3=st.selectbox("",menu3)
            
            if choice3==menu3[0]:                 
                pass
            elif choice3==menu3[1]:
                Khi2_conformity_count(df)
            else:
                st.write("coding error")
                
        elif choice2==menu2[3]: # QT + Theory
            st.markdown('<u>QT + Theory </u>',unsafe_allow_html=True)
            st.write(' - t-student : mean value of Qt is as expected by theory')
            st.write(' - Wicoxon sign (NP) : median value of Qt is as expected by theory')
            st.write(' - Chi² conformity : QT distribution is consistent with expected theory distribution')
            st.write(' - Sahpiro-Wilk : QT distribution is consistent with expected theory distribution')
            st.write(" - D'agostino-Pearson : QT distribution is consistent with expected theory distribution")
            st.write(' - Kolmogorov-Smirnov (NP): QT distribution is consistent with expected \
                    theory distribution')
            st.write(' - Anderson-Darling (NP) : QT distribution is consistent with expected theory distribution')
            menu3=["Make your choice","t-Student",
                  "Wilcoxon Sign","Chi² conformity","Sapiro-Wilk","Dagostino Pearson",
                  "Kolmogorov-Smirnov","Anderson-Darling"]
            choice3=st.selectbox("",menu3)
            if choice3==menu3[0]:                 
                pass
            elif choice3==menu3[1]:
                Tstudent_vs_Theoretical_Means(df)
            elif choice3==menu3[2]:
                Median_vs_Theory_Wilcoxon_Sign(df)
            elif choice3==menu3[3]:
                st.write("AFAIRE chi2 conformity")
            elif choice3==menu3[4]:
                Shapiro_Wilk(df)
            elif choice3==menu3[5]:
                Dagostino_Pearson(df)
            elif choice3==menu3[6]:
                st.write("A FAIRE Kolmogorov Smirnov")
            elif choice3==menu3[7]:
                Anderson_Darling(df)
            else:
                st.write('coding error')
        elif choice2==menu2[4]: # Q2 + Qk
            st.markdown('<u>Q2 + Qk </u>',unsafe_allow_html=True)
            st.write(' - Chi² homogeneity : same proportions of Q2 in each class of Qk')
            menu3=["Make your choice","Chi² Hoùogeneity"]
            choice3=st.selectbox("",menu3)
            if choice3==menu3[0]:                 
                pass
            elif choice3==menu3[1]:
                Khi2_homogeneity_proportion_several(df)
            else:
                st.write('coding error')
                
        elif choice2==menu2[5]: # Q2 + Qk + Theory
            st.markdown('<u>Q2 + Qk + Theory </u>',unsafe_allow_html=True)
            st.write(' - Chi² : all proportions of Q2 in each class of Qk are as expected by theory')
            menu3=["Make your choice","Chi² prooportions"]
            choice3=st.selectbox("",menu3)
            if choice3==menu3[0]:                 
                pass
            elif choice3==menu3[1]:
                st.write("Chi² proportions ???????")
            else:
                st.write('coding error')
                
        elif choice2==menu2[6]: # Q2 + QA (= 2xQ2 paired)
            st.markdown('<u>Q2 + QA = 2xQ2 paired </u>',unsafe_allow_html=True)
            st.write(' - Binomial paired : same proportions in two paired binomial classes ')
            st.write(' - MacNemar paired : same proportions in two paired binomial classes ')
            menu3=["Make your choice","Binomial paired","MacNemar paired"]
            choice3=st.selectbox("",menu3)
            if choice3==menu3[0]:                 
                pass
            elif choice3==menu3[1]:
                Binomial_Paired_Proportions(df)
            elif choice3==menu3[2]:
                MacNemar_Paired_Proportions(df)
            else:
                st.write('doeing error')
                
        elif choice2==menu2[7]: # QT + Q2
            st.markdown('<u>QT + Q2 </u>',unsafe_allow_html=True)
            st.write(' - t-student with same variance: mean value of QT is the same for the two classes of Q2')
            st.write(" - t-student Welch's corrected for different variances: \
                     mean value of QT is the same for the two classes of Q2")
            st.write(' - Mann-Whitney-Wilcoxon (NP) : median value of QT is the same for the two classes of Q2')
            st.write(' - Fisher-Snedecor : variance of QT is the same for the two classes of Q2')
            st.write(' - Ansari-Bradley (NP) : variance of QT is the same for the two classes of Q2')
            st.write(' - Kolmogorov-Smirnov : same distribution of QT in both classes of Q2')
            menu3=["Make your choice"," t-student with same variance","t-student Welch's corrected",
                  "Mann-Whitney-Wilcoxon ", "Fisher-Snedecor","Ansari-Bradley", "Kolmogorov-Smirnov"]
            choice3=st.selectbox("",menu3)
            if choice3==menu3[0]:                 
                pass
            elif choice3==menu3[1]:
                Two_Samples_Tstudent_Same_Variances_Means(df) 
            elif choice3==menu3[2]:
                Two_Samples_Tstudent_Different_Variances_Means(df) 
            elif choice3==menu3[3]:
                Two_Independent_Samples_Mann_Whitney_Median(df)
            elif choice3==menu3[4]:
                Two_Samples_Fisher_Snedecor_Parametric_Variances(df)
            elif choice3==menu3[5]:
                Two_Samples_Ansary_Bradley_Non_Parametric_Variances(df)
            elif choice3==menu3[6]:
                st.write('kolmogorov smirnov ??????')
            else:
                st.write('coding error')
                                
        elif choice2==menu2[8]: # QT + Qk
            st.markdown('<u>QT + Qk </u>',unsafe_allow_html=True)
            st.write(' - ANOVA : same mean values for QT in all classes of Qk (same variances)')
            st.write(" - ANOVA with WELCH's correction : \
                     same mean values for QT in all classes of Qk (different variances)")
            st.write(' - Kruskal-Wallis (NP) : same median values for QT in all classes of Qk ')
            st.write(' - Median test (NP) : same median (quantile) values for QT in all classes of Qk ')
            st.write(' - Bartlett : same variance values for QT in all classes of Qk')
            st.write(' - Fligner-Killeen (NP) : same variance values for QT in all classes of Qk')
            menu3=["Make your choice","ANOVA  with same variance","ANOVA with WELCH's correction",
                  "Kruskal-Wallis", "Median test","Bartlett", "Fligner-Killeen"]
            choice3=st.selectbox("",menu3)
            if choice3==menu3[0]:                 
                pass
            elif choice3==menu3[1]:
                One_Factor_Anova_Same_Variances_Means(df) 
            elif choice3==menu3[2]:
                One_Factor_Anova_Different_Variances_Means(df)
            elif choice3==menu3[3]:
                Several_Samples_Kruskal_Wallis_Median(df)
            elif choice3==menu3[4]:
                Quantiles(df)
            elif choice3==menu3[5]:                    
                Several_Samples_Bartlett_Parametric_Variances(df)
            elif choice3==menu3[6]:
                Several_Samples_Fligner_Killeen_Non_Parametric_Variances(df)
            else:
                st.write("coding error")
                                
        elif choice2==menu2[9]: # QT + QA (=2xQT paired)
            st.markdown('<u>QT + QA = 2xQT paired </u>',unsafe_allow_html=True)
            st.write(' - t-student paired : mean value of QT is the same for both class of QA')
            st.write(' - Wilcoxon sign paired (NP) : median value of QT is the same for both class of QA')
            menu3=["Make your choice","t-student paired","Wilcoxon sign paired"]
            choice3=st.selectbox("",menu3)
            if choice3==menu3[0]:                 
                pass
            elif choice3==menu3[1]:
                Two_Paired_Samples_Tstudent_Means(df) 
            elif choice3==menu3[2]:
                Two_Paired_Samples_Wilcoxon_Sign_Median(df)
            else:
                st.write('coding error')
                                
        elif choice2==menu2[10]: # 2xQ2
            st.markdown('<u>2xQ2 </u>',unsafe_allow_html=True)
            st.write(' - ChI² homogeneity : same proportions in both Q2 classes')
            st.write(' - Fisher exact test 2x2 : same proportions in both Q2 classes')
            menu3=["Make your choice","ChI² homogeneity","Fisher exact test 2x2"]
            choice3=st.selectbox("",menu3)
            if choice3==menu3[0]:                 
                pass
            elif choice3==menu3[2]:
                Fisher_test_2x2_Proportions(df)
            elif choice3==menu3[1]:
                Khi2_homogeneity_proportion(df)
            else:
                st.write('coding error')
                                
        elif choice2==menu2[11]: # 2xQk
            st.markdown('<u>2xQk </u>',unsafe_allow_html=True)
            st.write(' - Chi² Homogeneity : independance between the two Qk classes')
            st.write(' - G test : independance between the two Qk classes')
            st.write(' - Fisher exact test cxk : independance between the two Qk classes')
            menu3=["Make your choice","Chi² Homogeneity",
                  "G test ","Fisher exact test cxk"]
            choice3=st.selectbox("",menu3)
            if choice3==menu3[0]:                 
                pass
            elif choice3==menu3[1]:
                Khi2_homogeneity_count(df)
            elif choice3==menu3[2]:
                G_count(df,Graph=False)
            elif choice3==menu3[3]:
                Fisher_test_cxk_Count(df)
            else:
                st.write('coding error')
                                
        elif choice2==menu2[12]: # 2xQt 
            st.markdown('<u>2xQT</u>',unsafe_allow_html=True)
            st.write(' - Pearson : correlation between two QT (expected linear correlation)')
            st.write(' - Spearman (NP) : correlation between two QT (expected monotonous correlation)')
            st.write(' - Kendall (NP) : correlation between two QT (expected monotonous correlation)')
            menu3=["Make your choice","Pearson","Spearman","Kendall"]
            choice3=st.selectbox("",menu3)
            if choice3==menu3[0]:                 
                pass
            elif choice3==menu3[1]:
                Pearson_Linear_Parametric_Correlation(df)
            elif choice3==menu3[2]:
                Spearman_Monotonous__Non_Parametric_Correlation(df)
            elif choice3==menu3[3]:
                Kendall_Monotonous__Non_Parametric_Correlation(df)
            else:
                st.write('coding error')
                                
        elif choice2==menu2[13]: # 2xQ2+Qk 
            st.markdown('<u>2xQ2 + Qk</u>',unsafe_allow_html=True)
            st.write(' - Mantel Haenszel : independance of binary classes in each class of Qk')
            st.write(' - Mantel Haenszel : same proportions of binary classes in all classes of Qk')
            menu3=["Make your choice","Mantel Haenszel Independance",
                  "Mantel Haenszel proportions"]
            choice3=st.selectbox("",menu3)
            if choice3==menu3[0]:                 
                pass
            elif choice3==menu3[1]:
                Mantel_Haenszel_Count(df)
            elif choice3==menu3[2]:
                Mantel_Haenszel_Proportion(df)
            else:
                st.write('coding error')
                                
        else:
            st.write('coding error')
    elif choice==menu[3]:
        st.header('Work on-going')
    elif choice==menu[4]: 
        Notations()
    elif choice==menu[5]: 
        References()
    else:
        st.write("coding error")
    
    st.sidebar.markdown("******") 
    st.sidebar.text("Version 1.1 18/05/2021")
    st.sidebar.text("By F. Bocage")
    st.sidebar.text("Contact : frederic_91@orange.fr")

    
    
    
if __name__=="__main__":
    main()