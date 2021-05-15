import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
from statsmodels.stats.contingency_tables import SquareTable
import seaborn as sns
from PIL import Image
from scipy.stats import chi2


def Khi2_conformity_proportion(df):
    st.subheader('<<< See Khi2_conformity_proportion() function >>>')
    obs1=df['Couleur']
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
    st.pyplot(fig)
    f_obs=y
    
    with st.echo():
        from scipy.stats import chisquare
        res=chisquare(f_obs, f_exp=[50,20,30])
        # inputs :
        #    - f_obs :        observed proportions (in %)
        #    - f_exp :        expected proportinos (in %)
        # returns : 
        #    - statistics :   Khi2 statistic value
        #    - pvalue :       p-value
    st.write(res)
    
def Khi2_conformity_count(df):
    st.subheader('<<< See Khi2_conformity_count() function >>>')
    obs1=df['Couleur']
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
    plt.ylabel("Count")
    plt.title("")
    #st.write(y)
    plt.xticks(x_pos, y.index)
    col1,col2=st.beta_columns([0.5,1]) 
    with col1 : 
            st.pyplot(fig)
    f_obs=y
    
    with st.echo():
        from scipy.stats import chisquare
        res=chisquare(f_obs, f_exp=[48,29,25])
        # inputs :
        #    - f_obs :        observed counts 
        #    - f_exp :        expected counts
        # returns : 
        #    - statistics :   Khi2 statistic value
        #    - pvalue :       p-value
    st.write(res)

def Khi2_homogeneity_count(df):
    st.subheader('<<< See Khi2_homogeneity_count() function >>>')
    obs1=df[['Enracinement','Couleur']]
    obs1=obs1.dropna()
    #st.dataframe(obs1,width=900) 
    #st.write(nber)
    fig=plt.figure()
    #plt.style.use('ggplot')
    props = {}
    for yname in ["Fort","Tres.fort","Faible","Moyen"]:
        props[('Rouge', yname)] = {'color': 'tomato'}
        props[('Jaune', yname)] = {'color': 'goldenrod'}
        props[('Jaune.rouge', yname)] = {'color': 'peru'}
        
    mosaic(obs1, ['Couleur', 'Enracinement'],properties=props,gap=0.015)
    plt.xlabel("Enracinement vs Couleur") 
    plt.ylabel("Couleur")
    plt.title("")
    plt.tight_layout()
    #st.pyplot(fig)
    plt.savefig('output.png')
    img=Image.open('output.png')
    st.image('output.png')
    conting=pd.crosstab(index=obs1['Couleur'], columns=obs1['Enracinement'],margins=True)
    st.write(conting) 
    conting=pd.crosstab(index=obs1['Couleur'], columns=obs1['Enracinement'],margins=False)
    
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
    expected=pd.DataFrame(data=expected,columns=conting.columns.tolist(),index=conting.index.tolist())
    st.write("Expected values = \n",expected)
    khi2=(conting-expected)**2/expected
    st.write("Splitted khi2 = \n",khi2)
    khi2pct=(conting-expected)**2/expected/chi2*100
    st.write("Splitted khi2 in % = \n",khi2pct)
    #st.write(khi2.sum().sum())
    
def G_count(df,Graph=False):
    st.subheader('<<< See G_count() function >>>')
    obs1=df[['Enracinement','Couleur']]
    obs1=obs1.dropna()
    conting=pd.crosstab(index=obs1['Couleur'], columns=obs1['Enracinement'],margins=False)
    if Graph==True:
        #st.dataframe(obs1,width=900) 
        #st.write(nber)
        fig=plt.figure()
        #plt.style.use('ggplot')
        props = {}
        for yname in ["Fort","Tres.fort","Faible","Moyen"]:
            props[('Rouge', yname)] = {'color': 'tomato'}
            props[('Jaune', yname)] = {'color': 'goldenrod'}
            props[('Jaune.rouge', yname)] = {'color': 'peru'}

        mosaic(obs1, ['Couleur', 'Enracinement'],properties=props,gap=0.015)
        plt.xlabel("Enracinement vs Couleur") 
        plt.ylabel("Couleur")
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

    
def Khi2_homogeneity_proportion(df):
    st.subheader('<<< See Khi2_homogeneity_proportion() function >>>\n\n\n')
    obs1=df[['Enracinement','Couleur']]
    obs1=obs1.dropna()
    
    #st.dataframe(obs1,width=900) 
    #st.write(nber)
    fig=plt.figure()
    #plt.style.use('ggplot')
    props = {}
    for yname in ["Fort","Tres.fort","Faible","Moyen"]:
        props[('Rouge', yname)] = {'color': 'tomato'}
        props[('Jaune', yname)] = {'color': 'goldenrod'}
        props[('Jaune.rouge', yname)] = {'color': 'peru'}
        
    mosaic(obs1, ['Couleur', 'Enracinement'],properties=props,gap=0.015)
    plt.xlabel("Enracinement vs Couleur") 
    plt.ylabel("Couleur")
    plt.title("")
    plt.tight_layout()
    #st.pyplot(fig)
    plt.savefig('output.png')
    img=Image.open('output.png')
    st.image('output.png')
    conting=pd.crosstab(index=obs1['Couleur'], columns=obs1['Enracinement'],margins=False)
    conting_sum=conting.sum().sum()
    conting=pd.crosstab(index=obs1['Couleur'], columns=obs1['Enracinement'],margins=True)/conting_sum
    st.write(conting) 
    conting=pd.crosstab(index=obs1['Couleur'], columns=obs1['Enracinement'],margins=False)/conting_sum
    
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
    expected=pd.DataFrame(data=expected,columns=conting.columns.tolist(),index=conting.index.tolist())
    st.write("Expected proportions = \n",expected)
    st.write("Total of expected proportion",expected.sum().sum())
    khi2=(conting-expected)**2/expected
    st.write("Splitted khi2 = \n",khi2)
    st.write("Total of splitted Khi2",khi2.sum().sum())
    khi2pct=(conting-expected)**2/expected/chi2*100
    st.write("Splitted khi2 in % = \n",khi2pct)


def Khi2_homogeneity_proportion_several(df):
    st.subheader('<<< See Khi2_homogeneity_proportion_several() function >>>\n\n\n')
    obs1=df[['Verse','Couleur']]
    obs1=obs1.dropna()
    conting=pd.crosstab(index=obs1['Couleur'], columns=obs1['Verse'],margins=False)
    st.write('with proportions of "Non" in the different classes')
    conting['Proportions']=conting.apply(lambda row: row["Non"]/(row['Non']+row["Oui"])*100,axis=1) 
    conting['N by colour']=conting.apply(lambda row: (row['Non']+row["Oui"]),axis=1) 
    st.dataframe(conting)
    Ntot=conting["N by colour"].sum()
    st.write('total =',Ntot )
    Ntot1=conting["Non"].sum() 
    st.write('total for "Non" =',Ntot1 )
    Ptot1=Ntot1/Ntot
    st.write("Proportion of 'Non' =",Ptot1*100)
    #conting=pd.crosstab(index=obs1['Couleur'], columns=obs1['Verse'],margins=False)
    #conting_sum=conting.sum().sum()
    #conting=pd.crosstab(index=obs1['Couleur'], columns=obs1['Verse'],margins=True)/conting_sum
    #st.write(conting) 
    #conting=pd.crosstab(index=obs1['Couleur'], columns=obs1['Verse'],margins=False)/conting_sum
    
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
    st.write("Degrees of Freedom = ",dof)
    st.write("p-value = ",pvalue)
    st.write('As p-value is below 5%, the proportions of "Non" in the different k classes are different \
            (H0 rejected)')
    
def Mantel_Haenszel_Count(df):    
    st.subheader('<<< See Mantel_Haenszel_Count() function >>>\n\n\n')
    st.write("For two binary classes F and G (2 x Q2) and one k-level class K (1 x Qk) ")
    st.subheader("Are the classes F and G independent in each of the k classes of K ? ")
    obs1=df[['Verse','Couleur',"Attaque"]]
    obs1=obs1.dropna()
    color_label=obs1.Couleur.unique()
    #st.write("labels for couleur : ", color_label)   
    
    st.write('The proportions of "Attaque" = "Oui" given "Verse" for the different "Couleur" are :')
    crosstab_list=[]
    for lab in color_label:
        st.write('For "Couleur" label = ',lab)
        tmp=obs1[obs1['Couleur']==lab]
        tmp=pd.crosstab(index=tmp['Verse'], columns=tmp['Attaque'],margins=True,\
                        rownames=['Verse'], colnames=['Attaque']) 
        tmp['Proportions of Oui']=tmp.apply(lambda row: row["Oui"]/(row['Oui']+row['Non']),axis=1)
        col1,col2=st.beta_columns([0.5,1]) 
        #st.write(tmp)
        with col2 : 
            st.write ('row="Verse", col="Attaque"')
            st.dataframe(tmp)
        crosstab_list.append(tmp.loc[['Oui','Non'],['Oui','Non']])
        tmp=tmp.loc[['Oui','Non'],['Proportions of Oui']]
        fig=plt.figure() 
        
        plt.bar(tmp.index.tolist(),tmp['Proportions of Oui'])
        plt.xlabel('Verse' )
        plt.title('for "Couleur" = '+lab)
        plt.ylabel('Proportions of "Attaque" = "Oui"') 
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
    st.write('.test_null_odds gives the statistics and p-value' + \
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


def Mantel_Haenszel_Proportion(df):    
    st.subheader('<<< See Mantel_Haenszel_Proportion() function >>>\n\n\n')
    st.write("For two binary classes F and G (2 x Q2) and one k-level class K (1 x Qk) ")
    st.subheader("Are the proportions within class F and G identical whatever level of class K ? ")
    obs1=df[['Verse','Couleur',"Attaque"]]
    obs1=obs1.dropna()
    color_label=obs1.Couleur.unique()
    #st.write("labels for couleur : ", color_label)   
    
    st.write('The proportions of "Attaque" = "Oui" given "Verse" for the different "Couleur" are :')
    crosstab_list=[]
    for lab in color_label:
        st.write('For "Couleur" label = ',lab)
        tmp=obs1[obs1['Couleur']==lab]
        tmp=pd.crosstab(index=tmp['Verse'], columns=tmp['Attaque'],margins=True,\
                        rownames=['Verse'], colnames=['Attaque']) 
        tmp['Proportions of Oui']=tmp.apply(lambda row: row["Oui"]/(row['Oui']+row['Non']),axis=1)
        col1,col2=st.beta_columns([0.5,1]) 
        #st.write(tmp)
        with col2 : 
            st.write ('row="Verse", col="Attaque"')
            st.dataframe(tmp)
        crosstab_list.append(tmp.loc[['Oui','Non'],['Oui','Non']])
        tmp=tmp.loc[['Oui','Non'],['Proportions of Oui']]
        fig=plt.figure() 
        
        plt.bar(tmp.index.tolist(),tmp['Proportions of Oui'])
        plt.xlabel('Verse' )
        plt.title('for "Couleur" = '+lab)
        plt.ylabel('Proportions of "Attaque" = "Oui"') 
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
    
def Cochran_Description():
    st.subheader('Cochran condition test is as follows :')
    st.write('   - a minimum of 5 elements in each classes')
    st.write('   - it is allowed to have classes with between 1 and 5 elements' +\
             'only if 80% of all classes have more than 6 elements')    
    
def Fisher_test_cxk_Count(df):
    st.subheader('<<< See Fisher_test_cxk_Count() function >>>\n\n\n')
    st.write('This test check the independance between two classes F and G' +\
             'with different levels c and k, i.e. 2 x Qk')
    st.write('Independance means that for every level in class F, the elements are ' +\
             'evenly splitted in class G, and vice versa')
    obs1=df[['Germination.epi',"Enracinement"]]
    obs1=obs1.dropna()
    fig=plt.figure()
    #plt.style.use('ggplot')
    props = {}
    for yname in ["Oui","Non"]:
        props[('Faible', yname)] = {'color': 'tomato'}
        props[('Fort', yname)] = {'color': 'goldenrod'}
        props[('Moyen', yname)] = {'color': 'peru'}
        props[('Tres.fort', yname)] = {'color': 'olivedrab'}
    
    mosaic(obs1.sort_values('Germination.epi',ascending=False).sort_values('Enracinement')  
           ,['Enracinement','Germination.epi'],properties=props,gap=0.015)
    plt.xlabel("Enracinement vs Germination.epi") 
    plt.ylabel("Germination.epi")
    plt.title("")
    plt.tight_layout()
    #st.pyplot(fig)
    plt.savefig('output.png')
    img=Image.open('output.png')
    st.image('output.png') 
    
    conting=pd.crosstab(index=obs1['Enracinement'], columns=obs1['Germination.epi'],margins=False)
    st.write(conting)
    
    #with st.echo():
        # from scipy.stats import fisher_exact # only on 2x2 table !
        #from FisherExact import fisher_exact
        #res=fisher_exact(conting)    
    st.subheader('Python librairy to perform cxk Fisher Exact Test not found')
    st.subheader('Potential solution is to use R existing librairies as described in reference [TBD]')

def Fisher_test_2x2_Proportions(df):
    st.subheader('<<< See Fisher_test_2x2_Proportions() function >>>\n\n\n')
    st.write('This test concerns two classes F and G with two levels, i.e. 2 x Q2')
    st.write('This test checks that the proportions between both class F levels are identical'+\
            'in both level of class G')
    obs1=df[['Germination.epi',"Verse"]]
    obs1=obs1.dropna()
    fig=plt.figure()
    #plt.style.use('ggplot')
    props = {}
    for yname in ["Oui","Non"]:
        props[('Oui', yname)] = {'color': 'tomato'}
        props[('Non', yname)] = {'color': 'goldenrod'}
        #props[('Moyen', yname)] = {'color': 'peru'}
        #props[('Tres.fort', yname)] = {'color': 'olivedrab'}
    
    mosaic(obs1.sort_values('Germination.epi',ascending=False).sort_values('Verse',ascending=False)  
           ,['Verse','Germination.epi'],properties=props,gap=0.015)
    plt.xlabel("Verse vs Germination.epi") 
    plt.ylabel("Germination.epi")
    plt.title("")
    plt.tight_layout()
    #st.pyplot(fig)
    plt.savefig('output.png')
    img=Image.open('output.png')
    st.image('output.png') 
    
    conting=pd.crosstab(index=obs1['Verse'], columns=obs1['Germination.epi'],margins=False)
    #conting=pd.DataFrame([[40,60],[40,60]],index=conting.index,columns=conting.columns)
    st.write(conting)
    st.write('Total :',conting.sum().sum())
    
    with st.echo():
        from scipy.stats import fisher_exact # only on 2x2 table !
        #from FisherExact import fisher_exact
        odds_ratio,pvalue=fisher_exact(conting) 
    st.write("Odds ratio : ",odds_ratio)
    st.write("if odds ratio is 1, both proportions are equal")
    st.write("p-value regarding 'both proportions are equal' : ",pvalue)
    
    
    
def Binomial_vs_Theory_Proportions(df):
        st.subheader('<<< See Binomial_vs_Theory_Proportions() function >>>\n\n\n')
        obs1=df['Attaque']
        obs1=obs1.dropna()
        conting=obs1.value_counts()
        st.write('Count : \n',conting)
        total=conting.sum()
        #st.write('Total : ',total)
        #conting=conting/total*100
        #st.write('Proportions (%) :\n',conting)
        with st.echo():    
            from scipy.stats import binom_test
            Ptheo=60 # theoretical proportions assumed to be 60%
            pvalue=binom_test(x=conting['Oui'],n=100,p=Ptheo/100,alternative='two-sided')
        st.write('p-value for Probability of "Oui" is 60% : ',pvalue)
        with st.echo():    
            from scipy.stats import binom_test
            Ptheo=60 # theoretical proportions assumed to be 60%
            pvalue=binom_test(x=conting['Non'],n=100,p=Ptheo/100,alternative='two-sided')
        st.write('p-value for Probability of "Non" is 60% : ',pvalue)
                
def Binomial_Paired_Proportions(df):
        st.subheader('<<< See Binomial_Paired_Proportions() function >>>\n\n\n')
        st.write('This test applies for two binary classes F and G, with levels of G being paired '+ \
                '(for example two measured) corresponding to 1Q2 + 1QA')
        #st.write(df.columns)
        obs1=df[['Verse','Verse.Traitement']]
        obs1=obs1.dropna()
        st.write(obs1)
        st.write("In this example, two measures are performed 'Verse' and 'Verse.Traitement' (class G)"  + \
                "with two outcomes 'Oui' or 'Non' (Class F)")
        conting=pd.crosstab(index=obs1['Verse'], columns=obs1['Verse.Traitement'],margins=False,\
                        rownames=['Verse'], colnames=['Verse.Traitement']) 
        
        
        st.write('verse in rows, verse.Traitement in columns : \n',conting)
        total=conting.sum().sum()
        st.write('Total : ',total)
        
        Ns=conting.loc['Oui','Non']+conting.loc['Non','Oui']
        st.write('Total number of inconsistent measurements Ns = : ', Ns)
        p12=conting.loc['Oui','Non']/Ns*100
        p21=conting.loc['Non','Oui']/Ns*100
        st.write('Probability of "Oui/Non" inconsistency (given inconsistency) is ',p12 )
        st.write('Probability of "Non/Oui" inconsistency (given inconsistency) is ',p21 )
        st.write('The purpose of the test is to check whether both these probabilities are identical, thus 0.5%')
        st.write('This corresponds to a Binomial test B(Nmax=Ns,p=0.5) applied to the number of ' +\
                'inconsistencies "Oui.Non"')
        
        #st.write(conting)
        #conting=conting/total*100
        #st.write('Proportions (%) :\n',conting)
        with st.echo():    
            from scipy.stats import binom_test
            pvalue=binom_test(x=conting.loc['Oui','Non'],n=Ns,p=0.5,alternative='two-sided')
        st.write('p-value for Probability of proportion equality : ',pvalue)

def MacNemar_Paired_Proportions(df):
        st.subheader('<<< See MacNemar_Paired_Proportions() function >>>\n\n\n')
        st.write('This test applies for two binary classes F and G, with levels of G being paired '+ \
                '(for example two measured) corresponding to 1Q2 + 1QA')
        #st.write(df.columns)
        obs1=df[['Verse','Verse.Traitement']]
        obs1=obs1.dropna()
        st.write(obs1)
        st.write("In this example, two measures are performed 'Verse' and 'Verse.Traitement' (class G)"  + \
                "with two outcomes 'Oui' or 'Non' (Class F)")
        conting=pd.crosstab(index=obs1['Verse'], columns=obs1['Verse.Traitement'],margins=False,\
                        rownames=['Verse'], colnames=['Verse.Traitement']) 
        
        
        st.write('verse in rows, verse.Traitement in columns : \n',conting)
        total=conting.sum().sum()
        st.write('Total : ',total)
        
        Ns=conting.loc['Oui','Non']+conting.loc['Non','Oui']
        st.write('Total number of inconsistent measurements Ns = : ', Ns)
        p12=conting.loc['Oui','Non']/Ns*100
        p21=conting.loc['Non','Oui']/Ns*100
        st.write('Probability of "Oui/Non" inconsistency (given inconsistency) is ',p12 )
        st.write('Probability of "Non/Oui" inconsistency (given inconsistency) is ',p21 )
        st.write('The purpose of the test is to check whether both these probabilities are identical, thus 0.5%')
        st.write('This corresponds to a Binomial test B(Nmax=Ns,p=0.5) applied to the number of ' +\
                'inconsistencies "Oui.Non"')    
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
            #                     min(n12,n21) if exact is False (binary test) to be compared  with (n12+n21) / 2
            #                         where n12 and n21 are the number of inconsistency cases in table 
            #     - pvalue :     p-value
            res1=mcnemar(table=conting,exact=True)
            st.write('with binomial exact test :')
            st.write('    min of n12 and n21 : ', res1.statistic)
            st.write('    min of p12 and p21 : ', \
                     res1.statistic/(conting.loc['Oui','Non']+conting.loc['Non','Oui'])*100)
            st.write('p-value : ',res1.pvalue)
            res2=mcnemar(table=conting,exact=False,correction=False)
            st.write('\n\n\n')
            st.write('with chi2 approximative test without continuity correction :')
            st.write('    chi2 : ', res2.statistic)
            st.write('p-value : ',res2.pvalue)           
            res3=mcnemar(table=conting,exact=False,correction=True)
            st.write('\n\n\n')
            st.write('with chi2 approximative test with continuity correction :')
            st.write('    chi2 : ', res3.statistic)
            st.write('p-value : ',res3.pvalue)
        

def Khi2_proportion_several(df):
    st.subheader('<<< See Khi2_proportion_several() function >>>\n\n\n')
    st.subheader("ATTENTION : Cochran's Rule has to be respected !!!!!!")
    st.write('This test compares k observed proportions to k theoretical proportions')
    st.write('One class F having k levels and one class G having 2 levels, i.e. 1 x Qk and 1 x Q2')
        
    obs1=df[['Verse','Couleur']]
    obs1=obs1.dropna()
    conting=pd.crosstab(index=obs1['Couleur'], columns=obs1['Verse'],margins=False)
    st.write('with proportions of "Non" in the different classes')
    conting['Proportions']=conting.apply(lambda row: row["Non"]/(row['Non']+row["Oui"])*100,axis=1) 
    conting['N by level']=conting.apply(lambda row: (row['Non']+row["Oui"]),axis=1) 
    st.dataframe(conting)
    Ntot=conting["N by level"].sum()
    st.write('total =',Ntot )
    #Ntot1=conting["Non"].sum() 
    #st.write('total for "Non" =',Ntot1 )
    #Ptot1=Ntot1/Ntot
    #st.write("Proportion of 'Non' =",Ptot1*100)
    #conting=pd.crosstab(index=obs1['Couleur'], columns=obs1['Verse'],margins=False)
    #conting_sum=conting.sum().sum()
    #conting=pd.crosstab(index=obs1['Couleur'], columns=obs1['Verse'],margins=True)/conting_sum
    #st.write(conting) 
    #conting=pd.crosstab(index=obs1['Couleur'], columns=obs1['Verse'],margins=False)/conting_sum
    
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
        khi2,pvalue,_=proportions_chisquare(count=conting["Non"], nobs=conting["N by level"], value=Ptheo)
        
    st.write("Chi2 = ",khi2)
    st.write('Theoretical expectations are supposed not to be estimated from data => degrees of freedom')
    st.write("Degrees of Freedom = ",len(Ptheo))
    st.write("p-value = ",pvalue)
    st.write('As p-value is below 5%, the proportions of "Non"  in the different levels are not consistent ',
             'with theoretical expectations')        

    
def Tstudent_vs_Theoretical_Means(df):
    st.subheader('<<< See Tstudent_vs_Theoretical_Means() function >>>\n\n\n')
    st.write('This test compares the mean value of an observed distribution with a theoretical \
            expected mean value')
    
    st.subheader ('One sample t-Student test')
    
    st.write("example based on the 'Hauteur' values for East Parcelle in Mais example")
    obs1=df[df['Parcelle']=='Est'].Hauteur
    obs1=obs1.dropna()  
    st.write("Observed data :\n")
    st.dataframe(obs1,height=200)
    st.write('mean observed value : ',obs1.mean())
    
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

        
def Two_Samples_Tstudent_Same_Variances_Means(df) :
    st.subheader('<<< See Two_Samples_Tstudent_Same_Variances_Means function >>>\n\n\n')
    st.write('This test compares the mean value of two observed distributions with same known variance')
    
    st.subheader ('Two samples t-Student test with same variance')
    st.write("example based on the 'Hauteur' values for North and South Parcelle in Mais example")
    obs1=df[df['Parcelle']=='Nord'].Hauteur
    obs1=obs1.dropna()  
    st.write("Observed data for 'Nord':\n")
    st.dataframe(obs1,height=200)
    st.write("mean observed value for 'Nord' : ",obs1.mean())
    obs2=df[df['Parcelle']=='Sud'].Hauteur
    obs2=obs2.dropna()  
    st.write("Observed data for 'Sud':\n")
    st.dataframe(obs2,height=200)
    st.write("mean observed value for 'Sud': ",obs2.mean())

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
    st.write('statistic t-value : ',statistic)
    st.write(' p-value : ',pvalue)
    
   
        
def Two_Samples_Tstudent_Different_Variances_Means(df) :
    st.subheader('<<< See Two_Samples_Tstudent_Different_Variances_Means function >>>\n\n\n')
    st.write('This test compares the mean value of two observed distributions with different variances')
    
    st.subheader ('Two samples t-Student test with different variances - WELCH test')
    st.write("example based on the 'Masse' values for North and South Parcelle in Mais example")
    obs1=df[df['Parcelle']=='Nord'].Masse
    obs1=obs1.dropna()  
    st.write("Observed data for 'Nord':\n")
    st.dataframe(obs1,height=200)
    st.write("mean observed value for 'Nord' : ",obs1.mean())
    obs2=df[df['Parcelle']=='Sud'].Masse
    obs2=obs2.dropna()  
    st.write("Observed data for 'Sud':\n")
    st.dataframe(obs2,height=200)
    st.write("mean observed value for 'Sud': ",obs2.mean())

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
    st.write('statistic t-value : ',statistic)
    st.write(' p-value : ',pvalue)
        
def Two_Paired_Samples_Tstudent_Means(df) :        
    st.subheader('<<< See Two_Paired_Samples_Tstudent_Means() function >>>\n\n\n')
    st.write("This test compares the mean value of two observed distributions corresponding \
             to paired (not independant) measurements")
    st.write('This test consists to perform a one-sample t-test \
            on observed differences between both paired measurements, with a theoretical mean of 0')
    
    st.subheader ('Two paired samples t-Student test')
    st.write("example based on the 'Hauteur' and 'Hauteur.J7' values for North Parcelle in Mais example")
    obs1=df[df['Parcelle']=='Nord'].Hauteur
    obs1=obs1.dropna()  
    st.write("Observed data for 'Hauteur':\n")
    st.dataframe(obs1,height=200)
    st.write("mean observed value for 'Hauteur' : ",obs1.mean())
    obs2=df[df['Parcelle']=='Nord']
    obs2=obs2['Hauteur.J7']
    obs2=obs2.dropna()  
    st.write("Observed data for 'Hauteur.J7':\n")
    st.dataframe(obs2,height=200)
    st.write("mean observed value for 'Hauteur.J7': ",obs2.mean())        

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

    
def One_Factor_Anova_Same_Variances_Means(df) : 
    st.subheader('<<< See One_Factor_Anova_Same_Variances_Means() function >>>\n\n\n')
    st.write("This test compares the mean value of more than 2 observed distributions \
            with same expected known variances in population (homoscedasticity)")
    st.write("example based on the 'Hauteur' values for the 4 Parcelles in Mais example")
    obs=df[['Hauteur','Parcelle']]
    obs=obs.dropna()  
    st.write("Observed data for 'Hauteur':\n")
    #st.dataframe(obs,height=200)
    obs1=obs[obs['Parcelle']=='Est'].Hauteur
    st.write("For 'Parcelle' = 'Est' :")
    st.dataframe(obs1,height=200)
    obs2=obs[obs['Parcelle']=='Nord'].Hauteur
    st.write("For 'Parcelle' = 'Nord' :")
    st.dataframe(obs2,height=200)
    obs3=obs[obs['Parcelle']=='Ouest'].Hauteur
    st.write("For 'Parcelle' = 'Ouest' :")
    st.dataframe(obs3,height=200)
    obs4=obs[obs['Parcelle']=='Sud'].Hauteur
    st.write("For 'Parcelle' = 'Sud' :")
    st.dataframe(obs4,height=200)
    
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
    
def One_Factor_Anova_Different_Variances_Means(df) :      
    st.subheader('<<< See One_Factor_Anova_Different_Variances_Means() function >>>\n\n\n')
    st.write("This test compares the mean value of more than 2 observed distributions \
            with different expected known variances in population")
    st.write("In this case, Welch corrected ANOVA has to be applied")
    st.write("example based on the 'Masse' values for the 4 Parcelles in Mais example")
    #st.write(df.columns)
    obs=df[['Masse','Parcelle']]
    obs=obs.dropna()  
    st.write("Observed data for 'Masse':\n")
    #st.dataframe(obs,height=200)
    obs1=obs[obs['Parcelle']=='Est'].Masse
    st.write("For 'Parcelle' = 'Est' :")
    st.dataframe(obs1,height=200)
    obs2=obs[obs['Parcelle']=='Nord'].Masse
    st.write("For 'Parcelle' = 'Nord' :")
    st.dataframe(obs2,height=200)
    obs3=obs[obs['Parcelle']=='Ouest'].Masse
    st.write("For 'Parcelle' = 'Ouest' :")
    st.dataframe(obs3,height=200)
    obs4=obs[obs['Parcelle']=='Sud'].Masse
    st.write("For 'Parcelle' = 'Sud' :")
    st.dataframe(obs4,height=200)
    
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

    
def Median_vs_Theory_Wilcoxon_Sign(df):
    st.subheader('<<< See Median_vs_Theory_Wilcoxon_Sign() function >>>\n\n\n')
    st.write("This test is a non parametric test used to compare a median \
            value of an observed sample with a median theoretical value")
    obs=df[df['Parcelle']=='Sud']
    obs=obs[['Masse.grains']]
    obs=obs.dropna()  
    st.write("Observed data for 'Masse.grains':\n")
    st.write(obs)
    st.write("Median value : ",obs.median())
    
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
        #     - zero_method :  management of zero differences
        #     - correction :   boolean ; for continuity correction
        #     - alternative :  alternative method H1
        #     - mode :         pvalue calculation method
        # OUTPUTS :
        #     - Statistic :    F value
        #     - pvalue :       p-value     
        statistic,pvalue=wilcoxon(x=obs["Masse.grains"],y=Theo,zero_method='wilcox',
                                  correction=False,alternative="two-sided",mode='exact')
    st.write('statistic : ',statistic)
    st.write(' p-value : ',pvalue)        
    
def Two_Paired_Samples_Wilcoxon_Sign_Median(df):
    st.subheader('<<< See Two_Paired_Samples_Wilcoxon_Sign_Median() function >>>\n\n\n')
    st.write("This test is a non parametric test used to compare median \
            value for two paired measurements")
    obs=df[df['Parcelle']=='Sud']
    obs=obs[['Hauteur','Hauteur.J7']]
    obs=obs.dropna()  
    st.write("Observed data :\n")
    st.write(obs)
    st.write("Median values : ",obs["Hauteur"].median(),obs["Hauteur.J7"].median())

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
        statistic,pvalue=wilcoxon(x=obs["Hauteur"],y=obs["Hauteur.J7"],zero_method='wilcox',
                                  correction=False,alternative="two-sided",mode='exact')
    st.write('statistic : ',statistic)
    st.write(' p-value : ',pvalue)    
    
def Two_Independent_Samples_Mann_Whitney_Median(df):
    st.subheader('<<< See Two_Independent_Samples_Mann_Whitney_Median() function >>>\n\n\n')    
    obs1=df[df['Parcelle']=='Nord']
    obs1=obs1[['Masse.grains']]
    obs1=obs1.dropna()  
    obs2=df[df['Parcelle']=='Sud']
    obs2=obs2[['Masse.grains']]
    obs2=obs2.dropna()  
    st.write('"Masse.grains" for "Parcelle"="Nord" : ')
    st.dataframe(obs1.sort_values(by="Masse.grains",ascending=True),height=200)
    st.write("Median : ",obs1.median())
    st.write('"Masse.grains" for "Parcelle"="Sud" : ')
    st.dataframe(obs2.sort_values(by="Masse.grains",ascending=True),height=200)
    st.write("Median : ",obs2.median())
    
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

def Several_Samples_Kruskal_Wallis_Median(df):
    st.subheader('<<< See Several_Samples_Kruskal_Wallis_Median() function >>>\n\n\n')    
    obs1=df[df['Parcelle']=='Nord']
    obs1=obs1[['Masse.grains']]
    obs1=obs1.dropna()  
    obs2=df[df['Parcelle']=='Sud']
    obs2=obs2[['Masse.grains']]
    obs2=obs2.dropna()  
    obs3=df[df['Parcelle']=='Est']
    obs3=obs3[['Masse.grains']]
    obs3=obs3.dropna()  
    obs4=df[df['Parcelle']=='Ouest']
    obs4=obs4[['Masse.grains']]
    obs4=obs4.dropna()  
    st.markdown('<u>obs1 : "Masse.grains" for "Parcelle"="Nord" : </u>',unsafe_allow_html=True) 
    st.dataframe(obs1.sort_values(by="Masse.grains",ascending=True),height=200)
    st.write("Median : ",obs1.median())
    st.markdown('<u>obs2 : "Masse.grains" for "Parcelle"="Sud" : </u>',unsafe_allow_html=True)    
    st.dataframe(obs2.sort_values(by="Masse.grains",ascending=True),height=200)
    st.write("Median : ",obs2.median())
    st.markdown('<u>obs3 : "Masse.grains" for "Parcelle"="Est" : </u>',unsafe_allow_html=True)
    st.dataframe(obs3.sort_values(by="Masse.grains",ascending=True),height=200)
    st.write("Median : ",obs3.median())
    st.markdown('<u>obs4 : "Masse.grains" for "Parcelle"="Ouest" </u>: ',unsafe_allow_html=True)    
    st.dataframe(obs4.sort_values(by="Masse.grains",ascending=True),height=200)
    st.write("Median : ",obs4.median())

    with st.echo():
        from scipy.stats import kruskal
        # INPUTS :
        #     - liste of input arrays as observations
        #     - nan_policy :       policy regarding nan values (propagate’, ‘raise’, ‘omit’)
        # OUTPUTS :
        #     - Statistic :        F value
        #     - pvalue :           p-value 
        statistic,pvalue=kruskal(obs1,obs2,obs3,obs4,
                                  nan_policy='omit')
    st.write('statistic : ',statistic)
    st.write(' p-value : ',pvalue)   
    st.write('Degrees of freedom (k-1) : ',3)

    st.write('This test can also be performed as a 50% quantile test, but for "Median" this particular \
            test is often more effective.')

def Quantiles(df):
    st.title("Test Median as 50% quantile")
    st.subheader('<<< See Quantiles() function >>>\n\n\n') 
    
    
    st.write("Currently, I didn't found a 'scipy' or 'statsmodel' method allowing to perform a quantile test \
            comparison between several observations.")
    st.write("The existing test, is the median test corresponding to 50% quantile test.")
    st.write("As detailed in reference [TBD], the median test method can be generalized to either quantiles.")
    st.write("Potentially a future work to be done.")
    st.write("Below, the test median for information (oten less effective than Kruskal Wallis test)")
    
    obs1=df[df['Parcelle']=='Nord']
    obs1=obs1['Masse.grains']
    obs1=obs1.dropna()  
    obs2=df[df['Parcelle']=='Sud']
    obs2=obs2['Masse.grains']
    obs2=obs2.dropna()  
    obs3=df[df['Parcelle']=='Est']
    obs3=obs3['Masse.grains']
    obs3=obs3.dropna()  
    obs4=df[df['Parcelle']=='Ouest']
    obs4=obs4['Masse.grains']
    obs4=obs4.dropna()  
    st.markdown('<u>obs1 : "Masse.grains" for "Parcelle"="Nord" : </u>',unsafe_allow_html=True) 
    st.dataframe(obs1.sort_values(ascending=True),height=200)
    st.write("Median : ",obs1.median())
    st.markdown('<u>obs2 : "Masse.grains" for "Parcelle"="Sud" : </u>',unsafe_allow_html=True)    
    st.dataframe(obs2.sort_values(ascending=True),height=200)
    st.write("Median : ",obs2.median())
    st.markdown('<u>obs3 : "Masse.grains" for "Parcelle"="Est" : </u>',unsafe_allow_html=True)
    st.dataframe(obs3.sort_values(ascending=True),height=200)
    st.write("Median : ",obs3.median())
    st.markdown('<u>obs4 : "Masse.grains" for "Parcelle"="Ouest" </u>: ',unsafe_allow_html=True)    
    st.dataframe(obs4.sort_values(ascending=True),height=200)
    st.write("Median : ",obs4.median())

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
        statistic,pvalue,m,table=median_test(obs1,obs2,obs3,obs4, ties='ignore',correction=False,
                                  nan_policy='omit')
    st.write('statistic : ',statistic)
    st.write(' p-value : ',pvalue)   
    #st.write('Degrees of freedom (k-1) : ',3)
    st.write('Grand median',m)
    table=pd.DataFrame(data=table.transpose(),index=['a','b','c','d'],columns=['False','True'])
    st.write("Contingency table :")
    st.dataframe(table)
    st.write("""This is the contingency table. The shape of the table is (2, n), where n is the number of samples. 
            The first row holds the counts of the values above the grand median, and the second row holds
            the counts of the values below the grand median. """)
    
def Two_Samples_Fisher_Snedecor_Parametric_Variances(df):
    st.title("Fisher-Snedecor Parametric Variance Test for 2 observations")
    st.subheader('<<< See Two_Samples_Fisher_Snedecor_Parametric_Variances() function >>>\n\n\n') 
        
    obs1=df[df['Parcelle']=='Nord']
    obs1=obs1['Hauteur']
    obs1=obs1.dropna()  
    obs2=df[df['Parcelle']=='Sud']
    obs2=obs2['Hauteur']
    obs2=obs2.dropna()     
    st.markdown('<u>obs1 : "Masse.grains" for "Parcelle"="Nord" : </u>',unsafe_allow_html=True) 
    st.dataframe(obs1,height=200)
    st.write("Variance : ",obs1.std()**2.)
    st.markdown('<u>obs2 : "Masse.grains" for "Parcelle"="Sud" : </u>',unsafe_allow_html=True)    
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
       
        
def Two_Samples_Ansary_Bradley_Non_Parametric_Variances(df):
    st.title("Ansary_Bradley non Parametric Variance Test for two observations")
    st.subheader('<<< See Two_Samples_Ansary_Bradley_Non_Parametric_Variances() function >>>\n\n\n') 
    
    obs1=df[df['Parcelle']=='Est']
    obs1=obs1['Masse.grains']
    obs1=obs1.dropna()  
    obs2=df[df['Parcelle']=='Ouest']
    obs2=obs2['Masse.grains']
    obs2=obs2.dropna()     
    st.markdown('<u>obs1 : "Masse.grains" for "Parcelle"="Ouest" : </u>',unsafe_allow_html=True) 
    st.dataframe(obs1,height=200)
    st.write("Variance : ",obs1.std()**2.)
    st.markdown('<u>obs2 : "Masse.grains" for "Parcelle"="Est" : </u>',unsafe_allow_html=True)    
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

    
def Several_Samples_Bartlett_Parametric_Variances(df):
    st.title("Bartlett Parametric Variance Test for several observations")
    st.subheader('<<< See Several_Samples_Bartlett_Parametric_Variances() function >>>\n\n\n') 
    
    obs1=df[df['Parcelle']=='Est']
    obs1=obs1['Masse']
    obs1=obs1.dropna()  
    obs2=df[df['Parcelle']=='Nord']
    obs2=obs2['Masse']
    obs2=obs2.dropna()  
    obs3=df[df['Parcelle']=='Ouest']
    obs3=obs3['Masse']
    obs3=obs3.dropna()  
    obs4=df[df['Parcelle']=='Sud']
    obs4=obs4['Masse']
    obs4=obs4.dropna() 
    st.markdown('<u>obs1 : "Masse" for "Parcelle"="Est" : </u>',unsafe_allow_html=True) 
    st.dataframe(obs1,height=200)
    st.write("Variance : ",obs1.std()**2.)
    st.markdown('<u>obs2 : "Masse" for "Parcelle"="Nord" : </u>',unsafe_allow_html=True)    
    st.dataframe(obs2,height=200)
    st.write("Variance : ",obs2.std()**2.)    
    st.markdown('<u>obs1 : "Masse" for "Parcelle"="Ouest" : </u>',unsafe_allow_html=True) 
    st.dataframe(obs3,height=200)
    st.write("Variance : ",obs3.std()**2.)
    st.markdown('<u>obs2 : "Masse" for "Parcelle"="Sud" : </u>',unsafe_allow_html=True)    
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
               
def Several_Samples_Fligner_Killeen_Non_Parametric_Variances(df):
    st.title("Fligner-Killeen Non Parametric Variance Test for several observations")
    st.subheader('<<< See Several_Samples_Fligner_Killeen_Non_Parametric_Variances() function >>>\n\n\n') 
    
    obs1=df[df['Parcelle']=='Est']
    obs1=obs1['Masse.grains']
    obs1=obs1.dropna()  
    obs2=df[df['Parcelle']=='Nord']
    obs2=obs2['Masse.grains']
    obs2=obs2.dropna()  
    obs3=df[df['Parcelle']=='Ouest']
    obs3=obs3['Masse.grains']
    obs3=obs3.dropna()  
    obs4=df[df['Parcelle']=='Sud']
    obs4=obs4['Masse.grains']
    obs4=obs4.dropna() 
    st.markdown('<u>obs1 : "Masse.grains" for "Parcelle"="Est" : </u>',unsafe_allow_html=True) 
    st.dataframe(obs1,height=200)
    st.write("Variance : ",obs1.std()**2.)
    st.markdown('<u>obs2 : "Masse.grains" for "Parcelle"="Nord" : </u>',unsafe_allow_html=True)    
    st.dataframe(obs2,height=200)
    st.write("Variance : ",obs2.std()**2.)    
    st.markdown('<u>obs1 : "Masse.grains" for "Parcelle"="Ouest" : </u>',unsafe_allow_html=True) 
    st.dataframe(obs3,height=200)
    st.write("Variance : ",obs3.std()**2.)
    st.markdown('<u>obs2 : "Masse.grains" for "Parcelle"="Sud" : </u>',unsafe_allow_html=True)    
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
    
    
                 
    
def Pearson_Linear_Parametric_Correlation(df):
    st.title("Pearson Parametric Linear Correlation")
    st.subheader('<<< See Pearson_Linear_Parametric_Correlation() function >>>\n\n\n') 
    
    st.subheader('H0: both oservations are NOT correlated')

    obs=df[df['Parcelle']=='Est']
    obs=obs[['Hauteur','Masse']]
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
    
def Spearman_Monotonous__Non_Parametric_Correlation(df):
    st.title("Spearman Non Parametric Monotonous Correlation")
    st.subheader('<<< See Spearman_Monotonous__Non_Parametric_Correlation() function >>>\n\n\n') 
    
    st.subheader('H0: both oservations are NOT correlated')

    obs=df[df['Parcelle']=='Est']
    obs=obs[['Hauteur','Masse.grains']]
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


def Kendall_Monotonous__Non_Parametric_Correlation(df):
    st.title("Kendall Non Parametric Monotonous Correlation")
    st.subheader('<<< See Kendall_Monotonous__Non_Parametric_Correlation() function >>>\n\n\n') 
    
    st.subheader('H0: both oservations are NOT correlated')

    obs=df[df['Parcelle']=='Est']
    obs=obs[['Hauteur','Masse.grains']]
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

def Anderson_Darling(df):
    st.title("Anderson-Darling")
    st.subheader('<<< See Anderson_Darling() function >>>\n\n\n') 
    
    st.subheader('H0: ................')

    obs=df[df['Parcelle']=='Est']
    obs=obs['Hauteur']
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
       
        
    
def Shapiro_Wilk(df):
    st.title("Shapiro-Wilk")
    st.subheader('<<< See Shapiro-Wilk() function >>>\n\n\n') 
    
    st.subheader('H0: data is normally distributed')

    obs=df[df['Parcelle']=='Est']
    obs=obs['Masse.grains']
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
 
          
def Dagostino_Pearson(df):
    st.title("D'Agostino-Pearson")
    st.subheader('<<< See Dagostino_Pearson() function >>>\n\n\n') 
    
    st.subheader('H0: data is normally distributed')

    obs=df[df['Parcelle']=='Est']
    obs=obs['Masse.grains']
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
                
    

def main():

    # reading datafile
    df=pd.read_csv("mais.txt",sep='\t')
    #st.dataframe(df,width=900) 
    #st.write(df.columns.tolist())
    
    st.title("STATISTICAL TEST TOOL CHOICE")
    
    st.subheader("Notations :")
    st.write('Q2  : qualitative variable with 2 independant classes')
    st.write('Qk  : qualitative variable with 2 independant classes or more')
    st.write('QA  : qualitative variable with 2 paired classes (for example repeated measures)')
    st.write('QT  : quantitiave variable (eventually qualitative ordinal variable)')
    st.write('NP  : Non-Parametric test (test that does not suppose normal distributions)')
    st.write('\n\n\n')
    st.subheader('In this tool, all alternatives H1 are considered to be Bilateral')
    
    menu=["Make your choice","By type of measure","By Dependent variable", "Other"]
    
    choice=st.sidebar.selectbox("MENU",menu)
    st.write('\n\n\n\n\n')
    
    if choice==menu[0]:
        #st.write('MAKE YOUR CHOICE')
        pass
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
                st.subheader("Khi2 Confirmity test for Count Measure")
                st.write('For 1 classes with 2 or more levels (1 x Qk), and a theoretical expected repartition')
                st.write('Is the repartition of the individus in the different levels compatible with \
                         expected repartition ?')
                st.write('\n\n\n\n\n')
                Khi2_conformity_count(df)
                
            elif choice3==menu3[2]:
                menu4=["Make your choice","YES","NO"]
                Cochran_Description()
                st.subheader("Is Cochran condition respected ?")
                choice4=st.selectbox("",menu4)
                if choice4==menu4[0]:
                    pass
                elif choice4==menu4[2]:
                    st.subheader("Fisher's Exact Test")
                    Fisher_test_cxk_Count(df)
                elif choice4==menu4[1]:
                    st.subheader("Khi2 homogeneity Test for Count measure")
                    st.write("For two classes with 2 or more levels (2 x Qk)")
                    st.write('Are the count for the different classes of both observation independent ?')
                    st.write('(i.e. with count compatible with a completly random split)')
                    Khi2_homogeneity_count(df)
                    st.write('\n\n\n\n\n')
                    st.subheader("G Test (based on log count) for Count Measure")
                    G_count(df)
                else:
                    st.write('coding error')
                    
            elif choice3==menu3[3]:
                st.write('\n\n\n\n') 
                st.write('\n\n\n\n')
                st.subheader("Mantel-Haenszel test for Count Measure")
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
                    st.subheader("Binomial Test")
                    Binomial_vs_Theory_Proportions(df)
                elif choice4==menu4[2]:
                    st.subheader("Khi2 Conformity test")
                    st.write('\n\n\n\n\n')
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
                        st.subheader("Khi2 Homogeneity test between 2 observations")
                        st.write("For two classes with 2 or more levels (2 x Qk)")
                        st.write('Are the proportion for the different classes of both observation independent ?')
                        st.write('(i.e. with proportion compatible with a completly random split)')
                        Khi2_homogeneity_proportion(df)
                    elif choice5==menu5[2]:
                        st.subheader("Fisher's exact test")
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
                        st.subheader("Binomial Test")
                        Binomial_Paired_Proportions(df)
                    elif choice5==menu5[2]:
                        st.subheader("Mac Nemar test")
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
                        st.subheader("Mantel-Haenszel Test")
                        Mantel_Haenszel_Proportion(df)
                    elif choice5==menu5[2]:
                        st.subheader("Khi2 Homogeneity Test between several observations")
                        st.write("For one classe with 2 levels and one class with (k=) 2 or more levels (1 x Qk)\
                                    and 1 x Q2)")
                        st.write('Are the proportion repartitions in the k-level class independent from \
                                the 2-level class ?')
                        Khi2_homogeneity_proportion_several(df)
                    else:
                        st.write('coding error')
                        
            elif choice3==menu3[4]:
                st.subheader("Khi2 test")
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
                st.subheader("Normal Distribution")
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
                st.subheader("Normal Distribution")
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
    else:
        st.write("coding error")
    
    st.sidebar.markdown("******") 
    st.sidebar.text("Version 1.0 14/05/2021")
    st.sidebar.text("By F. Bocage")
    st.sidebar.text("Contact : frederic_91@yahoo.fr")

    
    
    
if __name__=="__main__":
    main()