#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:36:22 2018

@author: veronikasamborska
"""
import numpy as np
import pylab as plt
import matplotlib.pyplot as plot
import sys 
sys.path.append('/Users/veronikasamborska/Desktop/2018-12-12-Reversal_learning/code/reversal_learning/')
import utility as ut
import seaborn as sns
import pandas as pd

# Function to find where block transitions were for the session plosts 
def block_transsitions(prt): 
    ind = np.where(prt==0)[0]
    index= []
    for i in ind:
        if i<len(prt)-1:
            nexts = prt[i+1]
            if nexts != 0:
                index.append(i+1)
    for(i,j) in enumerate(index):
                    index[i]=j-1

    return index

# Reversals Plot -------------------------------------------------------------------------
def trials_till_reversal_plot(experiment, subject_IDs ='all' , fig_no=1):
    if subject_IDs == 'all':
        subject_IDs = experiment.subject_IDs
    sessions_block = []
    trials_from_prev_session = [] #List to hold data from all subjects
    tasks = 8 # Maximum number of tasks
    reversal_to_threshold = np.ones(shape=(9,tasks,21))
    reversal_to_threshold[:] = np.NaN 
    for n_subj, subject_ID in enumerate(subject_IDs):
        subject_sessions = experiment.get_sessions(subject_IDs=[subject_ID])
        task_number = 0 # Current task number
        reversal_number = 0 #
        previous_session_config = 0 
        subject_sessions = experiment.get_sessions(subject_ID)
        trials_from_prev_session = 0
        configuration = subject_sessions[0].trial_data['configuration_i']
        for j, session in enumerate(subject_sessions):
            # Check if the task is new 
            configuration = session.trial_data['configuration_i'] 
            if configuration[0]!= previous_session_config:
                reversal_number = 0
                task_number += 1
                trials_from_prev_session = 0
                previous_session_config = configuration[0]  
            # Find trials where reversals occured 
            sessions_block = session.trial_data['block']
            n_trials = session.trial_data['n_trials']
            Block_transitions = sessions_block[1:] - sessions_block[:-1] # Block transition
            reversal_trials = np.where(Block_transitions == 1)[0]
            # Find trials where threshold crossed.
            prt = (session.trial_data['pre-reversal trials'] > 0).astype(int)
            threshold_crossing_trials = np.where((prt[1:] - prt[:-1]) == 1)[0]
            n_reversals = len(reversal_trials)
            if len(reversal_trials) == 0:
                    trials_from_prev_session += n_trials
            else: 
                    for i, crossing_trial in enumerate(threshold_crossing_trials): 
                        if i< n_reversals:
                            if reversal_number <= 19:
                                if i == 0: # First element in the threshold_crossing_trials_list
                                    reversal_to_threshold[n_subj, task_number, reversal_number] = crossing_trial+trials_from_prev_session
                                    trials_from_prev_session = 0
                                elif (i > 0) and (i < n_reversals): # reversal occured.                     
                                    reversal_to_threshold[n_subj, task_number, reversal_number] = crossing_trial-reversal_trials[i-1]
                                reversal_number += 1  
                        else: # Revesal did not occur before end of session.
                            trials_from_prev_session = n_trials - reversal_trials[i-1]
                            
    # Exporting data as excel files using pandas to reversal_learning folder
    task1_pd_reversals= pd.DataFrame(data=reversal_to_threshold[:,1,:])
    task2_pd_reversals = pd.DataFrame(data=reversal_to_threshold[:,2,:])
    task3_pd_reversals = pd.DataFrame(data=reversal_to_threshold[:,3,:])
    task4_pd_reversals = pd.DataFrame(data=reversal_to_threshold[:,4,:])
    task5_pd_reversals = pd.DataFrame(data=reversal_to_threshold[:,5,:])
    task1_pd_reversals.to_csv('task1_pd_reversals.csv')
    task2_pd_reversals.to_csv('task2_pd_reversals.csv')
    task3_pd_reversals.to_csv('task3_pd_reversals.csv')
    task4_pd_reversals.to_csv('task4_pd_reversals.csv')
    task5_pd_reversals.to_csv('task5_pd_reversals.csv')
    mean_threshold=np.nanmean(reversal_to_threshold,axis = 0)
    x=np.arange(21)
    std_proportion=np.nanstd(reversal_to_threshold, axis = 0)
    sample_size=np.sqrt(9)
    std_err= std_proportion/sample_size
    sns.set()
    for i in range(tasks - 1): 
         plt.plot(i * 20 + x, mean_threshold[i + 1])
         plt.fill_between(i * 20 + x, mean_threshold[i + 1]-std_err[i + 1], mean_threshold[i + 1]+std_err[i + 1], alpha=0.2)
    plt.ylabel('Number of Trials Till Threshold ')
    #plt.title('m473, m474, m475')
    plt.xlabel('Reversal Number')
    
   
#Experiment plot of poke A or B following the choice of A or B per reversal -------------------------------------------------------------------------
def A_B_poke_reversal_plot(experiment,subject_IDs ='all', fig_no = 1): 
# =============================================================================
#   Define what you want to plot (Probability of A/B Poke, Number of A/B pokes on Rewared Trials,
#   Number of A/B pokes on Non-Rewared Trials or Number of A/B pokes on All Trials
#   Could also do Probability Plots for Rewarded vs Non-Rewarded Trials 
# =============================================================================
    probability_plot = False # True for probability and True for all_trials/rewareded_trials/non_rewarded_trials 
    all_trials = True # True for all_trials; If need a plot for number of pokes probability_plot = False 
    rewarded_trials = False # True for rewarded
    non_rewarded_trials = False # True for non-rewarded
    
# Find number of subjects 
    if subject_IDs == 'all':
        subject_IDs = experiment.subject_IDs
        n_subjects = len(subject_IDs)  
    else:
        n_subjects = len(subject_IDs)
    tasks = 6 # Maximum number of tasks completed 
    trial_count = 0 # Initiate trial count (used it for debugging)
    bad_pokes = np.zeros([n_subjects,tasks,20])  # Subject, task number, reversal number
    bad_pokes[:] = np.NaN # Fill it with NaNs 
    
    #Put all trials into one list for each subject
    for n_subj, subject_ID in enumerate(subject_IDs):
        subject_sessions = experiment.get_sessions(subject_ID)
        previous_session_config = 0 # Initiate previous configuration
        task_number = 0 # Initiate task number
        rev = 0 # Initiate reversal number
        wrong_count = 0 # Initiate count of wrong pokes
        outcomes_count = 0 # Initiate count of outcomes
        task = [] # Empty list to later append to task number from all sessions
        reversals = [] # Empty list to later append to reversal number from all sessions
        outcome = [] # Empty list to later append to outcomes on each trial from all sessions (rewarded = 1, non_rewarded = 0)
        all_sessions_wrong_ch = [] # Get the list with only trials that were treated as trials in task programme
        rewarded_trial = False # State for whether the trial was rewarded (initate at False)
        non_rewarded_trial = False # State for whether the trial was non-rewarded(initate at False)
        choice_state = False # State for whether the event happens in the choice state (initate at False)
        for j, session in enumerate(subject_sessions):
            trials=session.trial_data['trials']
            configuration = session.trial_data['configuration_i']
            sessions_block = session.trial_data['block']
            trials=session.trial_data['trials']
            Block_transitions = sessions_block[1:] - sessions_block[:-1] # Block transition
            reversal_trials = np.where(Block_transitions == 1)[0]
            outcomes= session.trial_data['outcomes']
            outcome_transitions = np.where(outcomes == 1)[0]
            poke_A = 'poke_'+str(session.trial_data['poke_A'][0])
            poke_B = 'poke_'+str(session.trial_data['poke_B'][0])
            sessions_block = session.trial_data['block']
            trials = session.trial_data['trials']
            trial_l = len(trials)
            session_wrong_choice =[]
            #session_trial_count=[] #Used it for debugging
            events = [event.name for event in session.events if event.name in ['choice_state', 'init_trial','sound_b_no_reward', 'sound_b_reward','sound_a_no_reward','sound_a_reward',poke_A, poke_B]]
            if configuration[0]!= previous_session_config:
                task_number += 1
                rev = 0
                previous_session_config = configuration[0]
            for trial in trials:
                task.append(task_number)
                reversals.append(rev)
                outcome.append(outcomes_count)
                #outcomes_count=0
                for reversal in reversal_trials:
                    if reversal == trial:
                        rev+=1
                for out in outcome_transitions:
                    if out == trial:
                        outcomes_count+=1                      
            for event in events:
                if event == 'choice_state':
                    session_wrong_choice.append(wrong_count)
                    #session_trial_count.append(trial_count) #Used it for debugging
                    wrong_count = 0
                    choice_state = True
                    rewarded_trial = False
                elif event == 'sound_b_reward':
                    if rewarded_trials == True: 
                        rewarded_trial = True
                        #trial_count+=1 #Used it for debugging
                elif event == 'sound_a_reward':
                    if rewarded_trials == True:
                        rewarded_trial = True
                        trial_count+=1
                elif event == 'sound_a_no_reward':
                    if non_rewarded_trials == True:
                        non_rewarded_trial = True
                        #trial_count+=1 #Used it for debugging
                elif event == 'sound_b_no_reward':
                    if non_rewarded_trials == True:
                        non_rewarded_trial = True
                        #trial_count+=1 #Used it for debugging
                elif event == poke_A : 
                    if choice_state == True:
                        prev_choice = 'Poke_A'
                        choice_state = False
                    elif choice_state == False:
                        if all_trials == True: 
                            if prev_choice == 'Poke_B': 
                                wrong_count += 1   
                        elif rewarded_trials == True:
                            if rewarded_trial == True and prev_choice == 'Poke_B': 
                                wrong_count += 1
                        elif non_rewarded_trials == True:
                            if non_rewarded_trial == True and prev_choice == 'Poke_B':
                                wrong_count += 1 
                elif event == poke_B :
                    if choice_state == True:
                        prev_choice = 'Poke_B'
                        choice_state = False
                    elif choice_state == False: 
                        if all_trials == True: 
                            if prev_choice == 'Poke_A': 
                                wrong_count += 1     
                        elif rewarded_trials == True:
                            if rewarded_trial == True and prev_choice == 'Poke_A': 
                                wrong_count += 1  
                        elif non_rewarded_trials ==True: 
                            if non_rewarded_trial == True and prev_choice == 'Poke_A':
                                wrong_count += 1 
                elif event == 'init_trial':   
                    choice_state = False  
            if j == 0: 
                all_sessions_wrong_ch = session_wrong_choice[0:trial_l]
            if j > 0: 
                all_sessions_wrong_ch +=session_wrong_choice[0:trial_l]
                
        # Change lists to numpy arrays       
        np_task = np.asarray(task)
        np_reversals = np.asarray(reversals)
        np_outcomes= np.asarray(outcome) # 1s on the trials that were rewarded 
        np_pokes = np.asarray(all_sessions_wrong_ch) 
        
        # Finding A/B pokes for each task (tn), and each reversal (rn) within the task
        for tn in range(tasks):
            if tn > 0:
                for rn in range(20):
                   a=np_pokes[(np_task==tn) & (np_reversals==rn)]
                   outcomes= np_outcomes[(np_task == tn) & (np_reversals == rn)] #For rewarded or non-rewarded trials
                   outcomes_sum_np=np.asarray(outcomes) #For rewarded trials
                   a_pokes = np.asarray(a)
                   if probability_plot == True:  #For probability calculations  
                       np_probability_index = np.where(a_pokes != 0) #For probability calculations
                       a_pokes[np_probability_index] = 1
                       mean_pokes=np.mean(a_pokes)
                   elif all_trials == True and probability_plot == False: 
                       a_pokes = np.asarray(a) #For both rewarded and non-rewarded trials
                       mean_pokes=np.mean(a_pokes) #For all trials
                   elif rewarded_trials == True:
                       outcomes_sum=sum(outcomes_sum_np)
                       mean_pokes=(sum(a_pokes))/outcomes_sum #outcomes_sum #num_zeros #outcomes_sum 
                   elif non_rewarded_trials == True:
                        num_zeros = (outcomes_sum_np == 0).sum()
                        mean_pokes=(sum(a_pokes))/num_zeros     
                   elif np.isinf(mean_pokes):
                       mean_pokes=0
                   bad_pokes[n_subj,tn,rn] = mean_pokes
    #task1_pd_ab= pd.DataFrame(data=bad_pokes[:,1,:])
    #task2_pd_ab = pd.DataFrame(data=bad_pokes[:,2,:])
    #task3_pd_ab = pd.DataFrame(data=bad_pokes[:,3,:])
    #task4_pd_ab = pd.DataFrame(data=bad_pokes[:,4,:])
    #task1_pd_ab.to_csv('task1_ab_reversals.csv')
    #task2_pd_ab.to_csv('task2_ab_reversals.csv')
    #task3_pd_ab.to_csv('task3_ab_reversals.csv')
    #task4_pd_ab.to_csv('task4_ab_reversals.csv')
    #Mean Pokes 
    mean_bad_pokes=np.nanmean(bad_pokes,axis = 0)
    x=np.arange(20)
    # Standard Errors 
    std_proportion=np.nanstd(bad_pokes, axis = 0)
    sample_size=np.sqrt(9)
    std_err= std_proportion/sample_size
    #Plotting
    for i in range(tasks - 1): 
        plt.plot(i * 20 + x, mean_bad_pokes[i + 1])
        plt.fill_between(i * 20 + x, mean_bad_pokes[i + 1]-std_err[i + 1], mean_bad_pokes[i + 1]+std_err[i + 1], alpha=0.2)
    plt.ylabel('Probability of A/B poke following A/B choice')
    #plt.title('Rewarded Trials')
    plt.legend(loc='upper right')
    plt.xlabel('Reversal')
                   
            
# Experiment plot of I pokes during the choice state when A or B should be chosen per reversal -------------------------------------------------------------------------
def I_poke_reversal_plot(experiment,subject_IDs ='all', fig_no = 1): 
    if subject_IDs == 'all':
        subject_IDs = experiment.subject_IDs
        n_subjects = len(subject_IDs)
    else:
        n_subjects = len(subject_IDs)
    reversals = []
    task = []
    task_number =0
    tasks=9
    bad_pokes = np.zeros([n_subjects,tasks,21])# subject, task number, reversal number
    bad_pokes[:] = np.NaN
    #Put all trials into one list for each subject
    for n_subj, subject_ID in enumerate(subject_IDs):
        subject_sessions = experiment.get_sessions(subject_ID)
        previous_session_config = 0
        task_number = 0
        rev=0
        trial = 0
        task=[]
        reversals =[]
        all_sessions_wrong_ch=[]
        prev_event_choice = False 
        period_before_ITI = False
        for j, session in enumerate(subject_sessions):
            trials=session.trial_data['trials']
            configuration = session.trial_data['configuration_i']
            sessions_block = session.trial_data['block']
            trials=session.trial_data['trials']
            Block_transitions = sessions_block[1:] - sessions_block[:-1] #block transition
            reversal_trials = np.where(Block_transitions == 1)[0]
            prev_event_choice = False
            poke_I = 'poke_'+str(session.trial_data['configuration_i'][0])
            poke_A = 'poke_'+str(session.trial_data['poke_A'][0])
            poke_B = 'poke_'+str(session.trial_data['poke_B'][0])
            events = [event.name for event in session.events if event.name in ['choice_state', 'period_before_iti', poke_I, poke_A, poke_B]]
            trials = [event.name for event in session.events if event.name in ['choice_state']]
            sessions_block = session.trial_data['block']
            trials=session.trial_data['trials']
            trial_l = len(trials)
            wrong_poke=0
            session_wrong_choice =[]    
            if configuration[0]!= previous_session_config:
                task_number += 1
                rev = 0
                previous_session_config = configuration[0]
            for trial in trials:
                task.append(task_number)
                reversals.append(rev)
                for reversal in reversal_trials:
                    if reversal == trial:
                        rev+=1  
            for event in events:
                if event == 'choice_state':
                    session_wrong_choice.append(wrong_poke)
                    prev_event_choice = True
                    period_before_ITI = True
                    wrong_poke = 0
                elif event == poke_I: 
                    if prev_event_choice == True and period_before_ITI== True:   
                        wrong_poke += 1
                elif event == 'period_before_iti':
                    period_before_ITI = False
            if j == 0: 
                all_sessions_wrong_ch = session_wrong_choice[0:trial_l]
            if j > 0: 
                all_sessions_wrong_ch +=session_wrong_choice[0:trial_l]
            np_task = np.asarray(task)
            np_reversals = np.asarray(reversals)
            np_pokes = np.asarray(all_sessions_wrong_ch)
        for tn in range(tasks):
            if tn > 0:
                for rn in range(21):
                    a=np_pokes[(np_task==tn) & (np_reversals==rn)]
                    a_pokes = np.asarray(a)
                    mean_pokes= np.nanmean(a_pokes)
                    bad_pokes[n_subj,tn,rn] = mean_pokes        
    mean_bad_pokes=np.nanmean(bad_pokes,axis = 0)
    std_proportion=np.nanstd(bad_pokes, axis = 0)
    sample_size=np.sqrt(9)
    std_err= std_proportion/sample_size
    x=np.arange(21)
    for i in range(tasks - 1): 
        plt.plot(i * 20 + x, mean_bad_pokes[i + 1])
        plt.fill_between(i * 20 + x, mean_bad_pokes[i + 1]-std_err[i + 1], mean_bad_pokes[i + 1]+std_err[i + 1], alpha=0.2)     
    plt.ylabel('Number of I pokes during Choice State')
    plt.xlabel('Reversal')
            
#Plot of pokes in order -------------------------------------------------------------------------
    
# Experiment plot of poke I in the period following trial initiation -------------------------------------------------------------------------
def I_poke_session_plot(experiment, subject_IDs ='all', fig_no = 1): 
    if subject_IDs == 'all':
        subject_IDs = experiment.subject_IDs
        #n_subjects = len(subject_IDs)
    wrong_poke=0
    correct_poke=0
    wrong=[]
    number_A_B = np.ones(shape=(9,30))
    number_A_B[:] = np.NaN 
    correct=[]
    number_I=np.ones(shape=(9,30))
    number_I[:] = np.NaN
    sessions=np.arange(1,31)
    prev_event_choice = False
      
    for n_subj, subject_ID in enumerate(subject_IDs):
        subject_sessions = experiment.get_sessions(subject_ID)
        wrong=[]
        correct=[]
        correct_poke = 0
        prev_event_choice = False
        period_before_ITI= False
        for j, session in enumerate(subject_sessions):
            wrong=[]
            correct=[]
            prev_event_choice = False
            poke_I = 'poke_'+str(session.trial_data['configuration_i'][0])
            poke_A = 'poke_'+str(session.trial_data['poke_A'][0])
            poke_B = 'poke_'+str(session.trial_data['poke_B'][0])
            events = [event.name for event in session.events if event.name in ['choice_state', 'period_before_iti', poke_I, poke_A, poke_B]]
            trials = [event.name for event in session.events if event.name in ['choice_state']]
            l_trials=len(trials)
            for event in events:
                if event == 'choice_state':
                    prev_event_choice = True
                    period_before_ITI = True
                    correct.append(correct_poke) 
                    wrong.append(wrong_poke)
                    wrong_poke = 0
                    correct_poke=0 
                elif event == poke_I: 
                    if prev_event_choice == True and period_before_ITI== True:   
                        wrong_poke += 1                     
                elif event == poke_A:  
                    if prev_event_choice == True and period_before_ITI== True:
                        correct_poke += 1
                elif event == poke_B:
                    if prev_event_choice == True and period_before_ITI== True:
                        correct_poke += 1
                elif event == 'period_before_iti':
                    period_before_ITI = False
            number_I[n_subj,j] = (sum(wrong)/l_trials)
            number_A_B[n_subj,j] = (sum(correct)/l_trials)            
    number_I_mean=np.nanmean(number_I,axis = 0)
    sns.set()
    std_proportion=np.nanstd(number_I_mean, axis = 0)
    sample_size=np.sqrt(9)
    std_err= std_proportion/sample_size
    plt.figure()
    plt.fill_between(sessions, number_I_mean-std_err, number_I_mean+std_err, alpha=0.2, facecolor='r')
    plt.plot(sessions,number_I_mean,'r')
    plt.ylabel('Proportion of I to A/B pokes during the choice period')
    plt.xlabel('Session') 
    
# Experiment plot of poke A or B following the choice of A or B -------------------------------------------------------------------------
def session_A_B_poke_exp(experiment,subject_IDs ='all', fig_no = 1): 
    if subject_IDs == 'all':
        subject_IDs = experiment.subject_IDs
        #n_subjects = len(subject_IDs)
    wrong_choice=[]
    prev_choice=[]
    wrong_count=0
    choice_state = False
    wrong_ch = np.ones(shape=(10,30))
    wrong_ch[:] = np.NaN 
    sessions=np.arange(1,31)
    for n_subj, subject_ID in enumerate(subject_IDs):
        subject_sessions = experiment.get_sessions(subject_ID)
        wrong_choice=[]
        for j, session in enumerate(subject_sessions):  
            wrong_choice=[]
            poke_A = 'poke_'+str(session.trial_data['poke_A'][0])
            poke_B = 'poke_'+str(session.trial_data['poke_B'][0])
            events = [event.name for event in session.events if event.name in ['choice_state', 'init_trial', poke_A, poke_B]]
            trials = [event.name for event in session.events if event.name in ['choice_state']]
            l_trials=len(trials)
            for event in events:
                if event == 'choice_state':
                    wrong_choice.append(wrong_count)
                    wrong_count = 0
                    choice_state = True
                elif event == poke_A : 
                    if choice_state == True:
                        prev_choice = 'Poke_A'
                        choice_state = False
                    elif choice_state == False:
                        if prev_choice == 'Poke_B': 
                            wrong_count += 1                   
                elif event == poke_B :
                    if choice_state == True:
                        prev_choice = 'Poke_B'
                        choice_state = False
                    elif choice_state == False: 
                        if prev_choice == 'Poke_A': 
                            wrong_count += 1
                elif event == 'init_trial':   
                    choice_state = False
                    wrong_count = 0 
            wrong_ch[n_subj,j] = (sum(wrong_choice)/l_trials)
    wrong_ch_mean = np.nanmean(wrong_ch, axis = 0)
    std_dev=np.nanstd(wrong_ch, axis = 0)
    sample_size=np.sqrt(9)
    std_err= std_dev/sample_size
    #plt.errorbar(sessions, wrong_ch_mean, yerr = std_err)
    plt.fill_between(sessions, wrong_ch_mean-std_err, wrong_ch_mean+std_err, alpha=0.2, facecolor='b')
    plt.plot(sessions,wrong_ch_mean,'b')
    plt.ylabel('Number of A/B poke following A/B choice ')
    plt.xlabel('Session') 
    

 # Session plot of reversals-------------------------------------------------------------------------  
def session_plot_moving_average(session, fig_no = 1, is_subplot = False):
    block=session.trial_data['block']
    'Plot reward probabilities and moving average of choices for a single session.'
    if not is_subplot: plt.figure(fig_no, figsize = [7.5, 1.8]).clf()
    Block_transitions = block[1:]-block[:-1]
    choices = session.trial_data['choices']
    #threshold = block_transsitions(session.trial_data['pre-reversal trials']) # If you want threshold crossed on the plot 
    index_block = []
    for i in Block_transitions:
        index_block = np.where(Block_transitions == 1)[0]
    for i in index_block:
        plot.axvline(x = i,color = 'g',linestyle = '-', lw = '0.6')
    #for i in threshold: # If you want threshold crossed on the plot 
    #    plot.axvline(x=i,color='k',linestyle='--', lw='0.6')
    plot.axhline(y = 0.25, color = 'r', lw = 0.1)
    plot.axhline(y = 0.75, color = 'r', lw = 0.1)
        
    exp_average= ut.exp_mov_ave(choices,initValue = 0.5,tau = 8)
    plt.plot(exp_average,'--')
    plt.ylim(-0,1)
    plt.xlim(1,session.trial_data['n_trials'])
    if not is_subplot: 
        plt.ylabel('Exp moving average ')
        plt.xlabel('Trials') 
   

# Experiment plot -------------------------------------------------------------------------
def experiment_plot(experiment, subject_IDs = 'all', when = 'all', fig_no = 1,
                   ignore_bc_trials = False):
    'Plot specified set of sessions from an experiment in a single figure'
    if subject_IDs == 'all': 
        subject_IDs = experiment.subject_IDs
        plt.figure(fig_no, figsize = [11.7,8.3]).clf()
    else:
        plt.figure(fig_no,figsize = [11,1])
    n_subjects = len(subject_IDs)
    for i,subject_ID in enumerate(subject_IDs):
        subject_sessions = experiment.get_sessions(subject_ID, when)
        n_sessions = len(subject_sessions)
        for j, session in enumerate(subject_sessions):
            plt.subplot(n_subjects, n_sessions, i*n_sessions+j+1)
            session_plot_moving_average(session, is_subplot=True)
            if j == 0: plt.ylabel(session.subject_ID)
            if i == (n_subjects -1): plt.xlabel(str(session.datetime.date()))
    plt.tight_layout(pad=0.3)