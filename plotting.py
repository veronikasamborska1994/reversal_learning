#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:36:22 2018

@author: veronikasamborska
"""

import numpy as np
import pylab as plt
from scipy.stats import fisher_exact
import matplotlib.pyplot as plot
import collections as cl
import sys 
sys.path.append('/Users/veronikasamborska/Desktop/2018-12-12-Reversal_learning/code/reversal_learning/')
import plotting as pl 
import data_import as di
import utility as ut
import seaborn as sns
# Session plot ------------------------------------------------------------------------------

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
#Reversals Plot
def session_reversals_plot(experiment, subject_IDs ='all' , fig_no=1):
    if subject_IDs == 'all':
        subject_IDs = experiment.subject_IDs
    sessions_block = []
    trials_from_prev_session = []
    tasks = 8
    reversal_to_threshold = np.ones(shape=(9,tasks,21))
    reversal_to_threshold[:] = np.NaN   
    for n_subj, subject_ID in enumerate(subject_IDs):
        task_number = 0
        reversal_number = 0
        previous_session_config = 0
        subject_sessions = experiment.get_sessions(subject_ID)
        trials_from_prev_session = 0

        for j, session in enumerate(subject_sessions):
            sessions_block = session.trial_data['block']
            n_trials = session.trial_data['n_trials']
            # Find trials where reversals occured
            Block_transitions = sessions_block[1:] - sessions_block[:-1]#block transition
            reversal_trials = np.where(Block_transitions == 1)[0]
            # Find trials where threshold crossed.
            prt = (session.trial_data['pre-reversal trials'] > 0).astype(int)
            threshold_crossing_trials = np.where((prt[1:] - prt[:-1]) == 1)[0]
            n_reversals = len(reversal_trials)
            configuration = session.trial_data['configuration_i']
            if configuration[0]!= previous_session_config:
                reversal_number = 0
                task_number += 1
                trials_from_prev_session = 0
                previous_session_config = configuration[0]
            if not len(reversal_trials) > 0:
                    trials_from_prev_session += n_trials
            else: 
                    for i, crossing_trial in enumerate(threshold_crossing_trials): 
                        if reversal_number <= 20:
                            if i == 0:#first element in the threshold_crossing_trials_list
                                reversal_to_threshold[n_subj, task_number, reversal_number] = crossing_trial+trials_from_prev_session
                                trials_from_prev_session = 0
                            elif (i>0) and (i < n_reversals): # reversal occured.                     
                                reversal_to_threshold[n_subj, task_number, reversal_number] = crossing_trial-reversal_trials[i-1]
                            reversal_number += 1  
                        else: # revesal did not occur before end of session.
                            trials_from_prev_session = n_trials - reversal_trials[i-1]
    mean_threshold=np.nanmean(reversal_to_threshold,axis = 0)
    x=np.arange(21)
    print(mean_threshold[7,:])
    for i in range(tasks - 1): 
         plt.plot(i * 20 + x, mean_threshold[i + 1])
    plt.ylabel('Trials till threshold')
    plt.xlabel('reversal number')
          
#Plot of pokes in order 
#Session plot of poke I in the period following trial initiation
def session_I_poke(session):
    wrong_poke=0
    correct_poke=0
    wrong=[]
    correct=[]
    #choosing_I[:] = np.NaN
    #choosing_A_B[:] = np.NaN
    poke_I = 'poke_'+str(session.trial_data['configuration_i'][0])
    poke_A = 'poke_'+str(session.trial_data['poke_A'][0])
    poke_B = 'poke_'+str(session.trial_data['poke_B'][0])
    prev_event_choice = False
    #Event list only includes choice_state, init_trial and poke in events 
    events_I = [event.name for event in session.events if event.name in ['choice_state', 'period_before_iti', poke_I]]
    for i, event in enumerate(events_I):
        if i < (len(events_I)-1):
            if event == 'choice_state':
                prev_event_choice = True
                wrong_poke = 0
            elif event == poke_I: 
                if prev_event_choice == True:   
                    wrong_poke += 1
                    wrong.append(wrong_poke)                    
            elif event == 'period_before_iti':
                prev_event_choice == False                 
                
    number_I = (len(wrong))
    print(number_I)
    events_A_B = [event.name for event in session.events if event.name in ['choice_state', 'period_before_iti', poke_A, poke_B]]
    for i, event in enumerate(events_A_B):
        if i < (len(events_A_B)-1):
            if event == 'choice_state':
                prev_event_choice = True
                
            elif event == poke_A or event == poke_B: 
                if prev_event_choice == True:
                    correct_poke += 1
                    correct.append(correct_poke)                    
            elif events_A_B[i+1] == 'period_before_iti':
                    prev_event_choice == False 
    number_A_B = (len(correct))
    print(number_A_B)
    proportion_incorrect_correct=number_I/number_A_B
    print(proportion_incorrect_correct)
    
#Experiment plot of poke I in the period following trial initiation    
def session_I_poke_exp(experiment, subject_IDs ='all', fig_no = 1): 
    if subject_IDs == 'all':
        subject_IDs = experiment.subject_IDs
        #n_subjects = len(subject_IDs)
    wrong_poke=0
    correct_poke=0
    proportion_incorrect_correct= 0
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
            print(l_trials)
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
                
    #mean_proportion=np.nanmean(number_I,axis = 0)
    number_I_mean=np.nanmean(number_I,axis = 0)
    number_A_B_mean = np.nanmean(number_A_B,axis = 0)
    proportion_incorrect_correct=number_I_mean/number_A_B_mean
    sns.set()
    #std_proportion=np.nanstd(number_I, axis = 0)
    std_proportion=np.nanstd(number_I_mean, axis = 0)
    sample_size=np.sqrt(9)
    std_err= std_proportion/sample_size
    plt.figure()
    plt.fill_between(sessions, number_I_mean-std_err, number_I_mean+std_err, alpha=0.2, facecolor='r')
    plt.plot(sessions,number_I_mean,'r')
    plt.ylabel('Proportion of I to A/B pokes during the choice period')
    plt.xlabel('Session') 
    
#Experiment plot of poke A or B following the choice of A or B  
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
    
    
    
# Single session for number of poke A or B following A or B choice                                     
def session_A_B_poke_sess(session, fig_no = 1): 
    wrong_choice=[]
    prev_choice=[]
    wrong_count=0
    choice_state = False
    #choosing_I[:] = np.NaN
    #choosing_A_B[:] = np.NaN
    #poke_I = 'poke_'+str(session.trial_data['configuration_i'][0])
    poke_A = 'poke_'+str(session.trial_data['poke_A'][0])
    poke_B = 'poke_'+str(session.trial_data['poke_B'][0])
    #Event list only includes choice_state, init_trial and poke in events 
    events = [event.name for event in session.events if event.name in ['choice_state', 'init_trial', poke_A, poke_B]]
    trials = [event.name for event in session.events if event.name in ['choice_state']]
    l_trials=len(trials)
    print(l_trials)
    for event in events:
            if event == 'choice_state':
                wrong_choice.append(wrong_count)
                print(wrong_choice)
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
    wrong_ch = (sum(wrong_choice)/l_trials)
    print(wrong_ch)
    
 # Session plot of reversals    
def session_plot_moving_average(session, fig_no = 1, is_subplot = False):
    block=session.trial_data['block']
    'Plot reward probabilities and moving average of choices for a single session.'
    if not is_subplot: plt.figure(f.ig_no, figsize = [7.5, 1.8]).clf()
    Block_transitions = block[1:]-block[:-1]
    choices = session.trial_data['choices']
    threshold = block_transsitions(session.trial_data['pre-reversal trials'])# threshold
    index_block = []
    for i in Block_transitions:
        index_block = np.where(Block_transitions == 1)[0]
    for i in index_block:
        plot.axvline(x = i,color = 'g',linestyle = '-', lw = '0.6')
    #for i in threshold:
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