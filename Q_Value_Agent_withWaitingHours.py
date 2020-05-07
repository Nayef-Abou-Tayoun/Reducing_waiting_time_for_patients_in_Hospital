"""
Created on Mar 19 2020
@author: Jerry
"""

import numpy as np
import random
import pandas as pd
from typing import Dict

import state

from state import State
from hospital_env_withWaitingHours import env
import json

models = ['fixed_policy', 'random_policy', 'max_qValue_policy']


class agent():
    """
       Defines a state-action table that can be used to store Q-values or action counts.
       """

    q_table: Dict[str, Dict[int, float]]  # Q-value table   [Action, q-value]
    state_action_counts: Dict[str, Dict[int, int]]

    #    state_action_counts: Dict[State, Dict[int, int]]  # The number of updates to each state-action pair

    def __init__(self, alpha=0.1, gamma=1.0, numGames=10):
        print('Created an Agent ...')
        self.actions = [-3, -2, -1, 0, 1, 2, 3]
        self.q_table = dict()
        self.state_action_counts = dict()
        self.alpha = alpha
        self.gamma = gamma
        self.numGames = numGames
        self.totalReward = dict()
        self.totalDischarged = dict()
        self.totalDischargedtoGenerated = dict()
        self.waiting_time_dict = dict()
        print('Created an Agent ...END')

    def getGreedyPolicy(self):
        COLUMN_NAMES = ['ED_patients', 'S_medical', 'Medical_patients', 'S_surgical', 'Surgery_patients',
                        'A_move_From_Medical_To_Surgical', 'Q_val']
        df = pd.DataFrame(columns=COLUMN_NAMES)
        for pk, pv in self.q_table.items():
            state = json.loads(pk)
            action = max(pv, key=pv.get)
            df = df.append({'ED_patients': state.get('ED_patients'),
                            'S_medical': state.get('medical_doctors'),
                            'S_surgical': state.get('surgery_doctors'),
                            'Surgery_patients': state.get('surgery_patients'),
                            'Medical_patients': state.get('medical_patients'),
                            'A_move_From_Medical_To_Surgical': action,
                            'Q_val': pv.get(action)},
                           ignore_index=True)
        return df

    def getQValueTable(self):
        COLUMN_NAMES = ['ED_patients', 'S_medical', 'Medical_patients', 'S_surgical', 'Surgery_patients',
                        'A_move_From_Medical_To_Surgical', 'Q_val']
        df = pd.DataFrame(columns=COLUMN_NAMES)
        for pk, pv in self.q_table.items():
            for k, v in pv.items():
                state = json.loads(pk)
                df = df.append({'ED_patients': state.get('ED_patients'),
                                'S_medical': state.get('medical_doctors'),
                                'S_surgical': state.get('surgery_doctors'),
                                'Surgery_patients': state.get('surgery_patients'),
                                'Medical_patients': state.get('medical_patients'),
                                'A_move_From_Medical_To_Surgical': k,
                                'Q_val': v},
                               ignore_index=True)
        return df

    def maxAction(self, state: State, actions, model: str):
        if model.casefold() == 'fixed_policy'.casefold():
            return 0
        elif model.casefold() == 'random_policy'.casefold():
            return random.choice(actions)

        if self.q_table is None:
            return random.choice(actions)
        elif self.q_table.get(state.__str__()) is None:
            return random.choice(actions)
        else:  # found one state corresponding actions and values
            if len(self.q_table[state.__str__()]) == len(actions):
                # get max value actions
                # valuesPairs = np.array(self.q_table[state.__str__()])
                action = max(self.q_table[state.__str__()], key=self.q_table[state.__str__()].get)
                return action
            else:
                newAction = list()
                keys = self.q_table[state.__str__()].keys()
                for action in actions:
                    if action not in keys:
                        newAction.append(action)
                return random.choice(newAction)

    def getValueFromStateAndAction(self, state: State, action, reward: int = 0, isExecutedAction: bool = False):
        if self.q_table.get(state.__str__()) is None:
            q_act_dict = dict()
            q_act_dict[action] = reward
            if isExecutedAction:
                self.q_table[state.__str__()] = q_act_dict
        else:
            if self.q_table[state.__str__()].get(action) is None:
                if isExecutedAction:
                    self.q_table[state.__str__()][action] = reward

        if (self.q_table.get(state.__str__()) is None) or (self.q_table.get(state.__str__()).get(action) is None):
            return 0
        else:
            return self.q_table.get(state.__str__()).get(action)

    def updateStateActionCounts(self, state: State, action: int):
        if self.state_action_counts.get(state.__str__()) is None:
            # First time tried this action for this state, initial dict
            q_count_dict = dict()
            q_count_dict[action] = 1
            self.state_action_counts[state.__str__()] = q_count_dict
        elif self.state_action_counts[state.__str__()].get(action) is None:
            # First time tried this action for this state, update count for this action
            self.state_action_counts[state.__str__()][action] = 1
        else:
            self.state_action_counts[state.__str__()][action] = self.state_action_counts[state.__str__()].get(
                action) + 1

    def getCountsFromStateAndAction(self, state: State, action: int):
        if self.state_action_counts.get(state.__str__()) is None:
            return 1
        elif self.state_action_counts[state.__str__()].get(action) is None:
            return 1
        else:
            return self.state_action_counts[state.__str__()][action]

    def run(self, alpha: float, gamma: float, numGames: int, env: env, model: str = 'max_qValue_policy'):
        assert (model in models)
        seed = 41

        self.alpha = alpha
        self.gamma = gamma
        self.numGames = numGames
        self.totalReward = np.zeros(numGames)

        for i in range(self.numGames):
            print('Starting game ', i)
            initial_patients = env.surgery_patients + env.ED_patients + env.medical_patients + sum(
                env.patients_list_sim)
            epsReward = 0
            epsWaitingTime = 0
            seed += 1
            epsDischargePatients = 0
            terminal = False
            state = env.reset(seed)
            print(type(state))
            while not terminal:
                # choosing the max action
                action = self.maxAction(state, self.actions, model)
                state_, terminal, reward, dischargePatients, waiting_time = env.update_doctor(action)
                epsReward += reward
                epsDischargePatients += dischargePatients
                action_ = self.maxAction(state_, self.actions, model)
                value = self.getValueFromStateAndAction(state, action, isExecutedAction=True) + alpha * (
                        reward + gamma * self.getValueFromStateAndAction(state_,
                                                                         action_) - self.getValueFromStateAndAction(
                    state, action)
                )

                count = self.getCountsFromStateAndAction(state, action)
                updatedValue = value / 2

                # updating Q table
                self.q_table[state.__str__()][action] = updatedValue
                # updating action counts table
                self.updateStateActionCounts(state, action)

                state = state_
                epsWaitingTime += waiting_time
            self.totalReward[i] = epsReward
            # store total discharged for each game
            self.waiting_time_dict[i] = epsWaitingTime
            self.totalDischarged[i] = epsDischargePatients
            self.totalDischargedtoGenerated[i] = epsDischargePatients/initial_patients

        print('training end')

    def reset(self):
        print('Reset agent ...')
        self.q_table = dict()
        self.state_action_counts = dict()
        self.totalReward = np.zeros(self.numGames)
        print('Reset agent ...END')
