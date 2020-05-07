"""
Created on Mar 16 2020
@author: Jerry
"""
import numpy as np
from state import State
import random

Hospital_Actions = [-3, -2, -1, 0, 1, 2, 3]
Patients_Medical_Discharge_Rate = 0.9
Patients_Surgery_Discharge_Rate = 0.7
Patients_DM_to_Medical = 0.6
Patients_DM_to_Surgery = 0.4
Max_Doctor_At_Each_Department = 12
Min_Doctor_At_Each_Department = 2
Max_Patients_At_Each_Department = 12


class env():

    def __init__(self, num_doctors=10, distribution_patient=5):
        print("Creating Hospital environment...")
        self.ED_patients = int(np.random.random_integers(0, 5))
        # restriction for number of doctors in total
        self.num_doctors = num_doctors
        # Medical department
        self.medical_patients = int(np.random.random_integers(0, 2))
        self.medical_doctors = int(np.random.random_integers(1, 5))
        # Surgery department
        self.surgery_patients = int(np.random.random_integers(0, 3))
        self.surgery_doctors = self.num_doctors - self.medical_doctors
        # discharge department
        self.discharge_patients = 0
        self.totalReward = 0
        # hour setting
        self.num_hours = 23
        self.current_hour = 0
        self.terminal = False
        self.distribution_patient = distribution_patient
        # generating a list of number of patients will come to the hospital based on poisson distribution
        self.patients_list_sim = self.generate_patients(self.distribution_patient)
        print("Creating Hospital environment...END")

    def generate_patients(self, distribution_patient: int, seed: int = 104):
        # generate a list of 24 hourly patients coming in
        # based on poisson distribution
        np.random.seed(seed)
        patients_list = np.random.poisson(distribution_patient, 24)

        return patients_list

    def update_doctor(self, add_doctors: int):
        assert (add_doctors in Hospital_Actions)

        epRewards = 0
        Total_waiting = 0
        epDischargePatients = 0

        """
        Restriction: 
        Max # of doctors at each department is 8
        Min # of doctors at each department is 1
        Otherwise, the action is invalid, the number of doctors at each department will keep unchanged
        """
        if Min_Doctor_At_Each_Department <= (self.medical_doctors + add_doctors) <= Max_Doctor_At_Each_Department \
                and Min_Doctor_At_Each_Department <= (
                self.surgery_doctors - add_doctors) <= Max_Doctor_At_Each_Department:
            self.medical_doctors = self.medical_doctors + add_doctors
            self.surgery_doctors = self.surgery_doctors - add_doctors

            # move one doctor cause rewards -3
            # epRewards += (abs(add_doctors) * -3)
        else:
            print("Invalid movement, the model will keep unchanged")
            # epRewards += -5
            # print("reward -5 since the agent provided invalid action")

        # updating how many patients will be discharged
        dischargeMedical = self.getNumPatientsDischanged(self.medical_doctors, self.medical_patients,
                                                         Patients_Medical_Discharge_Rate)
        dischargeSurgery = self.getNumPatientsDischanged(self.surgery_doctors, self.surgery_patients,
                                                         Patients_Surgery_Discharge_Rate)
        # updating Treatment room
        self.medical_patients -= dischargeMedical
        self.surgery_patients -= dischargeSurgery

        epDischargePatients += dischargeMedical
        epDischargePatients += dischargeSurgery

        # from generating patients list, retrieve the number of patients will come in and move to ED Department
        newPatients = self.patients_list_sim[self.current_hour]
        self.ED_patients += newPatients

        numPatientsMedical = int(
            self.ED_patients * Patients_DM_to_Medical)  # number of patients going to medical department
        numPatientsSurgery = int(
            self.ED_patients * Patients_DM_to_Surgery)  # number of patients going to surgery department
        """
        Restriction: 
        Max # of patients at each department is 10
        Patients beyond this number will be stay at ED room
        """

        if (self.medical_patients + numPatientsMedical) > Max_Patients_At_Each_Department:
            numPatientsMedical -= (
                    Max_Patients_At_Each_Department - self.medical_patients)  # update numPatientsMedical
            self.medical_patients = Max_Patients_At_Each_Department  # update medical_patients to Max number
        else:
            self.medical_patients += numPatientsMedical  # update medical_patients based on numPatientsMedical

        if (self.surgery_patients + numPatientsSurgery) > Max_Patients_At_Each_Department:
            numPatientsSurgery -= (
                    Max_Patients_At_Each_Department - self.surgery_patients)  # update numPatientsSurgery
            self.surgery_patients = Max_Patients_At_Each_Department  # update surgery_patients to Max number
        else:
            self.surgery_patients += numPatientsSurgery  # update surgery_patients based on numPatientsMedical

        # updating ED patients number
        self.ED_patients = self.ED_patients - (numPatientsMedical + numPatientsSurgery)

        self.discharge_patients += epDischargePatients  # updating model attributes

        # # each patient going to discharge will get rewards +4
        # epRewards += (epDischargePatients * 4)

        # each patients at ED room will get reward -2
        epRewards += (self.ED_patients * -1)
        epRewards += (self.medical_patients * -2)
        epRewards += (self.surgery_patients * -3)
        Total_waiting += (self.ED_patients * 1)
        Total_waiting += (self.medical_patients * 1)
        Total_waiting += (self.surgery_patients * 1)
        epRewards += epDischargePatients * 2
        self.totalReward += epRewards

        # changing the current hour
        self.current_hour += 1
        # determine the terminal state
        if self.current_hour == self.num_hours:
            self.terminal = True
        return State(self.medical_patients, self.medical_doctors, self.surgery_patients,
                     self.surgery_doctors, int(self.ED_patients)), self.terminal, epRewards, epDischargePatients, Total_waiting

    def getNumPatientsDischanged(self, numberDoctors: int, numberPatients: int, probability: float):
        count = 0
        if numberDoctors >= numberPatients:
            for patient in range(numberPatients):
                if self.decision(probability):
                    count += 1
        else:
            for patient in range(numberDoctors):
                if self.decision(probability):
                    count += 1

        return count

    def decision(self, probability):
        return random.random() < probability

    def getTotalRewards(self):
        return self.totalReward

    def getState(self):
        return State(self.medical_patients, self.medical_doctors, self.surgery_patients, self.surgery_doctors,
                     int(self.ED_patients))

    def reset(self, seed: int = 104):
        print("Reset Environment ...")
        np.random.seed(seed)
        self.ED_patients = int(np.random.random_integers(5, 12))
        # Medical department
        self.medical_patients = int(np.random.random_integers(5, 10))
        self.medical_doctors = int(np.random.random_integers(2, 8))
        # Surgery department
        self.surgery_patients = int(np.random.random_integers(3, 6))
        self.surgery_doctors = self.num_doctors - self.medical_doctors
        # discharge department
        self.discharge_patients = 0
        self.totalReward = 0
        self.patients_list_sim = self.generate_patients(self.distribution_patient, seed)
        # hour setting
        self.num_hours = 23
        self.current_hour = 0
        self.terminal = False
        print("Reset Environment ...END")
        return State(self.medical_patients, self.medical_doctors, self.surgery_patients, self.surgery_doctors,
                     int(self.ED_patients))
