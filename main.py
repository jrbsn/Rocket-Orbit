import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from numpy import ndarray
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
import random

sims = 500000

gravitationalConstant = 6.67 * (10 ** -11)
airDensity = 1.225 # [kg/m^3] @ sea level  

# rocket parameters based off of Falcon 9
rocketMass = 549054 # [kg]
rocketRadius = 1.85 # [m]
rocketArea = 3.1415926 * rocketRadius**2
burnTime = 162 # [s]
dragCoefficient = 0.5 #   0.5 is used for simplicity

def createPlanet():
    # for reference... Earth's mass = 5.972 x 10E24 [kg] & radius = 4000 [mi]

    # Get the planet's mass
    number = random.randint(1,10)
    exponent = random.randint(23, 24)
    mass = number * (10 ** exponent) # [kg]
    # Get the planet's radius
    radius = random.randint(2000, 5000) # [mi]
    planetRadius = radius * 1609.34 # [m]
    # Orbit Radius
    orbitRadius = planetRadius * 1.1 # based off of space station orbit relative to earth

    return mass, planetRadius, orbitRadius

def findThrust(): # might need to base parameters relative to planetSize
    thrust = random.randint(100, 1800) * 100000 # [N]

    return thrust

def orbitalVelocity(planet):
    planetMass = planet[0]
    planetRadius = planet[1]
    orbitRadius = planet[2]

    # Find velocity 
    velocity = ((gravitationalConstant * planetMass) / orbitRadius) ** 0.5

    return velocity

def findAirDensity(planet, altitude):
    # simple nonlinear function to model the planet's air denisty similar to Earth
    # the rocket will reach a vaccuum once it hits 0.9 * the orbit altitude
    orbitRadius = planet[2]
    planetRadius = planet[1]
    orbitAltitude = orbitRadius - planetRadius

    if altitude <= orbitAltitude * 0.9:
        airD = (-1.225 * (altitude / (0.9 * orbitAltitude))) + 1.225
    else:
        airD = 0

    return airD

def findGravity(planet, altitude):
    orbitRadius = planet[2]
    planetRadius = planet[1]
    orbitAltitude = orbitRadius - planetRadius

    if altitude <= orbitAltitude * 0.95:
        gravity = (-9.81 * (altitude / (0.95 * orbitAltitude))) + 9.81
    else:
        gravity = 0

    return gravity

def flightModel(planet, thrust, requiredVelocity):
    
    # Separate function input variables: I do it this way to avoid numerous variables being inputted into the function

    planetRadius = planet[1]
    orbitRadius = planet[2]
    orbitAltitude = orbitRadius - planetRadius

    # Set starting model variables
    time = 0
    velocity = 0    # [m/s]
    acceleration = -9.81    # [m/s/s]
    altitude = 0    # [m]
    drag = 0

    deltaT = 0.5 # time interval
    # Run the model
    while altitude <= orbitAltitude and velocity >= 0:
        gravity = findGravity(planet, altitude)
        airDensity = findAirDensity(planet, altitude)
        weight = gravity * rocketMass # [N]

        drag = 0.5 * airDensity * (velocity**2) * dragCoefficient * rocketArea

        # Acceleration (thrust on during burnphase, off for the rest of flight)
        if time <= burnTime:
            acceleration = (thrust - drag - weight) / rocketMass
        else:
            acceleration = (-drag - weight) / rocketMass

        velocity += acceleration * deltaT
        altitude += velocity * deltaT + 0.5 * acceleration * deltaT**2
        
        time += deltaT
    
    if altitude >= orbitAltitude:
        return velocity
    else:
        return 0

def evaluateFlight(velocity, requiredVelocity):
    tolerance = 0.005   # 5%

    if velocity >= (requiredVelocity - requiredVelocity*tolerance) and velocity <= (requiredVelocity + requiredVelocity*tolerance): # success
        return 1
    else:
        return 0

def storeData(dataset, planet, thrust, velocity, requiredVelocity):
    planetMass = planet[0]
    planetRadius = planet[1]
    orbitRadius = planet[2]

    newData = np.array([[planetMass, planetRadius, orbitRadius, velocity, requiredVelocity, thrust]])
    return np.append(dataset, newData, axis=0)

def prepareData(dataset):
    dataset = np.delete(dataset, 0, axis=0) # remove row of zeros
    dataset = np.delete(dataset, 0, axis=1) # remove row of zeros
    columnNames = ["planetMass", "planetRadius", "orbitRadius", "velocity", "requiredVelcity", "thrust"]

    dataset = pd.DataFrame(dataset, columns=columnNames) # convert numpy array into pandas dataframe
    dataset.to_csv(r'C:\Users\josha\CSV\export_dataframe.csv')

    # Split Data
    X = dataset.drop('thrust', axis=1)
    Y = dataset['thrust']
    # Scale Data
    scaler = MaxAbsScaler()  # sklearn scaler
    X = scaler.fit_transform(X)

    return X, Y

def neuralNetwork(X, Y):
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    
    model = keras.Sequential([
        layers.Dense(10, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(5, activation='relu'),
        layers.Dense(5, activation='relu'),
        layers.Dense(1, activation='relu')
    ])
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.1), 
                    metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')])
    history = model.fit(X, Y, epochs=5, verbose=1)
    model.save('saved_model')
    loaded_model = tf.keras.models.load_model('saved_model')
   
    data = np.array([[0.9,  0.7554,  0.7554,  0.747778,  0.750377]])

    yhat = loaded_model.predict(data) #144400000
    print(yhat)
    return model
def save(sim, dataset):
    dataset = np.delete(dataset, 0, axis=0) # remove row of zeros
    columnNames = ["planetMass", "planetRadius", "orbitRadius", "velocity", "requiredVelcity", "thrust"]

    dataset = pd.DataFrame(dataset, columns=columnNames) # convert numpy array into pandas dataframe
    dataset.to_csv(r'C:\Users\josha\CSV\export_dataframe%s.csv' % sim)

def importCSV():
    fileCount = 100
    dataset = np.array([[0, 0, 0, 0, 0, 0, 0]])
    for file in range(fileCount):
        url = 'https://raw.githubusercontent.com/jrbsn/Rocket-Orbit/main/export_dataframe%i.csv' % file
        newData = pd.read_csv(url).to_numpy()
        dataset = np.vstack((dataset, newData))

    return dataset

    
def main():
    #dataset = np.array([[0, 0, 0, 0, 0, 0]]) # starting data
    dataset = importCSV()
    X, Y = prepareData(dataset)
    model = neuralNetwork(X.tolist(), Y.to_list())
    # 9000000000000000000000000,6078477.18,6686324.898,9428.46415645466,9475.244684258918,  ***  76100000
    #prediction = model.predict([9000000000000000000000000,6078477.18,6686324.898,9428.46415645466,9475.244684258918].tolist())
    #print(prediction)
    for epochsim in range(10000):
        break
        dataset = np.array([[0, 0, 0, 0, 0, 0]]) # starting data
        for sim in range(sims):
            planet = createPlanet()

            thrust = findThrust()
            requiredVelocity = orbitalVelocity(planet)
            velocity = flightModel(planet, thrust, requiredVelocity)
            result = evaluateFlight(velocity, requiredVelocity)
            if result == 1:
                dataset = storeData(dataset, planet, thrust, velocity, requiredVelocity)
            
            if sim % 10000 == 0:
                print("Sim # %i   Successes: %i" % (sim, dataset.shape[0]))
        # Use dataset for Neural Network Training
        #save(epochsim, dataset)
        #X, Y = prepareData(dataset)

main()
