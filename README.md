# JavaFAWT
JavaFAWT is a fan-array-wind tunnel (FAWT) open-source software and hardware architecture developed by the drones lab of Universidad Pública de Navarra. It includes diagrams and PCB layouts for electronics, CAD designs for hardware, and source files for running and adapting the software. 
## Introduction
### Features
* Modular low-cost FAWT architecture.
* Individual or multiple fan speed control through a simple unified interface.
* Websockets protocol implemented for wireless communication with each fan module.
### File structure
The files are organized as follows:

## How to install & run
Detailed instructions for installing and running the software.

## Files structure
|
|- Documentation
  |- private
    |- docs -> .md files for generating public documentation site.
    |- site -> compiled documentation website for dev purposes.
  |- public -> compiled documentation website for public purposes.
|- Examples -> provided examples.
|- Hardware
  |- module -> All the components required to assemble a single module.
  |- testing_equipment -> Other components developed for testing.
|- Software
  |- clientCode -> Python PWM client
  |- functionalities -> example functionalities provided for automatic fan control.
  |- pythonCode -> Python client file for controlling the fans from the Raspberry Pi.
  |- serverCode -> JavaFAWT server software.
  
## Work in progress
* Fan speed feedback is currenly under development.
  
## Acknowledgments
