# JavaFAWT
JavaFAWT is a fan-array-wind tunnel (FAWT) open-source software and hardware architecture developed by the drones lab of Universidad Pública de Navarra. It includes diagrams and PCB layouts for electronics, CAD designs for hardware, and source files for running and adapting the software. 
## Introduction
<img src="https://i.ibb.co/42yf4NL/javafawt-overview.png">

The proposed control scheme follows a wireless server-client architecture. A Java GUI is provided for the control station (server) to connect and command each fan, organized in series of modules. Each module is managed by a single Raspberry Pi client (currently tested with Raspberry Pi 3, 4, Zero 2 w), which receives the fan speed control messages from the server via a wireless WebSocket. The client receives the message, identifies its connected fans, and converts the speed signal to a PWM output for each fan, allowing for precise individual control of an indefinite number of fans. Thanks to the custom PCB interface, an external power supply can be used for each module, and a single control power source is used to elevate the 3.3V PWM control signal for each and every fan up to a standard required voltage.

### Features
* Modular low-cost FAWT architecture.
* Individual or multiple fan speed control through a simple unified interface.
* Websockets protocol implemented for wireless communication with each fan module.
* User-programmable automatic speed sequences via .CSV files for repeatability and reproducibility of experiments.

## Relevant links
Bill of Materials (BOM) can be found [here](https://github.com/UPNAdrone/JavaFAWT/blob/main/hardware/Bill%20of%20Materials.xlsx)

PCB Manufacturing project can be found [here](https://www.pcbway.com/project/shareproject/JavaFAWT_PCB_v0_1_1426b5d5.html)

## How to install & run
Detailed instructions for installing and running the software can be found [here](https://github.com/UPNAdrone/JavaFAWT/blob/main/documentation/private/docs/installation.md).

A Tutorial/Manual of use can be found [here](https://github.com/UPNAdrone/JavaFAWT/blob/main/documentation/private/docs/installation.md).

## Files structure
- Documentation
  - private
    - docs -> .md files for generating public documentation site.
    - site -> compiled documentation website for dev purposes.
  - public -> compiled documentation website for public purposes.
- Examples -> provided examples.
- Hardware
  - module -> All the components required to assemble a single module.
  - testing_equipment -> Other components developed for testing.
- Software
  - clientCode -> Python PWM client
  - functionalities -> example functionalities provided for automatic fan control.
  - pythonCode -> Python client file for controlling the fans from the Raspberry Pi.
  - serverCode -> JavaFAWT server software.



  
## Work in progress
* Fan speed feedback is currenly under development.
  
## Licensing
JavaFAWT is made available under the [MIT License](https://opensource.org/licenses/MIT). You are free to use, modify, and distribute this module in accordance with the terms of the MIT License.

### Reference and Citation
If you use this module for research, please consider citing our paper to acknowledge its contribution:

```bibtex
@article{pending,
  title={},
  author={},
  journal={},
  volume={},
  pages={},
  year={},
  publisher={}
}
```

---

<img src="https://avatars.githubusercontent.com/u/136073538?s=400&u=96a8df5bf4fe5348be8085165b4d86d6cfa5b150&v=4" width="180">
