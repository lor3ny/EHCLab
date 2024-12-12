# ACCELERATE A FUNCTION USING FPGA 

## STEP 1: CREATING THE IP (INTELLECTUAL PROPERTY) IN VITIS HLS

An IP is a pre-designed, reusable block of logic or functionality that can be integrated into your FPGA design.

1) Create the project (put it in the C: directory), select the code to be accelerated (maybe selecting a top function), select the testbench and the compiling flags to add to the source files. Choose the clock period (usually 10 or 20, that is 100 or 200 MHz) and the board we are working with.

2) Launch the C simulation and then the C synthesis to see the data about the performance

3) Select the function or the loop to accelerate inserting directives that guide the compiler into optimizing the code as we want:
   - LOOP PIPELINING enables pipelining of loop iterations for concurrent execution
   - LOOP UNROLLING unrolls the loop to increase parallelism
   - LOOP TRICOUNT specifies the expected range of loop iterations for accurate resource estimation
   - LOOP MERGE that merges consecutive loops into a single one
   - DEPENDENCE breaks false dependencies (when the compiler assumes there's a dependency that doesn't exist) to allow parallelism
     
   - ARRAY PARTITION partitions array to allow parallel access to memory
   - ARRAY RESHAPE reshapes arrays for better resource usage
   - STREAM converts arrays to FIFO for streaming data access
   - BIND STORAGE defines storage type (BRAM, registers, distributed RAM) for arrays or variables
  
   - INLINE for inling a function, that is substituting the function call with the function body to allow further optimizations
   - DATAFLOW enables task-level parallelism by pipelining functions or loops
   - ALLOCATION limits the number of hardware resources allocated for a function
  
   - RESOURCE maps operations to specific resources, such as DSP blocks (special units to perform high-speed arithmetic operations) or LUT (blocks to implement combinational logic)
   - LATENCY specifies the minimum or maximum latency for a block of code
   - ALLOCATION limits the usage of specific resources like multipliers or adders
     
   - INTERFACE defines the interface type (e.g. AXI or FIFO) for function arguments or global variables

4) Use "EXPORT RTL" to synthetize the block and include the result in the IP

### INTERFACES
HOST-FPGA INTERFACES:
- Memory-mapped Interfaces uses memory locations for transferring data, meaning that CPU accesses the FPGA's memory regions directly
- Streaminf Interfaces, used for continous or pipelined data tranfer, meaning that data flows through streams that are similar to FIFO queues
- Application Processor Interfaces

INTERNAL FPGA INTERFACES (Between modules/blocks):
- AXI (Advanced Extensible Interface) is a high-performance interface standard. There are different interfaces of this type: AXI4 (for memory-mapped data transfers), AXI Stream (for streaming data) and AXI Lite (Lightweight version for simple control and registers)
- FIFO, for handling data streams between different stages of a pipeline without the need of complex memory management
- Register-based Interfaces use control registers for communication between blocks, they're simpler but may be useful for control signals or status updates

## STEP 2: HARDWARE PLATFORM DESIGN IN VIVADO

Vivado is used to create the hardware platform that includes a timer, our accelerator and a DMA (Direct Memory Access) controller.

1) Create a project: Select the type, in our case it's RTL Project, that is used for a lower level abstraction that focuses on the register-transfer level (we don't have to specify it because we can use the already created IP). Then, select the ZedBoard to use.

2) Add the IP to the repository: Select the IP Catalog, right-click on it to select the IP Settings, then select the Repository and add a new folder. Then, select the User Repository and add IP to repository and choose the previously created IP.

3) Create the block design: In the IP folder there's a tcl file that should be added to the project directory. Then, in the TCL console enter the command "source <filename>.tcl, then click on "create block design" to create the design and an empty diagram table should appear. TCL scripts are used to mange the design flow, managing projects or automate tasks such as adding files, running synthesis, ...
In the end, in the control tab go to the project directory and type the command "source create_design.tcl", right-click on the diagram tab and select regenerate design. Now, validate the design using the validation icon and we're done!


## STEP 3: USING THE ACCELERATOR IN XILINX VITIS

Now, we'll create a workspace and an application project in vitis to actually use the accelerator we created.

1) Choose “Create a new platform from hardware (XSA)” and select the file "system_wrapper.xsa" previously created in vivado, choose the domain (usually leave the defaults) and the templates (usually the "Hello World" one).

2) Develop a new program that interacts with the target hardware platform and that call the accelerators to perform the critical functions. Import these functions with import sources. If needed, we can change the size of the buffer and the heap.

3) Build the application, that is compiling the program (maybe using compiler flags and so on) and linking libraries, object files and the hardware platform (specified by the XSA file).

4) Debug and Optimize: run the application in debug mode (including breakpoints, inspecting variables and checking the status of the hardware accelerator) and configure the terminal to communicate with the port connected to the accelerator. We can do also optimize perfirmance by using vitis analyzer to profile everything.
