Group Members:

Arun Gimblett (21136295)
Lachlan (Lockie) Drysdale (21355384)



A combined MPI and OpenMP implementation of 1-dimesnional finite element method

This project is a continuation from the first project. The aim of the project is to implement the program from the first project into a combined MPI-OpenMP framework. This is quite a common apparoach for exploiting both coarse and fine grain parallelism in programs. A problem is partitioned coarsely at the top level and finely within each individual part. The coarse level partitioning is done using MPI and the finer level partitioning is usually done on a multi-core machine or on a Graphics Processing Unit (GPU).

The aim of the project is to partition the Finite Element Program into smaller parts and distribute these parts to different computers by using MPI. The computation on each part would now occur within the individual machines using the cores available on those machines.

The coding in the project will be minimal, it is the case quite often that parallelizing a piece of sequential code requires only small but well thought out modifications.

The deliverables:

-   The first deliverable is of course your modified C or FORTRAN code with appropriate MPI and OpenMP directives. You     should also comment your code suitably, so that the code itself can be read and your modifications can be              understood.

-   You have to do a scalability study by using 2,4,6 and 8 machines in the IBM cluster. Each machine in the cluster       has 4 cores. For each of the above number of machines, you should use 4 threads in each machine. Also you should do     the above study using at least two problem sizes. The IBM machines each have 8 GB of RAM, so you should be able to     allocate quite large arrays dynamically. You should plot the results in graphs and include in the report.

-   You have to submit a document where you should explain how you have implemented the parallelism in the code and        why. Include also all the decisions related to your implementation in the first project, so that the document can      be read without checking your first project again. The document should also include the graphs mentioned above. The     document should be strictly in pdf format.
    
-   You should develop your code on the IBM cluster. Please make as much use of it as you can. However, I will make the     login restricted after about ten days, so that you can conduct your final performance analysis without any             interference. I will allocate slots for group members when only they can login.

-   Of course the output from your parallelized program should match the output of the sequential program.


Deadline: The submission deadline is 11:59 pm on November 4, through cssubmit.

Note: The project can be done either individually or in a group consisting of a maximum of two students.

Note: I will give you hints within a few days if there are requests for hints. You are also welcome to post your thoughts/designs, however, you should not post any code. 
