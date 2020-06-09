# Capstone Project
# Subject Identification By Eyes Movement: A Machine Learning Approach
Human identification has always been a very important problem in our world. Not only for knowing if a person is who he says he is (authentication problem), but also for identifying who that person is (recognition problem). This can be seen in several areas as financial services, security & safety, health care, telecommunication, etc. when referred to giving someone access to resources, personalized services or individual care. [1][2] Once biometrics identification apply technology to accurately identify a person by his intrinsic physical or behavioral characteristics [3], it has been increasingly used for human identification. 

In this project I used machine learning techniques to develop an identification method based on eyes movement data, as proposed by Kasprowski and Komogortsev in the First Eye Movements' Verification and Identification Competition in the IEEE Fifth International Conference on Biometrics: Theory, Applications and Systems (BTAS 2012) [5]. 

## Contents
- Project files **(Read it to understand the problem and the solution.)**
    - proposal.pdf: outline the details of the problem, my research and proposes an approach to a solution.
    - project.pdf: final report submitted. 
- Implementation files
    - solutionRF.ipynb: main notebook.
    - utils.py: auxiliary functions.
- Data files directory
    - All data provided was collected by Dr. Paweł Kasprowski at Silesian University of Technology, Poland [6] and consists as samples from 37 subjects, each sample consisting in 2048 measures. 
    - Files used in this work were obtained directly with Mr. Kasproskwi, since files available at Kaggle does not have label informations for test data and the link for the files at BTAS2012 competition site [7] were not working.
    - Datasets structures are discussed in the final report.
    - All data is published for purpose of this work only. However, if you intend to use the data in your future research you may do it only if the databases are acknowledged with the following reference: 
        >KASPROWSKI, P., OBER, J. 2004. Eye Movement in Biometrics, In Proceedings of Biometric Authentication Workshop, European Conference on Computer Vision in Prague 2004, LNCS 3087, Springer-Verlag.the IEEE/IARP International Conference on Biometrics (ICB), pp. 1-8.

## License
All directories and files are MIT Licensed. Feel free to use contents as you please. If you do use them, a link back to https://github.com/guilhermeverissimo would be appreciated, but is not required.

## References
- [1]Kasprowski, P. (2012), "EMVIC - Eye Movements' Verification and Identification Competition", http://www.kasprowski.pl/emvic2012/biometrics.php. [Accessed: 10-May-2019].
- [2]Anil Jain, Lin Hong, and Sharath Pankanti. 2000. Biometric identification. Commun. ACM 43, 2 (February 2000), 90-98. DOI: https://doi.org/10.1145/328236.328110
- [3]Rouse, M. (2019). What is biometrics? Available at: https://searchsecurity.techtarget.com/definition/biometrics [Accessed May, 10th 2019].
- [4]Kasprowski, Pawel & Ober, Józef. (2004). Eye Movements in Biometrics. 248-258. 10.1007/978-3-540-25976-3_23. 
- [5]Kasprowski, Pawel & Komogortsev, Oleg & Karpov, Alex. (2012). First Eye Movement Verification and Identification Competition at BTAS 2012. 2012 IEEE 5th International Conference on Biometrics: Theory, Applications and Systems, BTAS 2012. 10.1109/BTAS.2012.6374577. 
- [6]Kasprowski, P., Ober, J. 2004. Eye Movement in Biometrics, In Proceedings of Biometric Authentication Workshop, European Conference on Computer Vision in Prague 2004, LNCS 3087, Springer-Verlag.the IEEE/IARP International Conference on Biometrics (ICB), pp. 1-8.
- [7] http://www.kasprowski.pl/emvic2012
