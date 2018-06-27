\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj: wcvar
Subject To
 w_0_ADM:                        w_0_ADM - 0.99 b_0_ADM + 1.01 s_0_ADM  = 0.1
 w_0_AXP:                        w_0_AXP - 0.99 b_0_AXP + 1.01 s_0_AXP  = 0.1
 w_0_DIS:                        w_0_DIS - 0.99 b_0_DIS + 1.01 s_0_DIS  = 0.1
 w_0_FDX:                        w_0_FDX - 0.99 b_0_FDX + 1.01 s_0_FDX  = 0.1
 w_0_IBM:                        w_0_IBM - 0.99 b_0_IBM + 1.01 s_0_IBM  = 0.1
 w_0_INTC:                       w_0_INTC - 0.99 b_0_INTC + 1.01 s_0_INTC
                                  = 0.1
 w_0_KO:                         w_0_KO - 0.99 b_0_KO + 1.01 s_0_KO  = 0.1
 w_0_MAT:                        w_0_MAT - 0.99 b_0_MAT + 1.01 s_0_MAT  = 0.1
 w_0_MSFT:                       w_0_MSFT - 0.99 b_0_MSFT + 1.01 s_0_MSFT
                                  = 0.1
 w_0_PG:                         w_0_PG - 0.99 b_0_PG + 1.01 s_0_PG  = 0.1
 w_0_balance_constraint:         b_0_ADM + b_0_AXP + b_0_DIS + b_0_FDX
                                 + b_0_IBM + b_0_INTC + b_0_KO + b_0_MAT
                                 + b_0_MSFT + b_0_PG - s_0_ADM - s_0_AXP
                                 - s_0_DIS - s_0_FDX - s_0_IBM - s_0_INTC
                                 - s_0_KO - s_0_MAT - s_0_MSFT - s_0_PG  = 0
 w_k1_ADM_s0_return_constraint:  1.0034 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.939497 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.3789 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.934721 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.17469 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.882395 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.1934 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.28799 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.829131 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.683782 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.922937 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.759691 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.932826 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.825766 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.2473 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.11557 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.13958 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.17617 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.868134 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.818873 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.797619 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.09248 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.15697 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.903101 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.36889 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.710851 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.892465 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.02768 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.893261 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.16407 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.848894 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.45077 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.897478 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.03674 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.16263 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.3052 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.20859 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.34604 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.16589 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.871337 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.22102 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.90445 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.32768 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.83253 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.74538 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.34005 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.991802 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.733079 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.29184 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.586745 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.885513 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.991702 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.07353 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.911095 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.878612 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.25318 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.823068 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.637949 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.624994 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.25542 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.38686 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.0818 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.28756 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.727902 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.709808 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.90852 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.918022 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.51389 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.775658 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.43421 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.922547 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.85558 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.30279 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.44805 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.992971 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.4227 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.17424 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.896798 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.3911 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.777611 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.50535 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.840385 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.88245 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.18259 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.880515 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.571046 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.26706 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.68232 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.83211 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.85896 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.16904 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.04922 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.38426 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.4927 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.883872 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.704766 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.703575 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.24283 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.30187 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.854901 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.16083 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.638422 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.648654 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.88007 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.38592 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.70872 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.691288 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.25756 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.02806 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.811966 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.981462 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.60107 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.47896 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.59947 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.09261 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.945361 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.23381 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.83759 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.27811 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.905853 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.57911 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.26251 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.8867 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.732627 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.08405 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.765542 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.02749 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.928672 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.25147 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.18519 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.0844 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.25895 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.956822 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.826598 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.751624 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.16128 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.14556 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.894505 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.24432 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.715139 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.785383 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.915 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.08384 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.37397 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.738543 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.28755 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.872541 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.95913 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.03002 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.33798 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.19267 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.16597 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.20849 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.817426 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.11569 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.958224 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.2217 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.949449 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.30553 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.857846 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.38047 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.734606 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.34216 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.63126 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.611743 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.873619 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.884973 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.60964 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.2094 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.98282 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.00331 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.625533 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.0056 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.31248 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.31706 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.36918 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.802085 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.19557 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.702878 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.867668 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.917081 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.63678 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.891375 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.829455 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.42245 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.656232 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.13065 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.20959 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.860813 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.77338 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.11555 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.897752 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.06243 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.914861 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.31592 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.914581 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.46478 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.18561 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.18648 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.742025 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.24874 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.770248 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.01454 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.09171 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.1503 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.08833 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.904937 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.27402 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.99789 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.747121 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.798653 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.08786 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.11899 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.948822 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.01397 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.750383 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.838961 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.798332 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.985587 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.25137 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.815069 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.36819 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.986178 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.01172 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.06203 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.1628 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.31555 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.39406 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.1396 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.868235 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.20728 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.885997 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.33348 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.915256 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.46229 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.02438 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.13281 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.681565 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.23066 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.674986 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.796456 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.93158 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.11218 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.824605 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.37629 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.49907 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.974028 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.698015 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.947525 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.06498 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.17386 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.09087 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.845858 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.87019 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.655265 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.867126 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.04791 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.23403 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.78963 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.10488 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.11403 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.730749 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.04373 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.3192 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.17197 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.57823 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.23303 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.82421 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.15829 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.0126 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.19833 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.936502 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.74266 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.901031 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.63734 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.671676 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.47956 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.387547 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.528847 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.967299 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.942432 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.510462 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.02909 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.91562 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.816766 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.512278 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.906683 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.38967 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.34575 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.64795 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.03465 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.13668 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.606284 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.807359 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.10943 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.38441 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.714781 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.919449 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.2121 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.564347 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.18012 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.20153 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.758073 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.86117 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.06003 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.953535 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.18825 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.902737 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.22752 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.91505 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.4434 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.19628 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.996159 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.787159 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.19751 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.744516 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.961066 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.882658 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.21157 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.04909 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.20122 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.18029 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.99348 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.822957 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.716325 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.01201 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.13829 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.01479 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.07147 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.762106 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.851496 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.840983 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.19128 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.32649 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.818774 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.20301 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.99372 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.906563 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.943956 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.2551 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.29871 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.15428 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.06511 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.939775 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.13224 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.944393 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.14784 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.899316 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.34138 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.05879 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.991917 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.770452 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.10925 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.795152 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.897742 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.928785 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.0387 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.970739 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.05565 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.2217 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.943993 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.832474 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.806582 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.05219 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.28721 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.985521 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.03968 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.935042 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.897974 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.977601 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.03111 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.28881 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.893528 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.22693 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.02487 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.942331 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.11517 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.25423 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.19964 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.1489 w_k1_PG_s11 - w_k1_PG_s117
                                 + 0.99 b_k1_PG_s117 - 1.01 s_k1_PG_s117  = 0
 k1_s0_balance_constraint:       b_k1_ADM_s0 + b_k1_AXP_s0 + b_k1_DIS_s0
                                 + b_k1_FDX_s0 + b_k1_IBM_s0 + b_k1_INTC_s0
                                 + b_k1_KO_s0 + b_k1_MAT_s0 + b_k1_MSFT_s0
                                 + b_k1_PG_s0 - s_k1_ADM_s0 - s_k1_AXP_s0
                                 - s_k1_DIS_s0 - s_k1_FDX_s0 - s_k1_IBM_s0
                                 - s_k1_INTC_s0 - s_k1_KO_s0 - s_k1_MAT_s0
                                 - s_k1_MSFT_s0 - s_k1_PG_s0  = 0
 k1_s1_balance_constraint:       b_k1_ADM_s1 + b_k1_AXP_s1 + b_k1_DIS_s1
                                 + b_k1_FDX_s1 + b_k1_IBM_s1 + b_k1_INTC_s1
                                 + b_k1_KO_s1 + b_k1_MAT_s1 + b_k1_MSFT_s1
                                 + b_k1_PG_s1 - s_k1_ADM_s1 - s_k1_AXP_s1
                                 - s_k1_DIS_s1 - s_k1_FDX_s1 - s_k1_IBM_s1
                                 - s_k1_INTC_s1 - s_k1_KO_s1 - s_k1_MAT_s1
                                 - s_k1_MSFT_s1 - s_k1_PG_s1  = 0
 k1_s00_balance_constraint:      b_k1_ADM_s00 + b_k1_AXP_s00 + b_k1_DIS_s00
                                 + b_k1_FDX_s00 + b_k1_IBM_s00 + b_k1_INTC_s00
                                 + b_k1_KO_s00 + b_k1_MAT_s00 + b_k1_MSFT_s00
                                 + b_k1_PG_s00 - s_k1_ADM_s00 - s_k1_AXP_s00
                                 - s_k1_DIS_s00 - s_k1_FDX_s00 - s_k1_IBM_s00
                                 - s_k1_INTC_s00 - s_k1_KO_s00 - s_k1_MAT_s00
                                 - s_k1_MSFT_s00 - s_k1_PG_s00  = 0
 k1_s01_balance_constraint:      b_k1_ADM_s01 + b_k1_AXP_s01 + b_k1_DIS_s01
                                 + b_k1_FDX_s01 + b_k1_IBM_s01 + b_k1_INTC_s01
                                 + b_k1_KO_s01 + b_k1_MAT_s01 + b_k1_MSFT_s01
                                 + b_k1_PG_s01 - s_k1_ADM_s01 - s_k1_AXP_s01
                                 - s_k1_DIS_s01 - s_k1_FDX_s01 - s_k1_IBM_s01
                                 - s_k1_INTC_s01 - s_k1_KO_s01 - s_k1_MAT_s01
                                 - s_k1_MSFT_s01 - s_k1_PG_s01  = 0
 k1_s10_balance_constraint:      b_k1_ADM_s10 + b_k1_AXP_s10 + b_k1_DIS_s10
                                 + b_k1_FDX_s10 + b_k1_IBM_s10 + b_k1_INTC_s10
                                 + b_k1_KO_s10 + b_k1_MAT_s10 + b_k1_MSFT_s10
                                 + b_k1_PG_s10 - s_k1_ADM_s10 - s_k1_AXP_s10
                                 - s_k1_DIS_s10 - s_k1_FDX_s10 - s_k1_IBM_s10
                                 - s_k1_INTC_s10 - s_k1_KO_s10 - s_k1_MAT_s10
                                 - s_k1_MSFT_s10 - s_k1_PG_s10  = 0
 k1_s11_balance_constraint:      b_k1_ADM_s11 + b_k1_AXP_s11 + b_k1_DIS_s11
                                 + b_k1_FDX_s11 + b_k1_IBM_s11 + b_k1_INTC_s11
                                 + b_k1_KO_s11 + b_k1_MAT_s11 + b_k1_MSFT_s11
                                 + b_k1_PG_s11 - s_k1_ADM_s11 - s_k1_AXP_s11
                                 - s_k1_DIS_s11 - s_k1_FDX_s11 - s_k1_IBM_s11
                                 - s_k1_INTC_s11 - s_k1_KO_s11 - s_k1_MAT_s11
                                 - s_k1_MSFT_s11 - s_k1_PG_s11  = 0
 k1_s000_balance_constraint:     b_k1_ADM_s000 + b_k1_AXP_s000 + b_k1_DIS_s000
                                 + b_k1_FDX_s000 + b_k1_IBM_s000
                                 + b_k1_INTC_s000 + b_k1_KO_s000
                                 + b_k1_MAT_s000 + b_k1_MSFT_s000
                                 + b_k1_PG_s000 - s_k1_ADM_s000 - s_k1_AXP_s000
                                 - s_k1_DIS_s000 - s_k1_FDX_s000
                                 - s_k1_IBM_s000 - s_k1_INTC_s000
                                 - s_k1_KO_s000 - s_k1_MAT_s000
                                 - s_k1_MSFT_s000 - s_k1_PG_s000  = 0
 k1_s001_balance_constraint:     b_k1_ADM_s001 + b_k1_AXP_s001 + b_k1_DIS_s001
                                 + b_k1_FDX_s001 + b_k1_IBM_s001
                                 + b_k1_INTC_s001 + b_k1_KO_s001
                                 + b_k1_MAT_s001 + b_k1_MSFT_s001
                                 + b_k1_PG_s001 - s_k1_ADM_s001 - s_k1_AXP_s001
                                 - s_k1_DIS_s001 - s_k1_FDX_s001
                                 - s_k1_IBM_s001 - s_k1_INTC_s001
                                 - s_k1_KO_s001 - s_k1_MAT_s001
                                 - s_k1_MSFT_s001 - s_k1_PG_s001  = 0
 k1_s002_balance_constraint:     b_k1_ADM_s002 + b_k1_AXP_s002 + b_k1_DIS_s002
                                 + b_k1_FDX_s002 + b_k1_IBM_s002
                                 + b_k1_INTC_s002 + b_k1_KO_s002
                                 + b_k1_MAT_s002 + b_k1_MSFT_s002
                                 + b_k1_PG_s002 - s_k1_ADM_s002 - s_k1_AXP_s002
                                 - s_k1_DIS_s002 - s_k1_FDX_s002
                                 - s_k1_IBM_s002 - s_k1_INTC_s002
                                 - s_k1_KO_s002 - s_k1_MAT_s002
                                 - s_k1_MSFT_s002 - s_k1_PG_s002  = 0
 k1_s003_balance_constraint:     b_k1_ADM_s003 + b_k1_AXP_s003 + b_k1_DIS_s003
                                 + b_k1_FDX_s003 + b_k1_IBM_s003
                                 + b_k1_INTC_s003 + b_k1_KO_s003
                                 + b_k1_MAT_s003 + b_k1_MSFT_s003
                                 + b_k1_PG_s003 - s_k1_ADM_s003 - s_k1_AXP_s003
                                 - s_k1_DIS_s003 - s_k1_FDX_s003
                                 - s_k1_IBM_s003 - s_k1_INTC_s003
                                 - s_k1_KO_s003 - s_k1_MAT_s003
                                 - s_k1_MSFT_s003 - s_k1_PG_s003  = 0
 k1_s004_balance_constraint:     b_k1_ADM_s004 + b_k1_AXP_s004 + b_k1_DIS_s004
                                 + b_k1_FDX_s004 + b_k1_IBM_s004
                                 + b_k1_INTC_s004 + b_k1_KO_s004
                                 + b_k1_MAT_s004 + b_k1_MSFT_s004
                                 + b_k1_PG_s004 - s_k1_ADM_s004 - s_k1_AXP_s004
                                 - s_k1_DIS_s004 - s_k1_FDX_s004
                                 - s_k1_IBM_s004 - s_k1_INTC_s004
                                 - s_k1_KO_s004 - s_k1_MAT_s004
                                 - s_k1_MSFT_s004 - s_k1_PG_s004  = 0
 k1_s005_balance_constraint:     b_k1_ADM_s005 + b_k1_AXP_s005 + b_k1_DIS_s005
                                 + b_k1_FDX_s005 + b_k1_IBM_s005
                                 + b_k1_INTC_s005 + b_k1_KO_s005
                                 + b_k1_MAT_s005 + b_k1_MSFT_s005
                                 + b_k1_PG_s005 - s_k1_ADM_s005 - s_k1_AXP_s005
                                 - s_k1_DIS_s005 - s_k1_FDX_s005
                                 - s_k1_IBM_s005 - s_k1_INTC_s005
                                 - s_k1_KO_s005 - s_k1_MAT_s005
                                 - s_k1_MSFT_s005 - s_k1_PG_s005  = 0
 k1_s006_balance_constraint:     b_k1_ADM_s006 + b_k1_AXP_s006 + b_k1_DIS_s006
                                 + b_k1_FDX_s006 + b_k1_IBM_s006
                                 + b_k1_INTC_s006 + b_k1_KO_s006
                                 + b_k1_MAT_s006 + b_k1_MSFT_s006
                                 + b_k1_PG_s006 - s_k1_ADM_s006 - s_k1_AXP_s006
                                 - s_k1_DIS_s006 - s_k1_FDX_s006
                                 - s_k1_IBM_s006 - s_k1_INTC_s006
                                 - s_k1_KO_s006 - s_k1_MAT_s006
                                 - s_k1_MSFT_s006 - s_k1_PG_s006  = 0
 k1_s007_balance_constraint:     b_k1_ADM_s007 + b_k1_AXP_s007 + b_k1_DIS_s007
                                 + b_k1_FDX_s007 + b_k1_IBM_s007
                                 + b_k1_INTC_s007 + b_k1_KO_s007
                                 + b_k1_MAT_s007 + b_k1_MSFT_s007
                                 + b_k1_PG_s007 - s_k1_ADM_s007 - s_k1_AXP_s007
                                 - s_k1_DIS_s007 - s_k1_FDX_s007
                                 - s_k1_IBM_s007 - s_k1_INTC_s007
                                 - s_k1_KO_s007 - s_k1_MAT_s007
                                 - s_k1_MSFT_s007 - s_k1_PG_s007  = 0
 k1_s010_balance_constraint:     b_k1_ADM_s010 + b_k1_AXP_s010 + b_k1_DIS_s010
                                 + b_k1_FDX_s010 + b_k1_IBM_s010
                                 + b_k1_INTC_s010 + b_k1_KO_s010
                                 + b_k1_MAT_s010 + b_k1_MSFT_s010
                                 + b_k1_PG_s010 - s_k1_ADM_s010 - s_k1_AXP_s010
                                 - s_k1_DIS_s010 - s_k1_FDX_s010
                                 - s_k1_IBM_s010 - s_k1_INTC_s010
                                 - s_k1_KO_s010 - s_k1_MAT_s010
                                 - s_k1_MSFT_s010 - s_k1_PG_s010  = 0
 k1_s011_balance_constraint:     b_k1_ADM_s011 + b_k1_AXP_s011 + b_k1_DIS_s011
                                 + b_k1_FDX_s011 + b_k1_IBM_s011
                                 + b_k1_INTC_s011 + b_k1_KO_s011
                                 + b_k1_MAT_s011 + b_k1_MSFT_s011
                                 + b_k1_PG_s011 - s_k1_ADM_s011 - s_k1_AXP_s011
                                 - s_k1_DIS_s011 - s_k1_FDX_s011
                                 - s_k1_IBM_s011 - s_k1_INTC_s011
                                 - s_k1_KO_s011 - s_k1_MAT_s011
                                 - s_k1_MSFT_s011 - s_k1_PG_s011  = 0
 k1_s012_balance_constraint:     b_k1_ADM_s012 + b_k1_AXP_s012 + b_k1_DIS_s012
                                 + b_k1_FDX_s012 + b_k1_IBM_s012
                                 + b_k1_INTC_s012 + b_k1_KO_s012
                                 + b_k1_MAT_s012 + b_k1_MSFT_s012
                                 + b_k1_PG_s012 - s_k1_ADM_s012 - s_k1_AXP_s012
                                 - s_k1_DIS_s012 - s_k1_FDX_s012
                                 - s_k1_IBM_s012 - s_k1_INTC_s012
                                 - s_k1_KO_s012 - s_k1_MAT_s012
                                 - s_k1_MSFT_s012 - s_k1_PG_s012  = 0
 k1_s013_balance_constraint:     b_k1_ADM_s013 + b_k1_AXP_s013 + b_k1_DIS_s013
                                 + b_k1_FDX_s013 + b_k1_IBM_s013
                                 + b_k1_INTC_s013 + b_k1_KO_s013
                                 + b_k1_MAT_s013 + b_k1_MSFT_s013
                                 + b_k1_PG_s013 - s_k1_ADM_s013 - s_k1_AXP_s013
                                 - s_k1_DIS_s013 - s_k1_FDX_s013
                                 - s_k1_IBM_s013 - s_k1_INTC_s013
                                 - s_k1_KO_s013 - s_k1_MAT_s013
                                 - s_k1_MSFT_s013 - s_k1_PG_s013  = 0
 k1_s014_balance_constraint:     b_k1_ADM_s014 + b_k1_AXP_s014 + b_k1_DIS_s014
                                 + b_k1_FDX_s014 + b_k1_IBM_s014
                                 + b_k1_INTC_s014 + b_k1_KO_s014
                                 + b_k1_MAT_s014 + b_k1_MSFT_s014
                                 + b_k1_PG_s014 - s_k1_ADM_s014 - s_k1_AXP_s014
                                 - s_k1_DIS_s014 - s_k1_FDX_s014
                                 - s_k1_IBM_s014 - s_k1_INTC_s014
                                 - s_k1_KO_s014 - s_k1_MAT_s014
                                 - s_k1_MSFT_s014 - s_k1_PG_s014  = 0
 k1_s015_balance_constraint:     b_k1_ADM_s015 + b_k1_AXP_s015 + b_k1_DIS_s015
                                 + b_k1_FDX_s015 + b_k1_IBM_s015
                                 + b_k1_INTC_s015 + b_k1_KO_s015
                                 + b_k1_MAT_s015 + b_k1_MSFT_s015
                                 + b_k1_PG_s015 - s_k1_ADM_s015 - s_k1_AXP_s015
                                 - s_k1_DIS_s015 - s_k1_FDX_s015
                                 - s_k1_IBM_s015 - s_k1_INTC_s015
                                 - s_k1_KO_s015 - s_k1_MAT_s015
                                 - s_k1_MSFT_s015 - s_k1_PG_s015  = 0
 k1_s016_balance_constraint:     b_k1_ADM_s016 + b_k1_AXP_s016 + b_k1_DIS_s016
                                 + b_k1_FDX_s016 + b_k1_IBM_s016
                                 + b_k1_INTC_s016 + b_k1_KO_s016
                                 + b_k1_MAT_s016 + b_k1_MSFT_s016
                                 + b_k1_PG_s016 - s_k1_ADM_s016 - s_k1_AXP_s016
                                 - s_k1_DIS_s016 - s_k1_FDX_s016
                                 - s_k1_IBM_s016 - s_k1_INTC_s016
                                 - s_k1_KO_s016 - s_k1_MAT_s016
                                 - s_k1_MSFT_s016 - s_k1_PG_s016  = 0
 k1_s017_balance_constraint:     b_k1_ADM_s017 + b_k1_AXP_s017 + b_k1_DIS_s017
                                 + b_k1_FDX_s017 + b_k1_IBM_s017
                                 + b_k1_INTC_s017 + b_k1_KO_s017
                                 + b_k1_MAT_s017 + b_k1_MSFT_s017
                                 + b_k1_PG_s017 - s_k1_ADM_s017 - s_k1_AXP_s017
                                 - s_k1_DIS_s017 - s_k1_FDX_s017
                                 - s_k1_IBM_s017 - s_k1_INTC_s017
                                 - s_k1_KO_s017 - s_k1_MAT_s017
                                 - s_k1_MSFT_s017 - s_k1_PG_s017  = 0
 k1_s100_balance_constraint:     b_k1_ADM_s100 + b_k1_AXP_s100 + b_k1_DIS_s100
                                 + b_k1_FDX_s100 + b_k1_IBM_s100
                                 + b_k1_INTC_s100 + b_k1_KO_s100
                                 + b_k1_MAT_s100 + b_k1_MSFT_s100
                                 + b_k1_PG_s100 - s_k1_ADM_s100 - s_k1_AXP_s100
                                 - s_k1_DIS_s100 - s_k1_FDX_s100
                                 - s_k1_IBM_s100 - s_k1_INTC_s100
                                 - s_k1_KO_s100 - s_k1_MAT_s100
                                 - s_k1_MSFT_s100 - s_k1_PG_s100  = 0
 k1_s101_balance_constraint:     b_k1_ADM_s101 + b_k1_AXP_s101 + b_k1_DIS_s101
                                 + b_k1_FDX_s101 + b_k1_IBM_s101
                                 + b_k1_INTC_s101 + b_k1_KO_s101
                                 + b_k1_MAT_s101 + b_k1_MSFT_s101
                                 + b_k1_PG_s101 - s_k1_ADM_s101 - s_k1_AXP_s101
                                 - s_k1_DIS_s101 - s_k1_FDX_s101
                                 - s_k1_IBM_s101 - s_k1_INTC_s101
                                 - s_k1_KO_s101 - s_k1_MAT_s101
                                 - s_k1_MSFT_s101 - s_k1_PG_s101  = 0
 k1_s102_balance_constraint:     b_k1_ADM_s102 + b_k1_AXP_s102 + b_k1_DIS_s102
                                 + b_k1_FDX_s102 + b_k1_IBM_s102
                                 + b_k1_INTC_s102 + b_k1_KO_s102
                                 + b_k1_MAT_s102 + b_k1_MSFT_s102
                                 + b_k1_PG_s102 - s_k1_ADM_s102 - s_k1_AXP_s102
                                 - s_k1_DIS_s102 - s_k1_FDX_s102
                                 - s_k1_IBM_s102 - s_k1_INTC_s102
                                 - s_k1_KO_s102 - s_k1_MAT_s102
                                 - s_k1_MSFT_s102 - s_k1_PG_s102  = 0
 k1_s103_balance_constraint:     b_k1_ADM_s103 + b_k1_AXP_s103 + b_k1_DIS_s103
                                 + b_k1_FDX_s103 + b_k1_IBM_s103
                                 + b_k1_INTC_s103 + b_k1_KO_s103
                                 + b_k1_MAT_s103 + b_k1_MSFT_s103
                                 + b_k1_PG_s103 - s_k1_ADM_s103 - s_k1_AXP_s103
                                 - s_k1_DIS_s103 - s_k1_FDX_s103
                                 - s_k1_IBM_s103 - s_k1_INTC_s103
                                 - s_k1_KO_s103 - s_k1_MAT_s103
                                 - s_k1_MSFT_s103 - s_k1_PG_s103  = 0
 k1_s104_balance_constraint:     b_k1_ADM_s104 + b_k1_AXP_s104 + b_k1_DIS_s104
                                 + b_k1_FDX_s104 + b_k1_IBM_s104
                                 + b_k1_INTC_s104 + b_k1_KO_s104
                                 + b_k1_MAT_s104 + b_k1_MSFT_s104
                                 + b_k1_PG_s104 - s_k1_ADM_s104 - s_k1_AXP_s104
                                 - s_k1_DIS_s104 - s_k1_FDX_s104
                                 - s_k1_IBM_s104 - s_k1_INTC_s104
                                 - s_k1_KO_s104 - s_k1_MAT_s104
                                 - s_k1_MSFT_s104 - s_k1_PG_s104  = 0
 k1_s105_balance_constraint:     b_k1_ADM_s105 + b_k1_AXP_s105 + b_k1_DIS_s105
                                 + b_k1_FDX_s105 + b_k1_IBM_s105
                                 + b_k1_INTC_s105 + b_k1_KO_s105
                                 + b_k1_MAT_s105 + b_k1_MSFT_s105
                                 + b_k1_PG_s105 - s_k1_ADM_s105 - s_k1_AXP_s105
                                 - s_k1_DIS_s105 - s_k1_FDX_s105
                                 - s_k1_IBM_s105 - s_k1_INTC_s105
                                 - s_k1_KO_s105 - s_k1_MAT_s105
                                 - s_k1_MSFT_s105 - s_k1_PG_s105  = 0
 k1_s106_balance_constraint:     b_k1_ADM_s106 + b_k1_AXP_s106 + b_k1_DIS_s106
                                 + b_k1_FDX_s106 + b_k1_IBM_s106
                                 + b_k1_INTC_s106 + b_k1_KO_s106
                                 + b_k1_MAT_s106 + b_k1_MSFT_s106
                                 + b_k1_PG_s106 - s_k1_ADM_s106 - s_k1_AXP_s106
                                 - s_k1_DIS_s106 - s_k1_FDX_s106
                                 - s_k1_IBM_s106 - s_k1_INTC_s106
                                 - s_k1_KO_s106 - s_k1_MAT_s106
                                 - s_k1_MSFT_s106 - s_k1_PG_s106  = 0
 k1_s107_balance_constraint:     b_k1_ADM_s107 + b_k1_AXP_s107 + b_k1_DIS_s107
                                 + b_k1_FDX_s107 + b_k1_IBM_s107
                                 + b_k1_INTC_s107 + b_k1_KO_s107
                                 + b_k1_MAT_s107 + b_k1_MSFT_s107
                                 + b_k1_PG_s107 - s_k1_ADM_s107 - s_k1_AXP_s107
                                 - s_k1_DIS_s107 - s_k1_FDX_s107
                                 - s_k1_IBM_s107 - s_k1_INTC_s107
                                 - s_k1_KO_s107 - s_k1_MAT_s107
                                 - s_k1_MSFT_s107 - s_k1_PG_s107  = 0
 k1_s110_balance_constraint:     b_k1_ADM_s110 + b_k1_AXP_s110 + b_k1_DIS_s110
                                 + b_k1_FDX_s110 + b_k1_IBM_s110
                                 + b_k1_INTC_s110 + b_k1_KO_s110
                                 + b_k1_MAT_s110 + b_k1_MSFT_s110
                                 + b_k1_PG_s110 - s_k1_ADM_s110 - s_k1_AXP_s110
                                 - s_k1_DIS_s110 - s_k1_FDX_s110
                                 - s_k1_IBM_s110 - s_k1_INTC_s110
                                 - s_k1_KO_s110 - s_k1_MAT_s110
                                 - s_k1_MSFT_s110 - s_k1_PG_s110  = 0
 k1_s111_balance_constraint:     b_k1_ADM_s111 + b_k1_AXP_s111 + b_k1_DIS_s111
                                 + b_k1_FDX_s111 + b_k1_IBM_s111
                                 + b_k1_INTC_s111 + b_k1_KO_s111
                                 + b_k1_MAT_s111 + b_k1_MSFT_s111
                                 + b_k1_PG_s111 - s_k1_ADM_s111 - s_k1_AXP_s111
                                 - s_k1_DIS_s111 - s_k1_FDX_s111
                                 - s_k1_IBM_s111 - s_k1_INTC_s111
                                 - s_k1_KO_s111 - s_k1_MAT_s111
                                 - s_k1_MSFT_s111 - s_k1_PG_s111  = 0
 k1_s112_balance_constraint:     b_k1_ADM_s112 + b_k1_AXP_s112 + b_k1_DIS_s112
                                 + b_k1_FDX_s112 + b_k1_IBM_s112
                                 + b_k1_INTC_s112 + b_k1_KO_s112
                                 + b_k1_MAT_s112 + b_k1_MSFT_s112
                                 + b_k1_PG_s112 - s_k1_ADM_s112 - s_k1_AXP_s112
                                 - s_k1_DIS_s112 - s_k1_FDX_s112
                                 - s_k1_IBM_s112 - s_k1_INTC_s112
                                 - s_k1_KO_s112 - s_k1_MAT_s112
                                 - s_k1_MSFT_s112 - s_k1_PG_s112  = 0
 k1_s113_balance_constraint:     b_k1_ADM_s113 + b_k1_AXP_s113 + b_k1_DIS_s113
                                 + b_k1_FDX_s113 + b_k1_IBM_s113
                                 + b_k1_INTC_s113 + b_k1_KO_s113
                                 + b_k1_MAT_s113 + b_k1_MSFT_s113
                                 + b_k1_PG_s113 - s_k1_ADM_s113 - s_k1_AXP_s113
                                 - s_k1_DIS_s113 - s_k1_FDX_s113
                                 - s_k1_IBM_s113 - s_k1_INTC_s113
                                 - s_k1_KO_s113 - s_k1_MAT_s113
                                 - s_k1_MSFT_s113 - s_k1_PG_s113  = 0
 k1_s114_balance_constraint:     b_k1_ADM_s114 + b_k1_AXP_s114 + b_k1_DIS_s114
                                 + b_k1_FDX_s114 + b_k1_IBM_s114
                                 + b_k1_INTC_s114 + b_k1_KO_s114
                                 + b_k1_MAT_s114 + b_k1_MSFT_s114
                                 + b_k1_PG_s114 - s_k1_ADM_s114 - s_k1_AXP_s114
                                 - s_k1_DIS_s114 - s_k1_FDX_s114
                                 - s_k1_IBM_s114 - s_k1_INTC_s114
                                 - s_k1_KO_s114 - s_k1_MAT_s114
                                 - s_k1_MSFT_s114 - s_k1_PG_s114  = 0
 k1_s115_balance_constraint:     b_k1_ADM_s115 + b_k1_AXP_s115 + b_k1_DIS_s115
                                 + b_k1_FDX_s115 + b_k1_IBM_s115
                                 + b_k1_INTC_s115 + b_k1_KO_s115
                                 + b_k1_MAT_s115 + b_k1_MSFT_s115
                                 + b_k1_PG_s115 - s_k1_ADM_s115 - s_k1_AXP_s115
                                 - s_k1_DIS_s115 - s_k1_FDX_s115
                                 - s_k1_IBM_s115 - s_k1_INTC_s115
                                 - s_k1_KO_s115 - s_k1_MAT_s115
                                 - s_k1_MSFT_s115 - s_k1_PG_s115  = 0
 k1_s116_balance_constraint:     b_k1_ADM_s116 + b_k1_AXP_s116 + b_k1_DIS_s116
                                 + b_k1_FDX_s116 + b_k1_IBM_s116
                                 + b_k1_INTC_s116 + b_k1_KO_s116
                                 + b_k1_MAT_s116 + b_k1_MSFT_s116
                                 + b_k1_PG_s116 - s_k1_ADM_s116 - s_k1_AXP_s116
                                 - s_k1_DIS_s116 - s_k1_FDX_s116
                                 - s_k1_IBM_s116 - s_k1_INTC_s116
                                 - s_k1_KO_s116 - s_k1_MAT_s116
                                 - s_k1_MSFT_s116 - s_k1_PG_s116  = 0
 k1_s117_balance_constraint:     b_k1_ADM_s117 + b_k1_AXP_s117 + b_k1_DIS_s117
                                 + b_k1_FDX_s117 + b_k1_IBM_s117
                                 + b_k1_INTC_s117 + b_k1_KO_s117
                                 + b_k1_MAT_s117 + b_k1_MSFT_s117
                                 + b_k1_PG_s117 - s_k1_ADM_s117 - s_k1_AXP_s117
                                 - s_k1_DIS_s117 - s_k1_FDX_s117
                                 - s_k1_IBM_s117 - s_k1_INTC_s117
                                 - s_k1_KO_s117 - s_k1_MAT_s117
                                 - s_k1_MSFT_s117 - s_k1_PG_s117  = 0
 k1_wcvar_constraint:            wcvar - var_1 - 0.303999425252415 z_k1_s0000
                                 - 0.335999364752669 z_k1_s0001
                                 - 0.203999931564016 z_k1_s0002
                                 - 0.156999993985527 z_k1_s0003
                                 - 0.421999466534516 z_k1_s0004
                                 - 0.217999323470568 z_k1_s0005
                                 - 0.223999576501779 z_k1_s0006
                                 - 0.249999262970823 z_k1_s0007
                                 - 0.533999137333543 z_k1_s0010
                                 - 0.33100066906845 z_k1_s0011
                                 - 0.770998770515287 z_k1_s0012
                                 - 0.154999999989382 z_k1_s0013
                                 - 0.181000134543117 z_k1_s0014
                                 - 0.317001365808926 z_k1_s0015
                                 - 0.178999948377404 z_k1_s0016
                                 - 0.389000639572857 z_k1_s0017
                                 - 0.687997782632343 z_k1_s0020
                                 - 0.677998993611409 z_k1_s0021
                                 - 0.342000122511888 z_k1_s0022
                                 - 0.761000810484002 z_k1_s0023
                                 - 0.49499939653044 z_k1_s0024
                                 - 0.800002021665065 z_k1_s0025
                                 - 0.314000051753611 z_k1_s0026
                                 - 0.86700182010455 z_k1_s0027
                                 - 0.284999024496832 z_k1_s0030
                                 - 0.113999883998403 z_k1_s0031
                                 - 0.213999132822345 z_k1_s0032
                                 - 0.298000201872412 z_k1_s0033
                                 - 0.0609999667920934 z_k1_s0034
                                 - 0.154000131927321 z_k1_s0035
                                 - 0.159000162918435 z_k1_s0036
                                 - 0.0859998475479963 z_k1_s0037
                                 - 0.673000172958465 z_k1_s0040
                                 - 0.153000080659235 z_k1_s0041
                                 - 0.568001703389528 z_k1_s0042
                                 - 0.628001291000677 z_k1_s0043
                                 - 0.430999356091735 z_k1_s0044
                                 - 0.267000267621107 z_k1_s0045
                                 - 0.346999286435667 z_k1_s0046
                                 - 0.168000139312137 z_k1_s0047
                                 - 0.186999266076977 z_k1_s0050
                                 - 0.276999304776773 z_k1_s0051
                                 - 0.351999337026602 z_k1_s0052
                                 - 0.166999839143164 z_k1_s0053
                                 - 0.213999626686601 z_k1_s0054
                                 - 0.130999823663246 z_k1_s0055
                                 - 0.156999776676573 z_k1_s0056
                                 - 0.259999530133268 z_k1_s0057
                                 - 0.277999491701808 z_k1_s0060
                                 - 0.467997841425222 z_k1_s0061
                                 - 0.311997659954541 z_k1_s0062
                                 - 0.215998920565893 z_k1_s0063
                                 - 0.425998832178047 z_k1_s0064
                                 - 0.227998830564144 z_k1_s0065
                                 - 0.397998591684323 z_k1_s0066
                                 - 0.376999087060736 z_k1_s0067
                                 - 0.61399837872218 z_k1_s0070
                                 - 0.352999915601747 z_k1_s0071
                                 - 0.385000111095162 z_k1_s0072
                                 - 0.520998686166558 z_k1_s0073
                                 - 1.03599938368491 z_k1_s0074
                                 - 0.918000948259822 z_k1_s0075
                                 - 0.637002541043553 z_k1_s0076
                                 - 0.284999915715854 z_k1_s0077
                                 - 0.375001475962252 z_k1_s0100
                                 - 0.333001014398469 z_k1_s0101
                                 - 0.6080006566374 z_k1_s0102
                                 - 0.199000100209276 z_k1_s0103
                                 - 0.317999967839942 z_k1_s0104
                                 - 0.437999082307812 z_k1_s0105
                                 - 0.618999161046902 z_k1_s0106
                                 - 0.195999890897571 z_k1_s0107
                                 - 0.567999430658937 z_k1_s0110
                                 - 0.403001260415614 z_k1_s0111
                                 - 0.195000106652441 z_k1_s0112
                                 - 0.306000095362253 z_k1_s0113
                                 - 0.165000042244347 z_k1_s0114
                                 - 0.245000005999151 z_k1_s0115
                                 - 0.809000842471116 z_k1_s0116
                                 - 0.429000234635371 z_k1_s0117
                                 - 0.27600060046659 z_k1_s0120
                                 - 0.220000855415755 z_k1_s0121
                                 - 0.131000146401765 z_k1_s0122
                                 - 0.142000096322464 z_k1_s0123
                                 - 0.174000136337403 z_k1_s0124
                                 - 0.394000063252278 z_k1_s0125
                                 - 0.110000056307525 z_k1_s0126
                                 - 0.40999999040966 z_k1_s0127
                                 - 0.22000014037303 z_k1_s0130
                                 - 0.204000193409563 z_k1_s0131
                                 - 0.305999130613876 z_k1_s0132
                                 - 0.161000118538906 z_k1_s0133
                                 - 0.37399919508373 z_k1_s0134
                                 - 0.449000323381791 z_k1_s0135
                                 - 0.362000684236715 z_k1_s0136
                                 - 0.243001151164704 z_k1_s0137
                                 - 0.704998584435619 z_k1_s0140
                                 - 0.298999693281611 z_k1_s0141
                                 - 0.400999789027305 z_k1_s0142
                                 - 0.794999958298132 z_k1_s0143
                                 - 0.751998814797658 z_k1_s0144
                                 - 0.691997898889316 z_k1_s0145
                                 - 0.351999528134988 z_k1_s0146
                                 - 0.875001179509254 z_k1_s0147
                                 - 0.565002124714913 z_k1_s0150
                                 - 0.362000852855418 z_k1_s0151
                                 - 0.359000711964676 z_k1_s0152
                                 - 1.45100129910859 z_k1_s0153
                                 - 0.696003395002345 z_k1_s0154
                                 - 0.976003401115924 z_k1_s0155
                                 - 0.673003816975875 z_k1_s0156
                                 - 0.551000716156845 z_k1_s0157
                                 - 0.828000193693917 z_k1_s0160
                                 - 0.840997276517681 z_k1_s0161
                                 - 0.825996607883312 z_k1_s0162
                                 - 0.238999464264336 z_k1_s0163
                                 - 0.878997222061737 z_k1_s0164
                                 - 0.408998999803997 z_k1_s0165
                                 - 0.914998826784222 z_k1_s0166
                                 - 0.307999392989012 z_k1_s0167
                                 - 0.537000777363349 z_k1_s0170
                                 - 0.413000172479251 z_k1_s0171
                                 - 0.862002629297715 z_k1_s0172
                                 - 1.69800393523963 z_k1_s0173
                                 - 0.770997997552398 z_k1_s0174
                                 - 0.837999147736481 z_k1_s0175
                                 - 1.26600002719018 z_k1_s0176
                                 - 0.959999707324029 z_k1_s0177
                                 - 0.24999928772879 z_k1_s1000
                                 - 0.200999413357942 z_k1_s1001
                                 - 0.174000042335439 z_k1_s1002
                                 - 0.397000657841989 z_k1_s1003
                                 - 0.0880000916926624 z_k1_s1004
                                 - 0.23299922435519 z_k1_s1005
                                 - 0.234000255730493 z_k1_s1006
                                 - 0.1700001096358 z_k1_s1007
                                 - 0.171000843473418 z_k1_s1010
                                 - 0.108000323308859 z_k1_s1011
                                 - 0.183001226753288 z_k1_s1012
                                 - 0.166000328522339 z_k1_s1013
                                 - 0.138000428905815 z_k1_s1014
                                 - 0.136000365025836 z_k1_s1015
                                 - 0.417000885185825 z_k1_s1016
                                 - 0.102000273769377 z_k1_s1017
                                 - 0.142999883579861 z_k1_s1020
                                 - 0.54399887528862 z_k1_s1021
                                 - 0.472000026294245 z_k1_s1022
                                 - 0.144999968096337 z_k1_s1023
                                 - 0.374999465594107 z_k1_s1024
                                 - 0.402000999716424 z_k1_s1025
                                 - 0.2109998740409 z_k1_s1026
                                 - 0.329000929014155 z_k1_s1027
                                 - 0.111999601796191 z_k1_s1030
                                 - 0.177000043757934 z_k1_s1031
                                 - 0.0419999988984986 z_k1_s1032
                                 - 0.0520000147695617 z_k1_s1033
                                 - 0.0970000861893458 z_k1_s1034
                                 - 0.0579999904122163 z_k1_s1035
                                 - 0.139999713995134 z_k1_s1036
                                 - 0.169000132701034 z_k1_s1037
                                 - 0.115000170547048 z_k1_s1040
                                 - 0.286000561172951 z_k1_s1041
                                 - 0.159000343466148 z_k1_s1042
                                 - 0.333000089280591 z_k1_s1043
                                 - 0.325000808242353 z_k1_s1044
                                 - 0.0760001110780457 z_k1_s1045
                                 - 0.375999976869112 z_k1_s1046
                                 - 0.206000246774587 z_k1_s1047
                                 - 0.319999296672034 z_k1_s1050
                                 - 0.519001219953966 z_k1_s1051
                                 - 0.310000638276877 z_k1_s1052
                                 - 0.268999570052532 z_k1_s1053
                                 - 0.240000645504776 z_k1_s1054
                                 - 0.319999296672034 z_k1_s1055
                                 - 0.263000844215738 z_k1_s1056
                                 - 0.104999989158152 z_k1_s1057
                                 - 0.104000263414726 z_k1_s1060
                                 - 0.223999894662787 z_k1_s1061
                                 - 0.0769999290090986 z_k1_s1062
                                 - 0.0549999833636266 z_k1_s1063
                                 - 0.0639999358322401 z_k1_s1064
                                 - 0.139000122737092 z_k1_s1065
                                 - 0.0709999871966777 z_k1_s1066
                                 - 0.0610000046760119 z_k1_s1067
                                 - 0.0869999878615975 z_k1_s1070
                                 - 0.0940000067676997 z_k1_s1071
                                 - 0.063999994384393 z_k1_s1072
                                 - 0.133999895145464 z_k1_s1073
                                 - 0.0689999804316135 z_k1_s1074
                                 - 0.142000218780958 z_k1_s1075
                                 - 0.240999538895993 z_k1_s1076
                                 - 0.130000213827635 z_k1_s1077
                                 - 1.52999774240999 z_k1_s1100
                                 - 1.2019972194585 z_k1_s1101
                                 - 0.438999721750151 z_k1_s1102
                                 - 0.673000000030221 z_k1_s1103
                                 - 0.64699981206584 z_k1_s1104
                                 - 0.450999482333859 z_k1_s1105
                                 - 0.454000129179467 z_k1_s1106
                                 - 1.67099846276696 z_k1_s1107
                                 - 0.239999726562029 z_k1_s1110
                                 - 0.352999094466388 z_k1_s1111
                                 - 0.574000822330849 z_k1_s1112
                                 - 0.702000723457211 z_k1_s1113
                                 - 0.611999637322796 z_k1_s1114
                                 - 0.16199984184434 z_k1_s1115
                                 - 0.151999838562605 z_k1_s1116
                                 - 0.726999851162545 z_k1_s1117
                                 - 0.272000601163072 z_k1_s1120
                                 - 0.862001326938569 z_k1_s1121
                                 - 0.582002404200157 z_k1_s1122
                                 - 0.316000631593745 z_k1_s1123
                                 - 0.205000284756798 z_k1_s1124
                                 - 0.350999972685124 z_k1_s1125
                                 - 0.698001658334115 z_k1_s1126
                                 - 0.209000319296006 z_k1_s1127
                                 - 0.331000283225377 z_k1_s1130
                                 - 0.437000047020857 z_k1_s1131
                                 - 0.638001897366 z_k1_s1132
                                 - 1.08200145545316 z_k1_s1133
                                 - 0.569000585493249 z_k1_s1134
                                 - 0.40900018925419 z_k1_s1135
                                 - 1.19599815610814 z_k1_s1136
                                 - 1.3830003798202 z_k1_s1137
                                 - 0.461001052081293 z_k1_s1140
                                 - 0.484001582013255 z_k1_s1141
                                 - 0.271000841788261 z_k1_s1142
                                 - 0.438000522149331 z_k1_s1143
                                 - 0.359002097612217 z_k1_s1144
                                 - 0.535001059247793 z_k1_s1145
                                 - 0.444000042999001 z_k1_s1146
                                 - 0.558001589179756 z_k1_s1147
                                 - 0.238000485919702 z_k1_s1150
                                 - 0.0979999549669713 z_k1_s1151
                                 - 0.417000890287876 z_k1_s1152
                                 - 0.447000970992037 z_k1_s1153
                                 - 0.102999891251009 z_k1_s1154
                                 - 0.16799998894337 z_k1_s1155
                                 - 0.421998974572179 z_k1_s1156
                                 - 0.421998974572179 z_k1_s1157
                                 - 0.189999969914185 z_k1_s1160
                                 - 0.158000012370731 z_k1_s1161
                                 - 0.115000047221722 z_k1_s1162
                                 - 0.221999927457639 z_k1_s1163
                                 - 0.275999189812436 z_k1_s1164
                                 - 0.332999891186459 z_k1_s1165
                                 - 0.299000158296467 z_k1_s1166
                                 - 0.183000756201301 z_k1_s1167
                                 - 0.331000408544082 z_k1_s1170
                                 - 0.444999372151165 z_k1_s1171
                                 - 0.314999365230561 z_k1_s1172
                                 - 0.344000651236193 z_k1_s1173
                                 - 0.199000029209237 z_k1_s1174
                                 - 0.219000123350888 z_k1_s1175
                                 - 0.249001111563539 z_k1_s1176
                                 - 0.318000165851971 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.864523 w_k1_ADM_s000
                                 + 0.730056 w_k1_AXP_s000
                                 + 0.66009 w_k1_DIS_s000
                                 + 0.752554 w_k1_FDX_s000
                                 + 0.806492 w_k1_IBM_s000
                                 + 0.866764 w_k1_INTC_s000
                                 + 0.775802 w_k1_KO_s000
                                 + 0.601896 w_k1_MAT_s000
                                 + 0.831488 w_k1_MSFT_s000
                                 + 0.802242 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.18809 w_k1_ADM_s000
                                 + 0.852056 w_k1_AXP_s000
                                 + 1.07048 w_k1_DIS_s000
                                 + 1.02484 w_k1_FDX_s000
                                 + 0.811632 w_k1_IBM_s000
                                 + 0.905472 w_k1_INTC_s000
                                 + 0.976558 w_k1_KO_s000
                                 + 0.924178 w_k1_MAT_s000
                                 + 1.11415 w_k1_MSFT_s000
                                 + 1.05399 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.06491 w_k1_ADM_s000
                                 + 1.19396 w_k1_AXP_s000
                                 + 1.43365 w_k1_DIS_s000
                                 + 1.19028 w_k1_FDX_s000
                                 + 1.10676 w_k1_IBM_s000
                                 + 0.978191 w_k1_INTC_s000
                                 + 1.19481 w_k1_KO_s000 + 1.29521 w_k1_MAT_s000
                                 + 1.23298 w_k1_MSFT_s000
                                 + 1.11643 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.44869 w_k1_ADM_s000
                                 + 1.30882 w_k1_AXP_s000
                                 + 1.12788 w_k1_DIS_s000
                                 + 1.06849 w_k1_FDX_s000
                                 + 1.28241 w_k1_IBM_s000
                                 + 1.06413 w_k1_INTC_s000 + 1.2707 w_k1_KO_s000
                                 + 1.24797 w_k1_MAT_s000
                                 + 1.11361 w_k1_MSFT_s000
                                 + 1.09183 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.950842 w_k1_ADM_s000
                                 + 0.989935 w_k1_AXP_s000
                                 + 0.778847 w_k1_DIS_s000
                                 + 0.819876 w_k1_FDX_s000
                                 + 1.15845 w_k1_IBM_s000
                                 + 1.02516 w_k1_INTC_s000
                                 + 1.01531 w_k1_KO_s000
                                 + 0.954352 w_k1_MAT_s000
                                 + 0.970617 w_k1_MSFT_s000
                                 + 0.983447 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.08385 w_k1_ADM_s000
                                 + 0.643086 w_k1_AXP_s000
                                 + 0.802147 w_k1_DIS_s000
                                 + 0.861894 w_k1_FDX_s000
                                 + 0.647293 w_k1_IBM_s000
                                 + 0.964997 w_k1_INTC_s000
                                 + 0.786573 w_k1_KO_s000
                                 + 0.527927 w_k1_MAT_s000
                                 + 0.916563 w_k1_MSFT_s000
                                 + 0.853918 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.3133 w_k1_ADM_s000
                                 + 1.31578 w_k1_AXP_s000
                                 + 1.39947 w_k1_DIS_s000 + 1.3852 w_k1_FDX_s000
                                 + 0.885119 w_k1_IBM_s000
                                 + 1.18958 w_k1_INTC_s000
                                 + 1.08983 w_k1_KO_s000
                                 + 0.990286 w_k1_MAT_s000
                                 + 1.18595 w_k1_MSFT_s000
                                 + 1.08336 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.06844 w_k1_ADM_s000
                                 + 0.913454 w_k1_AXP_s000
                                 + 1.32555 w_k1_DIS_s000
                                 + 1.04017 w_k1_FDX_s000
                                 + 1.86692 w_k1_IBM_s000
                                 + 0.869428 w_k1_INTC_s000
                                 + 1.51004 w_k1_KO_s000 + 1.47185 w_k1_MAT_s000
                                 + 1.03183 w_k1_MSFT_s000
                                 + 1.01731 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.933164 w_k1_ADM_s001
                                 + 1.03258 w_k1_AXP_s001
                                 + 1.25766 w_k1_DIS_s001
                                 + 1.06427 w_k1_FDX_s001
                                 + 1.41453 w_k1_IBM_s001
                                 + 1.18142 w_k1_INTC_s001
                                 + 1.21009 w_k1_KO_s001
                                 + 0.975256 w_k1_MAT_s001
                                 + 1.18184 w_k1_MSFT_s001 + 1.0526 w_k1_PG_s001
                                 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.702167 w_k1_ADM_s001
                                 + 0.733684 w_k1_AXP_s001
                                 + 0.750631 w_k1_DIS_s001
                                 + 0.817483 w_k1_FDX_s001
                                 + 0.874016 w_k1_IBM_s001
                                 + 0.922005 w_k1_INTC_s001
                                 + 0.825829 w_k1_KO_s001
                                 + 1.11911 w_k1_MAT_s001
                                 + 0.871498 w_k1_MSFT_s001
                                 + 0.922012 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.775726 w_k1_ADM_s001
                                 + 1.00624 w_k1_AXP_s001
                                 + 0.94796 w_k1_DIS_s001
                                 + 0.990516 w_k1_FDX_s001
                                 + 1.07919 w_k1_IBM_s001
                                 + 1.24424 w_k1_INTC_s001
                                 + 0.970625 w_k1_KO_s001
                                 + 1.18242 w_k1_MAT_s001
                                 + 0.867397 w_k1_MSFT_s001
                                 + 1.0518 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.926433 w_k1_ADM_s001
                                 + 1.02589 w_k1_AXP_s001
                                 + 1.25625 w_k1_DIS_s001
                                 + 1.02814 w_k1_FDX_s001
                                 + 0.94379 w_k1_IBM_s001
                                 + 1.11071 w_k1_INTC_s001
                                 + 0.980404 w_k1_KO_s001
                                 + 0.860733 w_k1_MAT_s001
                                 + 1.05407 w_k1_MSFT_s001
                                 + 1.03673 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.532057 w_k1_ADM_s001
                                 + 0.591729 w_k1_AXP_s001
                                 + 0.674647 w_k1_DIS_s001
                                 + 0.716971 w_k1_FDX_s001
                                 + 0.744226 w_k1_IBM_s001
                                 + 0.776305 w_k1_INTC_s001
                                 + 0.750208 w_k1_KO_s001
                                 + 0.756466 w_k1_MAT_s001
                                 + 0.856579 w_k1_MSFT_s001
                                 + 0.800872 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.25831 w_k1_ADM_s001
                                 + 1.0525 w_k1_AXP_s001 + 1.5281 w_k1_DIS_s001
                                 + 1.27387 w_k1_FDX_s001
                                 + 0.92564 w_k1_IBM_s001
                                 + 1.22497 w_k1_INTC_s001 + 1.0888 w_k1_KO_s001
                                 + 0.601961 w_k1_MAT_s001
                                 + 1.23211 w_k1_MSFT_s001
                                 + 1.25596 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.25279 w_k1_ADM_s001
                                 + 1.64123 w_k1_AXP_s001
                                 + 1.63338 w_k1_DIS_s001
                                 + 1.38322 w_k1_FDX_s001
                                 + 1.49461 w_k1_IBM_s001
                                 + 1.57429 w_k1_INTC_s001
                                 + 1.33679 w_k1_KO_s001 + 1.27539 w_k1_MAT_s001
                                 + 1.37405 w_k1_MSFT_s001
                                 + 1.18572 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.987439 w_k1_ADM_s001
                                 + 0.725468 w_k1_AXP_s001
                                 + 0.964343 w_k1_DIS_s001
                                 + 1.0048 w_k1_FDX_s001
                                 + 0.650566 w_k1_IBM_s001
                                 + 1.02402 w_k1_INTC_s001
                                 + 0.823634 w_k1_KO_s001
                                 + 0.553381 w_k1_MAT_s001
                                 + 1.03017 w_k1_MSFT_s001
                                 + 1.05582 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.967994 w_k1_ADM_s002
                                 + 1.06856 w_k1_AXP_s002 + 1.2438 w_k1_DIS_s002
                                 + 1.17981 w_k1_FDX_s002 + 0.9656 w_k1_IBM_s002
                                 + 1.03347 w_k1_INTC_s002 + 1.1353 w_k1_KO_s002
                                 + 0.859263 w_k1_MAT_s002
                                 + 1.10846 w_k1_MSFT_s002
                                 + 1.09629 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.930182 w_k1_ADM_s002
                                 + 1.28468 w_k1_AXP_s002 + 1.2544 w_k1_DIS_s002
                                 + 1.06088 w_k1_FDX_s002
                                 + 1.74271 w_k1_IBM_s002
                                 + 1.10523 w_k1_INTC_s002 + 1.2443 w_k1_KO_s002
                                 + 1.28913 w_k1_MAT_s002
                                 + 1.04912 w_k1_MSFT_s002
                                 + 1.15945 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.8957 w_k1_ADM_s002
                                 + 0.757766 w_k1_AXP_s002
                                 + 1.03708 w_k1_DIS_s002
                                 + 0.873694 w_k1_FDX_s002
                                 + 0.971523 w_k1_IBM_s002
                                 + 0.818481 w_k1_INTC_s002
                                 + 0.95082 w_k1_KO_s002
                                 + 0.628347 w_k1_MAT_s002
                                 + 0.922556 w_k1_MSFT_s002
                                 + 1.01307 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.984533 w_k1_ADM_s002
                                 + 0.818943 w_k1_AXP_s002
                                 + 1.18943 w_k1_DIS_s002
                                 + 1.02799 w_k1_FDX_s002
                                 + 0.770143 w_k1_IBM_s002
                                 + 0.854624 w_k1_INTC_s002
                                 + 0.9268 w_k1_KO_s002 + 0.508353 w_k1_MAT_s002
                                 + 1.04144 w_k1_MSFT_s002
                                 + 1.01289 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.913506 w_k1_ADM_s002
                                 + 0.627723 w_k1_AXP_s002
                                 + 0.648742 w_k1_DIS_s002
                                 + 0.750683 w_k1_FDX_s002
                                 + 0.610593 w_k1_IBM_s002
                                 + 0.638618 w_k1_INTC_s002
                                 + 0.636096 w_k1_KO_s002
                                 + 0.409251 w_k1_MAT_s002
                                 + 0.728394 w_k1_MSFT_s002
                                 + 0.777089 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.04746 w_k1_ADM_s002
                                 + 0.833097 w_k1_AXP_s002
                                 + 0.690951 w_k1_DIS_s002
                                 + 0.762691 w_k1_FDX_s002
                                 + 0.980033 w_k1_IBM_s002
                                 + 0.675679 w_k1_INTC_s002
                                 + 0.828946 w_k1_KO_s002
                                 + 0.872388 w_k1_MAT_s002
                                 + 0.797659 w_k1_MSFT_s002
                                 + 0.874568 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.12374 w_k1_ADM_s002
                                 + 1.26132 w_k1_AXP_s002
                                 + 0.953746 w_k1_DIS_s002
                                 + 1.07991 w_k1_FDX_s002
                                 + 0.740831 w_k1_IBM_s002
                                 + 0.807279 w_k1_INTC_s002
                                 + 0.811084 w_k1_KO_s002
                                 + 0.861484 w_k1_MAT_s002
                                 + 0.992565 w_k1_MSFT_s002
                                 + 1.09212 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.45981 w_k1_ADM_s002
                                 + 1.33989 w_k1_AXP_s002 + 1.2857 w_k1_DIS_s002
                                 + 1.17499 w_k1_FDX_s002
                                 + 1.29307 w_k1_IBM_s002
                                 + 1.04995 w_k1_INTC_s002
                                 + 1.11177 w_k1_KO_s002 + 1.03063 w_k1_MAT_s002
                                 + 1.08337 w_k1_MSFT_s002
                                 + 1.13195 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.12171 w_k1_ADM_s003
                                 + 1.06958 w_k1_AXP_s003
                                 + 0.992853 w_k1_DIS_s003
                                 + 1.04604 w_k1_FDX_s003
                                 + 0.949259 w_k1_IBM_s003
                                 + 0.923215 w_k1_INTC_s003
                                 + 0.971586 w_k1_KO_s003
                                 + 1.26058 w_k1_MAT_s003
                                 + 1.02081 w_k1_MSFT_s003
                                 + 1.08012 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.758936 w_k1_ADM_s003
                                 + 0.66504 w_k1_AXP_s003
                                 + 0.692046 w_k1_DIS_s003
                                 + 0.690832 w_k1_FDX_s003
                                 + 1.1237 w_k1_IBM_s003
                                 + 0.869776 w_k1_INTC_s003
                                 + 0.913682 w_k1_KO_s003
                                 + 0.835766 w_k1_MAT_s003
                                 + 0.757983 w_k1_MSFT_s003
                                 + 0.862117 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.00177 w_k1_ADM_s003
                                 + 0.95033 w_k1_AXP_s003
                                 + 0.969631 w_k1_DIS_s003
                                 + 0.956107 w_k1_FDX_s003
                                 + 1.2129 w_k1_IBM_s003
                                 + 1.11015 w_k1_INTC_s003
                                 + 1.04857 w_k1_KO_s003 + 1.30587 w_k1_MAT_s003
                                 + 0.970678 w_k1_MSFT_s003
                                 + 1.12181 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.02017 w_k1_ADM_s003
                                 + 0.84509 w_k1_AXP_s003
                                 + 0.674222 w_k1_DIS_s003
                                 + 0.865541 w_k1_FDX_s003
                                 + 0.754468 w_k1_IBM_s003
                                 + 1.02851 w_k1_INTC_s003
                                 + 0.772541 w_k1_KO_s003
                                 + 0.841203 w_k1_MAT_s003
                                 + 0.890031 w_k1_MSFT_s003
                                 + 0.900929 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.19548 w_k1_ADM_s003
                                 + 1.71676 w_k1_AXP_s003
                                 + 1.66358 w_k1_DIS_s003
                                 + 1.45257 w_k1_FDX_s003
                                 + 1.16787 w_k1_IBM_s003
                                 + 1.36256 w_k1_INTC_s003
                                 + 1.24618 w_k1_KO_s003 + 1.65778 w_k1_MAT_s003
                                 + 1.23802 w_k1_MSFT_s003
                                 + 1.45661 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.3119 w_k1_ADM_s003
                                 + 1.39855 w_k1_AXP_s003
                                 + 1.37313 w_k1_DIS_s003
                                 + 1.15071 w_k1_FDX_s003
                                 + 1.25387 w_k1_IBM_s003
                                 + 1.25072 w_k1_INTC_s003
                                 + 1.39424 w_k1_KO_s003 + 1.71925 w_k1_MAT_s003
                                 + 0.941374 w_k1_MSFT_s003
                                 + 1.19124 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.34451 w_k1_ADM_s003
                                 + 1.16078 w_k1_AXP_s003
                                 + 1.20995 w_k1_DIS_s003
                                 + 1.26688 w_k1_FDX_s003
                                 + 0.821417 w_k1_IBM_s003
                                 + 1.3903 w_k1_INTC_s003
                                 + 0.998179 w_k1_KO_s003
                                 + 1.09925 w_k1_MAT_s003
                                 + 1.12328 w_k1_MSFT_s003
                                 + 1.13955 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.02071 w_k1_ADM_s003
                                 + 1.19478 w_k1_AXP_s003
                                 + 1.23703 w_k1_DIS_s003
                                 + 0.992347 w_k1_FDX_s003
                                 + 1.7478 w_k1_IBM_s003
                                 + 1.18274 w_k1_INTC_s003
                                 + 1.36724 w_k1_KO_s003 + 2.01778 w_k1_MAT_s003
                                 + 1.00367 w_k1_MSFT_s003
                                 + 1.13613 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.914596 w_k1_ADM_s004
                                 + 0.628158 w_k1_AXP_s004
                                 + 0.742435 w_k1_DIS_s004
                                 + 0.818899 w_k1_FDX_s004
                                 + 0.694146 w_k1_IBM_s004
                                 + 0.686126 w_k1_INTC_s004
                                 + 0.749049 w_k1_KO_s004
                                 + 0.74505 w_k1_MAT_s004
                                 + 0.771085 w_k1_MSFT_s004
                                 + 0.919672 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.31781 w_k1_ADM_s004
                                 + 0.904859 w_k1_AXP_s004
                                 + 1.07227 w_k1_DIS_s004
                                 + 1.18376 w_k1_FDX_s004
                                 + 0.685019 w_k1_IBM_s004
                                 + 0.918353 w_k1_INTC_s004
                                 + 0.935872 w_k1_KO_s004
                                 + 0.672355 w_k1_MAT_s004
                                 + 1.00111 w_k1_MSFT_s004 + 1.093 w_k1_PG_s004
                                 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.0979 w_k1_ADM_s004
                                 + 1.00333 w_k1_AXP_s004
                                 + 1.26759 w_k1_DIS_s004
                                 + 1.16337 w_k1_FDX_s004
                                 + 0.960944 w_k1_IBM_s004
                                 + 1.06489 w_k1_INTC_s004 + 1.0171 w_k1_KO_s004
                                 + 0.806387 w_k1_MAT_s004
                                 + 1.09973 w_k1_MSFT_s004
                                 + 1.06283 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.026 w_k1_ADM_s004
                                 + 0.898517 w_k1_AXP_s004
                                 + 1.02015 w_k1_DIS_s004
                                 + 0.986723 w_k1_FDX_s004
                                 + 1.13945 w_k1_IBM_s004
                                 + 1.10787 w_k1_INTC_s004
                                 + 1.15324 w_k1_KO_s004 + 1.19053 w_k1_MAT_s004
                                 + 0.966339 w_k1_MSFT_s004
                                 + 1.06301 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.52825 w_k1_ADM_s004
                                 + 1.18373 w_k1_AXP_s004
                                 + 1.59171 w_k1_DIS_s004
                                 + 1.24529 w_k1_FDX_s004
                                 + 1.19934 w_k1_IBM_s004
                                 + 1.11467 w_k1_INTC_s004 + 1.3555 w_k1_KO_s004
                                 + 1.06724 w_k1_MAT_s004
                                 + 1.39688 w_k1_MSFT_s004
                                 + 1.21617 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.35514 w_k1_ADM_s004
                                 + 1.28821 w_k1_AXP_s004
                                 + 1.35289 w_k1_DIS_s004
                                 + 1.06032 w_k1_FDX_s004
                                 + 1.86608 w_k1_IBM_s004
                                 + 0.791966 w_k1_INTC_s004
                                 + 1.47005 w_k1_KO_s004 + 2.02103 w_k1_MAT_s004
                                 + 0.951268 w_k1_MSFT_s004
                                 + 1.21776 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.02925 w_k1_ADM_s004
                                 + 0.953887 w_k1_AXP_s004
                                 + 0.90234 w_k1_DIS_s004
                                 + 1.02432 w_k1_FDX_s004
                                 + 0.826191 w_k1_IBM_s004
                                 + 0.862828 w_k1_INTC_s004
                                 + 0.869072 w_k1_KO_s004
                                 + 0.819531 w_k1_MAT_s004
                                 + 0.846285 w_k1_MSFT_s004
                                 + 0.991778 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.924254 w_k1_ADM_s004
                                 + 0.798462 w_k1_AXP_s004
                                 + 0.650707 w_k1_DIS_s004
                                 + 0.763549 w_k1_FDX_s004
                                 + 1.08702 w_k1_IBM_s004
                                 + 0.948186 w_k1_INTC_s004
                                 + 0.88303 w_k1_KO_s004
                                 + 0.732285 w_k1_MAT_s004
                                 + 0.81501 w_k1_MSFT_s004 + 0.8697 w_k1_PG_s004
                                 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.908365 w_k1_ADM_s005
                                 + 1.08202 w_k1_AXP_s005
                                 + 1.08617 w_k1_DIS_s005
                                 + 1.06357 w_k1_FDX_s005
                                 + 0.660722 w_k1_IBM_s005
                                 + 1.2051 w_k1_INTC_s005
                                 + 0.939106 w_k1_KO_s005
                                 + 1.30126 w_k1_MAT_s005
                                 + 1.21352 w_k1_MSFT_s005
                                 + 0.973769 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.921619 w_k1_ADM_s005
                                 + 0.850358 w_k1_AXP_s005
                                 + 0.874755 w_k1_DIS_s005
                                 + 0.867749 w_k1_FDX_s005
                                 + 1.05441 w_k1_IBM_s005
                                 + 1.10302 w_k1_INTC_s005
                                 + 1.09322 w_k1_KO_s005 + 1.22611 w_k1_MAT_s005
                                 + 1.06382 w_k1_MSFT_s005
                                 + 0.970512 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.786471 w_k1_ADM_s005
                                 + 0.703765 w_k1_AXP_s005
                                 + 0.689982 w_k1_DIS_s005
                                 + 0.89299 w_k1_FDX_s005
                                 + 0.619769 w_k1_IBM_s005
                                 + 0.866391 w_k1_INTC_s005
                                 + 0.797986 w_k1_KO_s005
                                 + 0.809238 w_k1_MAT_s005
                                 + 1.00023 w_k1_MSFT_s005
                                 + 0.863615 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.08799 w_k1_ADM_s005
                                 + 1.33431 w_k1_AXP_s005
                                 + 1.40253 w_k1_DIS_s005 + 1.2716 w_k1_FDX_s005
                                 + 0.934882 w_k1_IBM_s005
                                 + 1.48691 w_k1_INTC_s005
                                 + 1.11167 w_k1_KO_s005 + 1.28703 w_k1_MAT_s005
                                 + 1.45838 w_k1_MSFT_s005
                                 + 1.18794 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.13159 w_k1_ADM_s005
                                 + 1.0659 w_k1_AXP_s005
                                 + 0.985758 w_k1_DIS_s005
                                 + 1.06377 w_k1_FDX_s005
                                 + 0.980741 w_k1_IBM_s005
                                 + 1.03652 w_k1_INTC_s005
                                 + 1.09989 w_k1_KO_s005 + 1.22567 w_k1_MAT_s005
                                 + 1.0941 w_k1_MSFT_s005 + 1.01971 w_k1_PG_s005
                                 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.683339 w_k1_ADM_s005
                                 + 0.686304 w_k1_AXP_s005
                                 + 0.549203 w_k1_DIS_s005
                                 + 0.651615 w_k1_FDX_s005
                                 + 0.864536 w_k1_IBM_s005
                                 + 0.825224 w_k1_INTC_s005
                                 + 0.737512 w_k1_KO_s005
                                 + 1.04945 w_k1_MAT_s005
                                 + 0.770947 w_k1_MSFT_s005
                                 + 0.785732 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.13019 w_k1_ADM_s005
                                 + 0.998028 w_k1_AXP_s005
                                 + 0.899096 w_k1_DIS_s005
                                 + 0.852169 w_k1_FDX_s005
                                 + 1.40084 w_k1_IBM_s005
                                 + 1.12247 w_k1_INTC_s005 + 1.0873 w_k1_KO_s005
                                 + 1.45037 w_k1_MAT_s005
                                 + 1.00071 w_k1_MSFT_s005
                                 + 1.10378 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.04253 w_k1_ADM_s005
                                 + 1.22328 w_k1_AXP_s005
                                 + 1.46191 w_k1_DIS_s005 + 1.1543 w_k1_FDX_s005
                                 + 1.42152 w_k1_IBM_s005
                                 + 1.18172 w_k1_INTC_s005
                                 + 1.47384 w_k1_KO_s005 + 1.63538 w_k1_MAT_s005
                                 + 1.23622 w_k1_MSFT_s005 + 1.2234 w_k1_PG_s005
                                 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.793149 w_k1_ADM_s006
                                 + 0.776909 w_k1_AXP_s006
                                 + 0.676881 w_k1_DIS_s006
                                 + 0.749719 w_k1_FDX_s006
                                 + 0.664279 w_k1_IBM_s006
                                 + 0.909617 w_k1_INTC_s006
                                 + 0.652709 w_k1_KO_s006
                                 + 0.667401 w_k1_MAT_s006
                                 + 0.806826 w_k1_MSFT_s006
                                 + 0.86179 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.02788 w_k1_ADM_s006
                                 + 1.31373 w_k1_AXP_s006
                                 + 1.13264 w_k1_DIS_s006
                                 + 1.12745 w_k1_FDX_s006
                                 + 0.995161 w_k1_IBM_s006
                                 + 1.04667 w_k1_INTC_s006
                                 + 0.996049 w_k1_KO_s006
                                 + 1.79457 w_k1_MAT_s006
                                 + 1.02306 w_k1_MSFT_s006
                                 + 1.05823 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.844441 w_k1_ADM_s006
                                 + 0.904083 w_k1_AXP_s006
                                 + 1.13242 w_k1_DIS_s006
                                 + 0.945678 w_k1_FDX_s006
                                 + 0.773734 w_k1_IBM_s006
                                 + 1.09635 w_k1_INTC_s006
                                 + 0.929711 w_k1_KO_s006
                                 + 1.20811 w_k1_MAT_s006
                                 + 1.09636 w_k1_MSFT_s006
                                 + 0.942802 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.06794 w_k1_ADM_s006
                                 + 1.09554 w_k1_AXP_s006
                                 + 1.26209 w_k1_DIS_s006
                                 + 1.24056 w_k1_FDX_s006
                                 + 0.637727 w_k1_IBM_s006
                                 + 1.04915 w_k1_INTC_s006
                                 + 0.853698 w_k1_KO_s006
                                 + 0.745538 w_k1_MAT_s006
                                 + 1.24228 w_k1_MSFT_s006
                                 + 1.08809 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.986953 w_k1_ADM_s006
                                 + 0.943515 w_k1_AXP_s006
                                 + 0.843788 w_k1_DIS_s006
                                 + 0.931122 w_k1_FDX_s006
                                 + 0.878503 w_k1_IBM_s006
                                 + 0.954777 w_k1_INTC_s006
                                 + 0.930357 w_k1_KO_s006
                                 + 1.06659 w_k1_MAT_s006
                                 + 1.0337 w_k1_MSFT_s006 + 0.9726 w_k1_PG_s006
                                 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.856295 w_k1_ADM_s006
                                 + 0.693336 w_k1_AXP_s006
                                 + 0.866832 w_k1_DIS_s006
                                 + 0.811773 w_k1_FDX_s006
                                 + 0.767402 w_k1_IBM_s006
                                 + 0.810337 w_k1_INTC_s006
                                 + 0.825431 w_k1_KO_s006
                                 + 1.04907 w_k1_MAT_s006
                                 + 1.00716 w_k1_MSFT_s006
                                 + 0.916796 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.98548 w_k1_ADM_s006
                                 + 1.09002 w_k1_AXP_s006
                                 + 0.938689 w_k1_DIS_s006
                                 + 0.904063 w_k1_FDX_s006
                                 + 1.32278 w_k1_IBM_s006
                                 + 1.12194 w_k1_INTC_s006
                                 + 1.11767 w_k1_KO_s006 + 1.88251 w_k1_MAT_s006
                                 + 1.00304 w_k1_MSFT_s006
                                 + 1.05653 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.23859 w_k1_ADM_s006
                                 + 1.47074 w_k1_AXP_s006
                                 + 1.61375 w_k1_DIS_s006
                                 + 1.30761 w_k1_FDX_s006
                                 + 1.26775 w_k1_IBM_s006
                                 + 1.2075 w_k1_INTC_s006 + 1.44527 w_k1_KO_s006
                                 + 2.06545 w_k1_MAT_s006
                                 + 1.46922 w_k1_MSFT_s006 + 1.2155 w_k1_PG_s006
                                 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.07307 w_k1_ADM_s007
                                 + 0.738099 w_k1_AXP_s007
                                 + 0.627492 w_k1_DIS_s007
                                 + 0.721263 w_k1_FDX_s007
                                 + 0.929969 w_k1_IBM_s007
                                 + 0.776213 w_k1_INTC_s007
                                 + 0.828581 w_k1_KO_s007
                                 + 0.866241 w_k1_MAT_s007
                                 + 0.830958 w_k1_MSFT_s007
                                 + 0.846294 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.46242 w_k1_ADM_s007
                                 + 1.37274 w_k1_AXP_s007
                                 + 1.29564 w_k1_DIS_s007
                                 + 1.31371 w_k1_FDX_s007
                                 + 1.00411 w_k1_IBM_s007
                                 + 1.07687 w_k1_INTC_s007
                                 + 1.10641 w_k1_KO_s007 + 1.46258 w_k1_MAT_s007
                                 + 1.3352 w_k1_MSFT_s007 + 1.3827 w_k1_PG_s007
                                 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.30239 w_k1_ADM_s007
                                 + 1.33641 w_k1_AXP_s007
                                 + 1.28619 w_k1_DIS_s007
                                 + 1.18761 w_k1_FDX_s007
                                 + 1.52082 w_k1_IBM_s007
                                 + 1.11938 w_k1_INTC_s007
                                 + 1.23384 w_k1_KO_s007 + 1.41156 w_k1_MAT_s007
                                 + 1.238 w_k1_MSFT_s007 + 1.15813 w_k1_PG_s007
                                 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.937657 w_k1_ADM_s007
                                 + 0.946796 w_k1_AXP_s007
                                 + 1.26016 w_k1_DIS_s007
                                 + 1.19111 w_k1_FDX_s007
                                 + 0.905428 w_k1_IBM_s007
                                 + 0.861297 w_k1_INTC_s007
                                 + 0.964164 w_k1_KO_s007
                                 + 0.839323 w_k1_MAT_s007
                                 + 1.25554 w_k1_MSFT_s007
                                 + 1.14441 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.924406 w_k1_ADM_s007
                                 + 0.732217 w_k1_AXP_s007
                                 + 0.799419 w_k1_DIS_s007
                                 + 0.875072 w_k1_FDX_s007
                                 + 0.891707 w_k1_IBM_s007
                                 + 0.724787 w_k1_INTC_s007
                                 + 0.891004 w_k1_KO_s007
                                 + 0.828714 w_k1_MAT_s007
                                 + 1.07764 w_k1_MSFT_s007
                                 + 0.976372 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.27771 w_k1_ADM_s007
                                 + 0.951112 w_k1_AXP_s007
                                 + 1.02098 w_k1_DIS_s007
                                 + 0.939052 w_k1_FDX_s007
                                 + 1.28242 w_k1_IBM_s007
                                 + 0.812839 w_k1_INTC_s007
                                 + 1.07205 w_k1_KO_s007 + 1.18266 w_k1_MAT_s007
                                 + 0.940538 w_k1_MSFT_s007
                                 + 1.05891 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.17777 w_k1_ADM_s007
                                 + 1.11835 w_k1_AXP_s007
                                 + 0.855357 w_k1_DIS_s007
                                 + 1.07605 w_k1_FDX_s007
                                 + 0.850573 w_k1_IBM_s007
                                 + 0.840919 w_k1_INTC_s007
                                 + 0.792653 w_k1_KO_s007
                                 + 0.982491 w_k1_MAT_s007
                                 + 0.948061 w_k1_MSFT_s007
                                 + 0.991278 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.35026 w_k1_ADM_s007
                                 + 1.15835 w_k1_AXP_s007
                                 + 1.92219 w_k1_DIS_s007
                                 + 1.32762 w_k1_FDX_s007
                                 + 1.24475 w_k1_IBM_s007
                                 + 1.36749 w_k1_INTC_s007
                                 + 1.42612 w_k1_KO_s007 + 1.03846 w_k1_MAT_s007
                                 + 1.5106 w_k1_MSFT_s007 + 1.27226 w_k1_PG_s007
                                 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.841444 w_k1_ADM_s010
                                 + 1.0603 w_k1_AXP_s010 + 1.16089 w_k1_DIS_s010
                                 + 0.983117 w_k1_FDX_s010
                                 + 1.30265 w_k1_IBM_s010
                                 + 0.868691 w_k1_INTC_s010
                                 + 1.05852 w_k1_KO_s010 + 1.09512 w_k1_MAT_s010
                                 + 0.920062 w_k1_MSFT_s010
                                 + 1.11213 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.2574 w_k1_ADM_s010
                                 + 1.5152 w_k1_AXP_s010 + 1.93459 w_k1_DIS_s010
                                 + 1.43462 w_k1_FDX_s010
                                 + 1.15868 w_k1_IBM_s010
                                 + 1.29238 w_k1_INTC_s010
                                 + 1.27471 w_k1_KO_s010 + 1.35952 w_k1_MAT_s010
                                 + 1.50678 w_k1_MSFT_s010
                                 + 1.23694 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.864555 w_k1_ADM_s010
                                 + 0.909662 w_k1_AXP_s010
                                 + 1.01284 w_k1_DIS_s010
                                 + 0.938811 w_k1_FDX_s010
                                 + 0.780654 w_k1_IBM_s010
                                 + 0.789125 w_k1_INTC_s010
                                 + 0.946748 w_k1_KO_s010
                                 + 0.792134 w_k1_MAT_s010
                                 + 0.934427 w_k1_MSFT_s010
                                 + 0.963706 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.15562 w_k1_ADM_s010
                                 + 1.50084 w_k1_AXP_s010
                                 + 1.34542 w_k1_DIS_s010
                                 + 1.18646 w_k1_FDX_s010 + 1.462 w_k1_IBM_s010
                                 + 0.978357 w_k1_INTC_s010
                                 + 1.1494 w_k1_KO_s010 + 2.2986 w_k1_MAT_s010
                                 + 1.09667 w_k1_MSFT_s010
                                 + 1.25797 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.987171 w_k1_ADM_s010
                                 + 1.09492 w_k1_AXP_s010
                                 + 0.883734 w_k1_DIS_s010
                                 + 1.04726 w_k1_FDX_s010
                                 + 0.83548 w_k1_IBM_s010
                                 + 1.02454 w_k1_INTC_s010
                                 + 0.852402 w_k1_KO_s010
                                 + 1.12211 w_k1_MAT_s010
                                 + 0.96484 w_k1_MSFT_s010
                                 + 1.04263 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.04149 w_k1_ADM_s010
                                 + 1.2982 w_k1_AXP_s010 + 1.27445 w_k1_DIS_s010
                                 + 1.21021 w_k1_FDX_s010
                                 + 0.898164 w_k1_IBM_s010
                                 + 1.04056 w_k1_INTC_s010
                                 + 1.02457 w_k1_KO_s010 + 1.09834 w_k1_MAT_s010
                                 + 1.10353 w_k1_MSFT_s010 + 1.1703 w_k1_PG_s010
                                 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.850216 w_k1_ADM_s010
                                 + 0.774004 w_k1_AXP_s010
                                 + 0.744466 w_k1_DIS_s010
                                 + 0.758735 w_k1_FDX_s010
                                 + 0.928757 w_k1_IBM_s010
                                 + 0.724706 w_k1_INTC_s010
                                 + 0.825871 w_k1_KO_s010
                                 + 0.798159 w_k1_MAT_s010
                                 + 0.750244 w_k1_MSFT_s010
                                 + 0.93302 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.882573 w_k1_ADM_s010
                                 + 1.06261 w_k1_AXP_s010
                                 + 0.723848 w_k1_DIS_s010
                                 + 0.80715 w_k1_FDX_s010
                                 + 1.21089 w_k1_IBM_s010
                                 + 0.829421 w_k1_INTC_s010
                                 + 0.835904 w_k1_KO_s010
                                 + 1.41014 w_k1_MAT_s010
                                 + 0.794545 w_k1_MSFT_s010
                                 + 0.987483 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.857958 w_k1_ADM_s011
                                 + 0.793089 w_k1_AXP_s011
                                 + 0.73254 w_k1_DIS_s011
                                 + 0.736175 w_k1_FDX_s011
                                 + 0.866486 w_k1_IBM_s011
                                 + 0.733222 w_k1_INTC_s011
                                 + 0.836478 w_k1_KO_s011
                                 + 1.17921 w_k1_MAT_s011
                                 + 0.882431 w_k1_MSFT_s011
                                 + 0.849522 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.35728 w_k1_ADM_s011
                                 + 1.18897 w_k1_AXP_s011
                                 + 1.40458 w_k1_DIS_s011 + 1.2291 w_k1_FDX_s011
                                 + 0.973497 w_k1_IBM_s011
                                 + 0.969146 w_k1_INTC_s011
                                 + 1.16206 w_k1_KO_s011 + 1.26803 w_k1_MAT_s011
                                 + 1.2914 w_k1_MSFT_s011 + 1.06187 w_k1_PG_s011
                                 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.901467 w_k1_ADM_s011
                                 + 0.694849 w_k1_AXP_s011
                                 + 0.896084 w_k1_DIS_s011
                                 + 0.987965 w_k1_FDX_s011
                                 + 0.638447 w_k1_IBM_s011
                                 + 0.980654 w_k1_INTC_s011
                                 + 0.739425 w_k1_KO_s011
                                 + 0.422242 w_k1_MAT_s011
                                 + 1.05804 w_k1_MSFT_s011
                                 + 0.934956 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.25676 w_k1_ADM_s011
                                 + 1.50718 w_k1_AXP_s011
                                 + 1.47616 w_k1_DIS_s011
                                 + 1.22677 w_k1_FDX_s011
                                 + 1.32433 w_k1_IBM_s011
                                 + 0.921628 w_k1_INTC_s011
                                 + 1.28743 w_k1_KO_s011 + 2.34975 w_k1_MAT_s011
                                 + 1.22105 w_k1_MSFT_s011
                                 + 1.09218 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.55901 w_k1_ADM_s011
                                 + 1.5756 w_k1_AXP_s011 + 1.20191 w_k1_DIS_s011
                                 + 1.34241 w_k1_FDX_s011
                                 + 0.93935 w_k1_IBM_s011
                                 + 1.06717 w_k1_INTC_s011
                                 + 1.17671 w_k1_KO_s011 + 1.37422 w_k1_MAT_s011
                                 + 1.21689 w_k1_MSFT_s011
                                 + 1.15585 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.32899 w_k1_ADM_s011
                                 + 1.38869 w_k1_AXP_s011
                                 + 1.45591 w_k1_DIS_s011
                                 + 1.13389 w_k1_FDX_s011
                                 + 1.63553 w_k1_IBM_s011
                                 + 1.01967 w_k1_INTC_s011
                                 + 1.49751 w_k1_KO_s011 + 1.26692 w_k1_MAT_s011
                                 + 1.07529 w_k1_MSFT_s011
                                 + 1.06153 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.900177 w_k1_ADM_s011
                                 + 0.973508 w_k1_AXP_s011
                                 + 1.06788 w_k1_DIS_s011
                                 + 0.928313 w_k1_FDX_s011
                                 + 1.22787 w_k1_IBM_s011
                                 + 0.859671 w_k1_INTC_s011
                                 + 1.06918 w_k1_KO_s011 + 1.10266 w_k1_MAT_s011
                                 + 0.921727 w_k1_MSFT_s011
                                 + 1.01439 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.00577 w_k1_ADM_s011
                                 + 1.04928 w_k1_AXP_s011
                                 + 1.09016 w_k1_DIS_s011
                                 + 1.04851 w_k1_FDX_s011
                                 + 0.817964 w_k1_IBM_s011
                                 + 0.914735 w_k1_INTC_s011
                                 + 0.919023 w_k1_KO_s011
                                 + 1.26143 w_k1_MAT_s011
                                 + 1.06185 w_k1_MSFT_s011 + 1.0037 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.783625 w_k1_ADM_s012
                                 + 0.792842 w_k1_AXP_s012
                                 + 0.667776 w_k1_DIS_s012
                                 + 0.730514 w_k1_FDX_s012
                                 + 0.999624 w_k1_IBM_s012
                                 + 1.14255 w_k1_INTC_s012
                                 + 0.777225 w_k1_KO_s012
                                 + 0.672457 w_k1_MAT_s012
                                 + 0.685173 w_k1_MSFT_s012
                                 + 0.754025 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.992027 w_k1_ADM_s012
                                 + 1.13067 w_k1_AXP_s012
                                 + 1.34004 w_k1_DIS_s012
                                 + 1.03888 w_k1_FDX_s012
                                 + 1.58575 w_k1_IBM_s012
                                 + 0.93142 w_k1_INTC_s012
                                 + 1.34615 w_k1_KO_s012 + 1.44101 w_k1_MAT_s012
                                 + 0.999872 w_k1_MSFT_s012
                                 + 1.17761 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.903366 w_k1_ADM_s012
                                 + 1.21295 w_k1_AXP_s012 + 1.5287 w_k1_DIS_s012
                                 + 1.16106 w_k1_FDX_s012
                                 + 1.26834 w_k1_IBM_s012
                                 + 1.23561 w_k1_INTC_s012
                                 + 1.21233 w_k1_KO_s012 + 1.19089 w_k1_MAT_s012
                                 + 1.08017 w_k1_MSFT_s012
                                 + 1.19494 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.13582 w_k1_ADM_s012
                                 + 1.11069 w_k1_AXP_s012
                                 + 1.28965 w_k1_DIS_s012
                                 + 1.17442 w_k1_FDX_s012
                                 + 0.847414 w_k1_IBM_s012
                                 + 1.15042 w_k1_INTC_s012
                                 + 0.951445 w_k1_KO_s012
                                 + 0.79452 w_k1_MAT_s012
                                 + 1.1665 w_k1_MSFT_s012 + 1.10015 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.807063 w_k1_ADM_s012
                                 + 0.927009 w_k1_AXP_s012
                                 + 0.870962 w_k1_DIS_s012
                                 + 0.789844 w_k1_FDX_s012
                                 + 1.44136 w_k1_IBM_s012
                                 + 0.997595 w_k1_INTC_s012
                                 + 0.978141 w_k1_KO_s012
                                 + 0.998548 w_k1_MAT_s012
                                 + 0.788383 w_k1_MSFT_s012
                                 + 0.917064 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.858738 w_k1_ADM_s012
                                 + 0.973919 w_k1_AXP_s012
                                 + 0.960333 w_k1_DIS_s012
                                 + 1.0547 w_k1_FDX_s012
                                 + 0.669316 w_k1_IBM_s012
                                 + 1.12444 w_k1_INTC_s012
                                 + 0.80179 w_k1_KO_s012 + 0.62718 w_k1_MAT_s012
                                 + 0.867716 w_k1_MSFT_s012
                                 + 0.940606 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.1235 w_k1_ADM_s012
                                 + 1.55538 w_k1_AXP_s012
                                 + 2.07716 w_k1_DIS_s012
                                 + 1.45652 w_k1_FDX_s012
                                 + 1.04213 w_k1_IBM_s012
                                 + 1.61005 w_k1_INTC_s012
                                 + 1.26207 w_k1_KO_s012 + 1.16154 w_k1_MAT_s012
                                 + 1.33436 w_k1_MSFT_s012
                                 + 1.09909 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.852032 w_k1_ADM_s012
                                 + 1.0842 w_k1_AXP_s012
                                 + 0.982238 w_k1_DIS_s012
                                 + 1.00427 w_k1_FDX_s012
                                 + 1.00195 w_k1_IBM_s012
                                 + 1.10955 w_k1_INTC_s012
                                 + 0.812174 w_k1_KO_s012 + 1.0342 w_k1_MAT_s012
                                 + 0.857386 w_k1_MSFT_s012
                                 + 0.904261 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.38663 w_k1_ADM_s013
                                 + 1.64991 w_k1_AXP_s013
                                 + 1.65596 w_k1_DIS_s013
                                 + 1.42753 w_k1_FDX_s013
                                 + 0.737586 w_k1_IBM_s013
                                 + 0.985585 w_k1_INTC_s013
                                 + 1.21672 w_k1_KO_s013
                                 + 0.952288 w_k1_MAT_s013
                                 + 1.38908 w_k1_MSFT_s013
                                 + 1.28984 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.865626 w_k1_ADM_s013
                                 + 0.735667 w_k1_AXP_s013
                                 + 0.820425 w_k1_DIS_s013
                                 + 0.696742 w_k1_FDX_s013
                                 + 1.22444 w_k1_IBM_s013
                                 + 0.596905 w_k1_INTC_s013
                                 + 1.07398 w_k1_KO_s013 + 1.14908 w_k1_MAT_s013
                                 + 0.813728 w_k1_MSFT_s013
                                 + 0.917076 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.33848 w_k1_ADM_s013
                                 + 1.38873 w_k1_AXP_s013
                                 + 1.65964 w_k1_DIS_s013
                                 + 1.25685 w_k1_FDX_s013
                                 + 1.29646 w_k1_IBM_s013
                                 + 1.07723 w_k1_INTC_s013
                                 + 1.38492 w_k1_KO_s013 + 1.5055 w_k1_MAT_s013
                                 + 1.36765 w_k1_MSFT_s013
                                 + 1.28064 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.01434 w_k1_ADM_s013
                                 + 1.55039 w_k1_AXP_s013
                                 + 1.08621 w_k1_DIS_s013
                                 + 0.959784 w_k1_FDX_s013
                                 + 1.85235 w_k1_IBM_s013
                                 + 0.957841 w_k1_INTC_s013
                                 + 1.36681 w_k1_KO_s013 + 1.76111 w_k1_MAT_s013
                                 + 0.869002 w_k1_MSFT_s013
                                 + 1.09019 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.23245 w_k1_ADM_s013
                                 + 1.26085 w_k1_AXP_s013
                                 + 1.29887 w_k1_DIS_s013
                                 + 1.12034 w_k1_FDX_s013
                                 + 0.903472 w_k1_IBM_s013
                                 + 0.919946 w_k1_INTC_s013
                                 + 0.980954 w_k1_KO_s013
                                 + 0.945373 w_k1_MAT_s013
                                 + 1.10611 w_k1_MSFT_s013
                                 + 1.05669 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.1152 w_k1_ADM_s013
                                 + 1.32844 w_k1_AXP_s013
                                 + 1.01945 w_k1_DIS_s013
                                 + 1.02269 w_k1_FDX_s013
                                 + 1.09201 w_k1_IBM_s013
                                 + 0.95161 w_k1_INTC_s013
                                 + 1.27624 w_k1_KO_s013 + 1.21187 w_k1_MAT_s013
                                 + 1.01285 w_k1_MSFT_s013
                                 + 1.10574 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.837248 w_k1_ADM_s013
                                 + 0.993931 w_k1_AXP_s013
                                 + 0.827995 w_k1_DIS_s013
                                 + 0.876235 w_k1_FDX_s013
                                 + 0.733307 w_k1_IBM_s013
                                 + 0.794433 w_k1_INTC_s013
                                 + 0.793787 w_k1_KO_s013
                                 + 0.752041 w_k1_MAT_s013
                                 + 0.992486 w_k1_MSFT_s013
                                 + 0.846679 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.674958 w_k1_ADM_s013
                                 + 0.775757 w_k1_AXP_s013
                                 + 0.632657 w_k1_DIS_s013
                                 + 0.642366 w_k1_FDX_s013
                                 + 0.833563 w_k1_IBM_s013
                                 + 0.617 w_k1_INTC_s013 + 0.708262 w_k1_KO_s013
                                 + 0.636299 w_k1_MAT_s013
                                 + 0.689223 w_k1_MSFT_s013
                                 + 0.799479 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.02226 w_k1_ADM_s014
                                 + 0.814082 w_k1_AXP_s014
                                 + 0.848609 w_k1_DIS_s014
                                 + 0.908806 w_k1_FDX_s014
                                 + 0.660389 w_k1_IBM_s014
                                 + 0.678594 w_k1_INTC_s014
                                 + 0.789634 w_k1_KO_s014
                                 + 0.737824 w_k1_MAT_s014
                                 + 0.856657 w_k1_MSFT_s014
                                 + 0.88405 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.56974 w_k1_ADM_s014
                                 + 1.47877 w_k1_AXP_s014
                                 + 1.66489 w_k1_DIS_s014
                                 + 1.34815 w_k1_FDX_s014
                                 + 1.19209 w_k1_IBM_s014
                                 + 0.982277 w_k1_INTC_s014
                                 + 1.35056 w_k1_KO_s014 + 1.31803 w_k1_MAT_s014
                                 + 1.08821 w_k1_MSFT_s014
                                 + 1.20557 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.0323 w_k1_ADM_s014
                                 + 0.75457 w_k1_AXP_s014
                                 + 0.700231 w_k1_DIS_s014
                                 + 0.710781 w_k1_FDX_s014
                                 + 0.915042 w_k1_IBM_s014
                                 + 0.81434 w_k1_INTC_s014
                                 + 0.91448 w_k1_KO_s014 + 1.05202 w_k1_MAT_s014
                                 + 0.787939 w_k1_MSFT_s014
                                 + 0.822811 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.16198 w_k1_ADM_s014
                                 + 1.16859 w_k1_AXP_s014
                                 + 1.22779 w_k1_DIS_s014
                                 + 1.19381 w_k1_FDX_s014
                                 + 0.920379 w_k1_IBM_s014
                                 + 0.963656 w_k1_INTC_s014
                                 + 1.11448 w_k1_KO_s014
                                 + 0.997471 w_k1_MAT_s014
                                 + 1.10414 w_k1_MSFT_s014
                                 + 1.16646 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.974379 w_k1_ADM_s014
                                 + 1.10317 w_k1_AXP_s014
                                 + 0.784769 w_k1_DIS_s014
                                 + 0.923427 w_k1_FDX_s014
                                 + 0.939971 w_k1_IBM_s014
                                 + 0.839996 w_k1_INTC_s014
                                 + 0.905467 w_k1_KO_s014
                                 + 1.04731 w_k1_MAT_s014
                                 + 0.85857 w_k1_MSFT_s014
                                 + 0.946919 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.07244 w_k1_ADM_s014
                                 + 1.05498 w_k1_AXP_s014
                                 + 0.941722 w_k1_DIS_s014
                                 + 0.86447 w_k1_FDX_s014
                                 + 1.37422 w_k1_IBM_s014
                                 + 0.800773 w_k1_INTC_s014
                                 + 1.19553 w_k1_KO_s014 + 1.37687 w_k1_MAT_s014
                                 + 0.724373 w_k1_MSFT_s014
                                 + 1.11899 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.1291 w_k1_ADM_s014
                                 + 0.91257 w_k1_AXP_s014
                                 + 1.15546 w_k1_DIS_s014 + 1.1003 w_k1_FDX_s014
                                 + 0.747741 w_k1_IBM_s014
                                 + 1.04486 w_k1_INTC_s014
                                 + 1.04367 w_k1_KO_s014
                                 + 0.820188 w_k1_MAT_s014
                                 + 1.08836 w_k1_MSFT_s014
                                 + 1.05317 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.07795 w_k1_ADM_s014
                                 + 1.17544 w_k1_AXP_s014
                                 + 1.39314 w_k1_DIS_s014
                                 + 1.04471 w_k1_FDX_s014
                                 + 1.36412 w_k1_IBM_s014
                                 + 0.898442 w_k1_INTC_s014
                                 + 1.27855 w_k1_KO_s014 + 1.62801 w_k1_MAT_s014
                                 + 1.01871 w_k1_MSFT_s014
                                 + 1.19723 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.20888 w_k1_ADM_s015
                                 + 1.24182 w_k1_AXP_s015
                                 + 1.04002 w_k1_DIS_s015
                                 + 1.16348 w_k1_FDX_s015
                                 + 1.14357 w_k1_IBM_s015
                                 + 1.12852 w_k1_INTC_s015 + 1.1443 w_k1_KO_s015
                                 + 1.26834 w_k1_MAT_s015
                                 + 1.11371 w_k1_MSFT_s015
                                 + 1.00354 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.1027 w_k1_ADM_s015
                                 + 1.23019 w_k1_AXP_s015
                                 + 0.789686 w_k1_DIS_s015
                                 + 1.03158 w_k1_FDX_s015
                                 + 0.78421 w_k1_IBM_s015
                                 + 1.02639 w_k1_INTC_s015
                                 + 0.74094 w_k1_KO_s015 + 1.04909 w_k1_MAT_s015
                                 + 0.838822 w_k1_MSFT_s015
                                 + 0.966971 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.923384 w_k1_ADM_s015
                                 + 0.633273 w_k1_AXP_s015
                                 + 0.838642 w_k1_DIS_s015
                                 + 0.804325 w_k1_FDX_s015
                                 + 0.986229 w_k1_IBM_s015
                                 + 0.917723 w_k1_INTC_s015
                                 + 1.07847 w_k1_KO_s015
                                 + 0.942302 w_k1_MAT_s015
                                 + 0.898668 w_k1_MSFT_s015
                                 + 0.976484 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.10858 w_k1_ADM_s015
                                 + 1.32639 w_k1_AXP_s015
                                 + 1.49745 w_k1_DIS_s015
                                 + 1.21165 w_k1_FDX_s015
                                 + 1.42388 w_k1_IBM_s015
                                 + 1.30707 w_k1_INTC_s015
                                 + 1.53089 w_k1_KO_s015 + 1.55418 w_k1_MAT_s015
                                 + 1.31867 w_k1_MSFT_s015
                                 + 1.10787 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.05817 w_k1_ADM_s015
                                 + 0.994081 w_k1_AXP_s015
                                 + 0.970683 w_k1_DIS_s015
                                 + 1.01986 w_k1_FDX_s015
                                 + 0.830309 w_k1_IBM_s015
                                 + 1.05876 w_k1_INTC_s015
                                 + 0.907029 w_k1_KO_s015
                                 + 0.865286 w_k1_MAT_s015
                                 + 0.95825 w_k1_MSFT_s015
                                 + 0.998258 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.992847 w_k1_ADM_s015
                                 + 0.743578 w_k1_AXP_s015
                                 + 0.6832 w_k1_DIS_s015
                                 + 0.802503 w_k1_FDX_s015
                                 + 0.736113 w_k1_IBM_s015
                                 + 0.856271 w_k1_INTC_s015
                                 + 0.781898 w_k1_KO_s015
                                 + 0.656664 w_k1_MAT_s015
                                 + 0.758241 w_k1_MSFT_s015
                                 + 0.808339 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.28495 w_k1_ADM_s015
                                 + 1.34117 w_k1_AXP_s015 + 1.2588 w_k1_DIS_s015
                                 + 1.37396 w_k1_FDX_s015
                                 + 0.871751 w_k1_IBM_s015
                                 + 1.32477 w_k1_INTC_s015
                                 + 1.28807 w_k1_KO_s015 + 1.14539 w_k1_MAT_s015
                                 + 1.34671 w_k1_MSFT_s015
                                 + 1.06626 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.744121 w_k1_ADM_s015
                                 + 0.786887 w_k1_AXP_s015
                                 + 0.656289 w_k1_DIS_s015
                                 + 0.717961 w_k1_FDX_s015
                                 + 0.971129 w_k1_IBM_s015
                                 + 0.930775 w_k1_INTC_s015
                                 + 0.83619 w_k1_KO_s015
                                 + 0.934077 w_k1_MAT_s015
                                 + 0.743915 w_k1_MSFT_s015
                                 + 0.775092 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.13531 w_k1_ADM_s016
                                 + 1.37643 w_k1_AXP_s016
                                 + 1.49499 w_k1_DIS_s016
                                 + 1.22972 w_k1_FDX_s016
                                 + 0.951896 w_k1_IBM_s016
                                 + 1.06009 w_k1_INTC_s016
                                 + 1.06272 w_k1_KO_s016
                                 + 0.865148 w_k1_MAT_s016
                                 + 1.38256 w_k1_MSFT_s016
                                 + 1.17108 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.87643 w_k1_ADM_s016
                                 + 1.04139 w_k1_AXP_s016
                                 + 1.00807 w_k1_DIS_s016
                                 + 1.03466 w_k1_FDX_s016
                                 + 0.724525 w_k1_IBM_s016
                                 + 1.09036 w_k1_INTC_s016
                                 + 0.820161 w_k1_KO_s016
                                 + 0.739125 w_k1_MAT_s016
                                 + 1.1361 w_k1_MSFT_s016 + 0.95584 w_k1_PG_s016
                                 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.07212 w_k1_ADM_s016
                                 + 1.02528 w_k1_AXP_s016
                                 + 1.17328 w_k1_DIS_s016
                                 + 1.01746 w_k1_FDX_s016
                                 + 1.33114 w_k1_IBM_s016
                                 + 1.03041 w_k1_INTC_s016 + 1.1857 w_k1_KO_s016
                                 + 1.15217 w_k1_MAT_s016
                                 + 1.31112 w_k1_MSFT_s016
                                 + 1.10783 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.940145 w_k1_ADM_s016
                                 + 0.911621 w_k1_AXP_s016
                                 + 1.02337 w_k1_DIS_s016
                                 + 0.990424 w_k1_FDX_s016
                                 + 0.922943 w_k1_IBM_s016
                                 + 0.973175 w_k1_INTC_s016
                                 + 0.92574 w_k1_KO_s016
                                 + 0.898147 w_k1_MAT_s016
                                 + 1.15187 w_k1_MSFT_s016
                                 + 1.05695 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.864529 w_k1_ADM_s016
                                 + 0.861742 w_k1_AXP_s016
                                 + 0.83909 w_k1_DIS_s016
                                 + 0.777137 w_k1_FDX_s016
                                 + 1.07251 w_k1_IBM_s016
                                 + 0.794681 w_k1_INTC_s016
                                 + 0.873938 w_k1_KO_s016
                                 + 0.810136 w_k1_MAT_s016
                                 + 0.972174 w_k1_MSFT_s016
                                 + 0.97542 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.01754 w_k1_ADM_s016
                                 + 1.38128 w_k1_AXP_s016 + 1.1712 w_k1_DIS_s016
                                 + 0.965689 w_k1_FDX_s016
                                 + 1.7877 w_k1_IBM_s016
                                 + 0.913075 w_k1_INTC_s016
                                 + 1.2292 w_k1_KO_s016 + 1.87843 w_k1_MAT_s016
                                 + 0.988261 w_k1_MSFT_s016
                                 + 1.05358 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.899872 w_k1_ADM_s016
                                 + 0.708333 w_k1_AXP_s016
                                 + 0.581018 w_k1_DIS_s016
                                 + 0.736424 w_k1_FDX_s016
                                 + 0.696494 w_k1_IBM_s016
                                 + 0.771554 w_k1_INTC_s016
                                 + 0.65763 w_k1_KO_s016
                                 + 0.452981 w_k1_MAT_s016
                                 + 0.813751 w_k1_MSFT_s016
                                 + 0.886244 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.17349 w_k1_ADM_s016
                                 + 1.54943 w_k1_AXP_s016
                                 + 1.03691 w_k1_DIS_s016
                                 + 1.02435 w_k1_FDX_s016
                                 + 1.05213 w_k1_IBM_s016
                                 + 0.992428 w_k1_INTC_s016
                                 + 0.976364 w_k1_KO_s016
                                 + 1.22338 w_k1_MAT_s016 + 1.128 w_k1_MSFT_s016
                                 + 1.17939 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.9492 w_k1_ADM_s017
                                 + 0.884928 w_k1_AXP_s017
                                 + 0.867097 w_k1_DIS_s017
                                 + 1.02705 w_k1_FDX_s017
                                 + 0.585909 w_k1_IBM_s017
                                 + 1.10254 w_k1_INTC_s017
                                 + 0.841615 w_k1_KO_s017
                                 + 0.531921 w_k1_MAT_s017
                                 + 1.02179 w_k1_MSFT_s017
                                 + 0.942254 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.949811 w_k1_ADM_s017
                                 + 0.896658 w_k1_AXP_s017
                                 + 1.14473 w_k1_DIS_s017
                                 + 1.09075 w_k1_FDX_s017
                                 + 0.914677 w_k1_IBM_s017
                                 + 1.11164 w_k1_INTC_s017
                                 + 1.17349 w_k1_KO_s017 + 1.00232 w_k1_MAT_s017
                                 + 1.20124 w_k1_MSFT_s017
                                 + 1.03005 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.06618 w_k1_ADM_s017
                                 + 0.766508 w_k1_AXP_s017
                                 + 0.746875 w_k1_DIS_s017
                                 + 0.86975 w_k1_FDX_s017
                                 + 0.735734 w_k1_IBM_s017
                                 + 0.899882 w_k1_INTC_s017
                                 + 0.925365 w_k1_KO_s017
                                 + 0.629187 w_k1_MAT_s017
                                 + 0.96832 w_k1_MSFT_s017
                                 + 0.881336 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.232 w_k1_ADM_s017
                                 + 1.31543 w_k1_AXP_s017
                                 + 1.50555 w_k1_DIS_s017
                                 + 1.31101 w_k1_FDX_s017
                                 + 0.931478 w_k1_IBM_s017
                                 + 1.07707 w_k1_INTC_s017
                                 + 1.25194 w_k1_KO_s017 + 1.12077 w_k1_MAT_s017
                                 + 1.36759 w_k1_MSFT_s017
                                 + 1.20322 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.976679 w_k1_ADM_s017
                                 + 1.30975 w_k1_AXP_s017
                                 + 0.993357 w_k1_DIS_s017
                                 + 1.07188 w_k1_FDX_s017
                                 + 0.996898 w_k1_IBM_s017
                                 + 1.21879 w_k1_INTC_s017 + 1.2105 w_k1_KO_s017
                                 + 1.30279 w_k1_MAT_s017
                                 + 1.14404 w_k1_MSFT_s017
                                 + 1.11644 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.866846 w_k1_ADM_s017
                                 + 0.721289 w_k1_AXP_s017
                                 + 0.774686 w_k1_DIS_s017
                                 + 0.815194 w_k1_FDX_s017
                                 + 0.883895 w_k1_IBM_s017
                                 + 0.64828 w_k1_INTC_s017
                                 + 1.01137 w_k1_KO_s017
                                 + 0.818252 w_k1_MAT_s017
                                 + 0.897656 w_k1_MSFT_s017
                                 + 0.889132 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.91668 w_k1_ADM_s017
                                 + 0.756627 w_k1_AXP_s017
                                 + 0.82294 w_k1_DIS_s017
                                 + 0.822808 w_k1_FDX_s017
                                 + 1.16329 w_k1_IBM_s017
                                 + 0.829239 w_k1_INTC_s017
                                 + 1.09577 w_k1_KO_s017 + 1.15611 w_k1_MAT_s017
                                 + 1.04272 w_k1_MSFT_s017
                                 + 1.07016 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.12 w_k1_ADM_s017
                                 + 1.07238 w_k1_AXP_s017
                                 + 1.38958 w_k1_DIS_s017
                                 + 0.993963 w_k1_FDX_s017
                                 + 1.53839 w_k1_IBM_s017
                                 + 1.00712 w_k1_INTC_s017
                                 + 1.69414 w_k1_KO_s017 + 1.44082 w_k1_MAT_s017
                                 + 1.26697 w_k1_MSFT_s017
                                 + 1.20151 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.02428 w_k1_ADM_s100
                                 + 0.839905 w_k1_AXP_s100
                                 + 0.940275 w_k1_DIS_s100
                                 + 0.969894 w_k1_FDX_s100
                                 + 0.97507 w_k1_IBM_s100
                                 + 0.971817 w_k1_INTC_s100
                                 + 0.921188 w_k1_KO_s100
                                 + 0.829683 w_k1_MAT_s100
                                 + 0.956087 w_k1_MSFT_s100
                                 + 0.782263 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.955203 w_k1_ADM_s100
                                 + 0.80091 w_k1_AXP_s100
                                 + 0.674012 w_k1_DIS_s100
                                 + 0.823484 w_k1_FDX_s100
                                 + 0.879952 w_k1_IBM_s100
                                 + 0.772125 w_k1_INTC_s100
                                 + 0.885967 w_k1_KO_s100
                                 + 0.895684 w_k1_MAT_s100
                                 + 0.854318 w_k1_MSFT_s100
                                 + 0.802797 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.65554 w_k1_ADM_s100
                                 + 1.49341 w_k1_AXP_s100
                                 + 1.68794 w_k1_DIS_s100
                                 + 1.47763 w_k1_FDX_s100
                                 + 1.08651 w_k1_IBM_s100
                                 + 1.18619 w_k1_INTC_s100
                                 + 1.48556 w_k1_KO_s100 + 1.4781 w_k1_MAT_s100
                                 + 1.35107 w_k1_MSFT_s100
                                 + 1.24164 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.983791 w_k1_ADM_s100
                                 + 1.25881 w_k1_AXP_s100
                                 + 1.13234 w_k1_DIS_s100
                                 + 1.11162 w_k1_FDX_s100
                                 + 1.14607 w_k1_IBM_s100
                                 + 1.21151 w_k1_INTC_s100
                                 + 1.02216 w_k1_KO_s100 + 1.0926 w_k1_MAT_s100
                                 + 1.00365 w_k1_MSFT_s100
                                 + 0.995004 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.680533 w_k1_ADM_s100
                                 + 0.552445 w_k1_AXP_s100
                                 + 0.527854 w_k1_DIS_s100
                                 + 0.699345 w_k1_FDX_s100
                                 + 0.897642 w_k1_IBM_s100
                                 + 0.653704 w_k1_INTC_s100
                                 + 0.786537 w_k1_KO_s100
                                 + 1.02212 w_k1_MAT_s100
                                 + 0.69437 w_k1_MSFT_s100
                                 + 0.763956 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.853029 w_k1_ADM_s100
                                 + 1.0947 w_k1_AXP_s100
                                 + 0.834307 w_k1_DIS_s100
                                 + 0.946612 w_k1_FDX_s100
                                 + 1.25161 w_k1_IBM_s100
                                 + 0.820148 w_k1_INTC_s100
                                 + 1.11142 w_k1_KO_s100 + 1.85794 w_k1_MAT_s100
                                 + 0.962284 w_k1_MSFT_s100
                                 + 0.82339 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.00135 w_k1_ADM_s100
                                 + 0.711626 w_k1_AXP_s100
                                 + 0.814022 w_k1_DIS_s100
                                 + 0.917285 w_k1_FDX_s100
                                 + 0.675237 w_k1_IBM_s100
                                 + 0.824327 w_k1_INTC_s100
                                 + 0.856209 w_k1_KO_s100
                                 + 0.715124 w_k1_MAT_s100
                                 + 0.847461 w_k1_MSFT_s100
                                 + 0.702148 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.32665 w_k1_ADM_s100
                                 + 0.974966 w_k1_AXP_s100
                                 + 1.13883 w_k1_DIS_s100
                                 + 1.21006 w_k1_FDX_s100
                                 + 0.680053 w_k1_IBM_s100
                                 + 1.09545 w_k1_INTC_s100
                                 + 1.13939 w_k1_KO_s100
                                 + 0.885487 w_k1_MAT_s100
                                 + 1.25596 w_k1_MSFT_s100
                                 + 1.01379 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.25815 w_k1_ADM_s101
                                 + 0.887847 w_k1_AXP_s101
                                 + 1.04678 w_k1_DIS_s101
                                 + 0.960112 w_k1_FDX_s101
                                 + 1.14033 w_k1_IBM_s101
                                 + 0.844711 w_k1_INTC_s101
                                 + 1.13643 w_k1_KO_s101
                                 + 0.906584 w_k1_MAT_s101
                                 + 0.901821 w_k1_MSFT_s101
                                 + 1.06925 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.32916 w_k1_ADM_s101
                                 + 1.58299 w_k1_AXP_s101
                                 + 1.57359 w_k1_DIS_s101
                                 + 1.41194 w_k1_FDX_s101
                                 + 1.18472 w_k1_IBM_s101
                                 + 1.37919 w_k1_INTC_s101 + 1.1227 w_k1_KO_s101
                                 + 1.11982 w_k1_MAT_s101
                                 + 1.10621 w_k1_MSFT_s101
                                 + 1.14046 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.12561 w_k1_ADM_s101
                                 + 0.969581 w_k1_AXP_s101
                                 + 1.11893 w_k1_DIS_s101
                                 + 1.17802 w_k1_FDX_s101
                                 + 0.827883 w_k1_IBM_s101
                                 + 0.977605 w_k1_INTC_s101
                                 + 0.905505 w_k1_KO_s101
                                 + 0.750335 w_k1_MAT_s101
                                 + 1.08695 w_k1_MSFT_s101
                                 + 0.960336 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.968897 w_k1_ADM_s101
                                 + 1.01004 w_k1_AXP_s101
                                 + 0.986309 w_k1_DIS_s101
                                 + 0.930647 w_k1_FDX_s101
                                 + 1.54589 w_k1_IBM_s101
                                 + 0.951926 w_k1_INTC_s101
                                 + 1.16042 w_k1_KO_s101 + 1.44076 w_k1_MAT_s101
                                 + 0.923359 w_k1_MSFT_s101
                                 + 1.2037 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.944475 w_k1_ADM_s101
                                 + 1.07565 w_k1_AXP_s101
                                 + 1.38498 w_k1_DIS_s101
                                 + 1.11217 w_k1_FDX_s101
                                 + 1.36547 w_k1_IBM_s101 + 1.22 w_k1_INTC_s101
                                 + 1.22984 w_k1_KO_s101 + 1.31891 w_k1_MAT_s101
                                 + 1.23723 w_k1_MSFT_s101
                                 + 1.09097 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.868567 w_k1_ADM_s101
                                 + 0.695816 w_k1_AXP_s101
                                 + 0.598863 w_k1_DIS_s101
                                 + 0.777764 w_k1_FDX_s101
                                 + 0.71596 w_k1_IBM_s101
                                 + 0.720129 w_k1_INTC_s101
                                 + 0.690455 w_k1_KO_s101
                                 + 0.535364 w_k1_MAT_s101
                                 + 0.595585 w_k1_MSFT_s101
                                 + 0.828464 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.08603 w_k1_ADM_s101
                                 + 0.895584 w_k1_AXP_s101
                                 + 0.877473 w_k1_DIS_s101
                                 + 0.889966 w_k1_FDX_s101
                                 + 0.902272 w_k1_IBM_s101
                                 + 0.934532 w_k1_INTC_s101
                                 + 0.921063 w_k1_KO_s101
                                 + 0.675836 w_k1_MAT_s101
                                 + 0.833614 w_k1_MSFT_s101
                                 + 0.971429 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.53049 w_k1_ADM_s101
                                 + 1.53828 w_k1_AXP_s101
                                 + 1.37814 w_k1_DIS_s101
                                 + 1.26086 w_k1_FDX_s101
                                 + 0.921783 w_k1_IBM_s101
                                 + 1.39096 w_k1_INTC_s101
                                 + 0.998451 w_k1_KO_s101
                                 + 0.912564 w_k1_MAT_s101
                                 + 1.07867 w_k1_MSFT_s101 + 1.233 w_k1_PG_s101
                                 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.09105 w_k1_ADM_s102
                                 + 1.34487 w_k1_AXP_s102
                                 + 0.803141 w_k1_DIS_s102
                                 + 0.943518 w_k1_FDX_s102
                                 + 1.00236 w_k1_IBM_s102
                                 + 0.995682 w_k1_INTC_s102
                                 + 1.07208 w_k1_KO_s102 + 1.20597 w_k1_MAT_s102
                                 + 0.895319 w_k1_MSFT_s102
                                 + 0.97787 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.902871 w_k1_ADM_s102
                                 + 0.890643 w_k1_AXP_s102
                                 + 0.805502 w_k1_DIS_s102
                                 + 0.975836 w_k1_FDX_s102
                                 + 0.802995 w_k1_IBM_s102
                                 + 0.857517 w_k1_INTC_s102
                                 + 0.887372 w_k1_KO_s102
                                 + 0.759147 w_k1_MAT_s102
                                 + 0.969932 w_k1_MSFT_s102
                                 + 1.02219 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.671071 w_k1_ADM_s102
                                 + 0.754514 w_k1_AXP_s102
                                 + 0.702169 w_k1_DIS_s102
                                 + 0.790687 w_k1_FDX_s102
                                 + 0.896745 w_k1_IBM_s102
                                 + 1.09881 w_k1_INTC_s102
                                 + 0.858657 w_k1_KO_s102
                                 + 0.691449 w_k1_MAT_s102
                                 + 0.862047 w_k1_MSFT_s102
                                 + 0.956435 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.626058 w_k1_ADM_s102
                                 + 0.76659 w_k1_AXP_s102
                                 + 0.562004 w_k1_DIS_s102
                                 + 0.64727 w_k1_FDX_s102
                                 + 0.943406 w_k1_IBM_s102
                                 + 0.889329 w_k1_INTC_s102
                                 + 1.00536 w_k1_KO_s102
                                 + 0.975558 w_k1_MAT_s102
                                 + 0.844173 w_k1_MSFT_s102
                                 + 0.885003 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.02627 w_k1_ADM_s102
                                 + 0.997598 w_k1_AXP_s102
                                 + 1.08349 w_k1_DIS_s102 + 1.2228 w_k1_FDX_s102
                                 + 0.893725 w_k1_IBM_s102
                                 + 1.15109 w_k1_INTC_s102
                                 + 0.96911 w_k1_KO_s102
                                 + 0.777535 w_k1_MAT_s102
                                 + 1.23047 w_k1_MSFT_s102
                                 + 1.08549 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.25609 w_k1_ADM_s102
                                 + 1.4232 w_k1_AXP_s102 + 1.63429 w_k1_DIS_s102
                                 + 1.3921 w_k1_FDX_s102 + 1.21441 w_k1_IBM_s102
                                 + 1.21461 w_k1_INTC_s102 + 1.2349 w_k1_KO_s102
                                 + 0.982134 w_k1_MAT_s102
                                 + 1.35845 w_k1_MSFT_s102
                                 + 1.18024 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.998994 w_k1_ADM_s102
                                 + 1.46528 w_k1_AXP_s102
                                 + 0.97673 w_k1_DIS_s102
                                 + 1.02224 w_k1_FDX_s102
                                 + 1.39712 w_k1_IBM_s102
                                 + 1.02113 w_k1_INTC_s102
                                 + 1.18056 w_k1_KO_s102 + 1.73838 w_k1_MAT_s102
                                 + 0.997185 w_k1_MSFT_s102
                                 + 1.04467 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.0188 w_k1_ADM_s102
                                 + 1.0322 w_k1_AXP_s102 + 1.01113 w_k1_DIS_s102
                                 + 0.998451 w_k1_FDX_s102
                                 + 1.31353 w_k1_IBM_s102
                                 + 0.989266 w_k1_INTC_s102
                                 + 1.21677 w_k1_KO_s102 + 1.17172 w_k1_MAT_s102
                                 + 1.05719 w_k1_MSFT_s102 + 0.9913 w_k1_PG_s102
                                 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.18931 w_k1_ADM_s103
                                 + 1.04523 w_k1_AXP_s103
                                 + 0.808642 w_k1_DIS_s103
                                 + 0.879234 w_k1_FDX_s103
                                 + 1.33638 w_k1_IBM_s103
                                 + 1.22244 w_k1_INTC_s103
                                 + 1.16183 w_k1_KO_s103 + 1.18786 w_k1_MAT_s103
                                 + 1.01572 w_k1_MSFT_s103
                                 + 0.936927 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.01389 w_k1_ADM_s103
                                 + 1.22115 w_k1_AXP_s103
                                 + 1.01914 w_k1_DIS_s103
                                 + 0.944112 w_k1_FDX_s103
                                 + 1.01027 w_k1_IBM_s103
                                 + 0.987503 w_k1_INTC_s103
                                 + 1.00725 w_k1_KO_s103 + 1.20148 w_k1_MAT_s103
                                 + 1.09474 w_k1_MSFT_s103
                                 + 0.998272 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.924536 w_k1_ADM_s103
                                 + 0.789832 w_k1_AXP_s103
                                 + 0.594188 w_k1_DIS_s103
                                 + 0.706367 w_k1_FDX_s103
                                 + 0.72194 w_k1_IBM_s103
                                 + 0.800306 w_k1_INTC_s103
                                 + 0.746613 w_k1_KO_s103
                                 + 0.635183 w_k1_MAT_s103
                                 + 0.788187 w_k1_MSFT_s103
                                 + 0.693616 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.26999 w_k1_ADM_s103
                                 + 1.21245 w_k1_AXP_s103 + 1.3036 w_k1_DIS_s103
                                 + 0.943928 w_k1_FDX_s103
                                 + 1.70109 w_k1_IBM_s103
                                 + 1.0019 w_k1_INTC_s103 + 1.38984 w_k1_KO_s103
                                 + 2.04992 w_k1_MAT_s103
                                 + 0.977611 w_k1_MSFT_s103
                                 + 1.14081 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.979701 w_k1_ADM_s103
                                 + 0.926231 w_k1_AXP_s103
                                 + 0.634919 w_k1_DIS_s103
                                 + 0.862137 w_k1_FDX_s103
                                 + 0.742105 w_k1_IBM_s103
                                 + 0.927666 w_k1_INTC_s103
                                 + 0.714523 w_k1_KO_s103
                                 + 0.708479 w_k1_MAT_s103
                                 + 0.819236 w_k1_MSFT_s103
                                 + 0.827716 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.33639 w_k1_ADM_s103
                                 + 1.38633 w_k1_AXP_s103
                                 + 1.35139 w_k1_DIS_s103
                                 + 1.23716 w_k1_FDX_s103
                                 + 0.800961 w_k1_IBM_s103
                                 + 1.24279 w_k1_INTC_s103
                                 + 1.11432 w_k1_KO_s103
                                 + 0.833342 w_k1_MAT_s103
                                 + 1.30505 w_k1_MSFT_s103
                                 + 1.04281 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.10914 w_k1_ADM_s103
                                 + 1.11851 w_k1_AXP_s103
                                 + 0.977362 w_k1_DIS_s103
                                 + 1.05715 w_k1_FDX_s103
                                 + 0.809144 w_k1_IBM_s103
                                 + 1.0697 w_k1_INTC_s103
                                 + 0.856492 w_k1_KO_s103
                                 + 0.656671 w_k1_MAT_s103
                                 + 1.01543 w_k1_MSFT_s103
                                 + 0.942113 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.18553 w_k1_ADM_s103
                                 + 1.16913 w_k1_AXP_s103
                                 + 1.43352 w_k1_DIS_s103
                                 + 1.23952 w_k1_FDX_s103
                                 + 1.15055 w_k1_IBM_s103
                                 + 1.02855 w_k1_INTC_s103 + 1.0915 w_k1_KO_s103
                                 + 1.08634 w_k1_MAT_s103
                                 + 1.13833 w_k1_MSFT_s103
                                 + 0.954614 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.09325 w_k1_ADM_s104
                                 + 1.38044 w_k1_AXP_s104
                                 + 1.17863 w_k1_DIS_s104
                                 + 0.928539 w_k1_FDX_s104
                                 + 1.80284 w_k1_IBM_s104
                                 + 0.887925 w_k1_INTC_s104
                                 + 1.59666 w_k1_KO_s104 + 2.25138 w_k1_MAT_s104
                                 + 0.909402 w_k1_MSFT_s104
                                 + 1.06524 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.01624 w_k1_ADM_s104
                                 + 1.15178 w_k1_AXP_s104
                                 + 1.18322 w_k1_DIS_s104
                                 + 1.29888 w_k1_FDX_s104
                                 + 0.832852 w_k1_IBM_s104
                                 + 1.09189 w_k1_INTC_s104
                                 + 1.06747 w_k1_KO_s104 + 1.06274 w_k1_MAT_s104
                                 + 1.12803 w_k1_MSFT_s104
                                 + 1.11107 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.924991 w_k1_ADM_s104
                                 + 1.09239 w_k1_AXP_s104
                                 + 1.34788 w_k1_DIS_s104
                                 + 1.08907 w_k1_FDX_s104 + 1.1385 w_k1_IBM_s104
                                 + 1.01333 w_k1_INTC_s104
                                 + 1.28501 w_k1_KO_s104 + 1.48731 w_k1_MAT_s104
                                 + 1.15771 w_k1_MSFT_s104
                                 + 1.01057 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.855479 w_k1_ADM_s104
                                 + 1.20293 w_k1_AXP_s104
                                 + 0.994923 w_k1_DIS_s104
                                 + 1.01053 w_k1_FDX_s104
                                 + 1.24028 w_k1_IBM_s104
                                 + 1.00506 w_k1_INTC_s104
                                 + 1.24938 w_k1_KO_s104 + 1.39964 w_k1_MAT_s104
                                 + 1.00446 w_k1_MSFT_s104
                                 + 0.891965 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.90509 w_k1_ADM_s104
                                 + 0.917459 w_k1_AXP_s104
                                 + 0.838596 w_k1_DIS_s104
                                 + 0.955125 w_k1_FDX_s104
                                 + 0.841733 w_k1_IBM_s104
                                 + 1.01341 w_k1_INTC_s104
                                 + 0.912198 w_k1_KO_s104
                                 + 0.73537 w_k1_MAT_s104
                                 + 0.899928 w_k1_MSFT_s104
                                 + 0.83286 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.59262 w_k1_ADM_s104
                                 + 1.58864 w_k1_AXP_s104
                                 + 1.68675 w_k1_DIS_s104 + 1.4926 w_k1_FDX_s104
                                 + 1.21702 w_k1_IBM_s104
                                 + 1.17781 w_k1_INTC_s104
                                 + 1.24999 w_k1_KO_s104 + 1.27296 w_k1_MAT_s104
                                 + 1.11437 w_k1_MSFT_s104
                                 + 1.18051 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.887661 w_k1_ADM_s104
                                 + 0.899752 w_k1_AXP_s104
                                 + 0.943742 w_k1_DIS_s104
                                 + 0.91311 w_k1_FDX_s104
                                 + 0.904283 w_k1_IBM_s104
                                 + 0.88076 w_k1_INTC_s104
                                 + 0.881363 w_k1_KO_s104
                                 + 0.589623 w_k1_MAT_s104
                                 + 0.876926 w_k1_MSFT_s104
                                 + 0.930701 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.69208 w_k1_ADM_s104
                                 + 0.592664 w_k1_AXP_s104
                                 + 0.550249 w_k1_DIS_s104
                                 + 0.638577 w_k1_FDX_s104
                                 + 0.904225 w_k1_IBM_s104
                                 + 0.723513 w_k1_INTC_s104
                                 + 0.82387 w_k1_KO_s104
                                 + 0.655975 w_k1_MAT_s104
                                 + 0.670031 w_k1_MSFT_s104
                                 + 0.787257 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.778496 w_k1_ADM_s105
                                 + 1.06895 w_k1_AXP_s105
                                 + 0.719093 w_k1_DIS_s105
                                 + 0.89271 w_k1_FDX_s105
                                 + 0.976975 w_k1_IBM_s105
                                 + 0.9485 w_k1_INTC_s105
                                 + 0.895145 w_k1_KO_s105
                                 + 1.39423 w_k1_MAT_s105
                                 + 0.94516 w_k1_MSFT_s105
                                 + 0.83039 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.87764 w_k1_ADM_s105
                                 + 0.824401 w_k1_AXP_s105
                                 + 1.07072 w_k1_DIS_s105
                                 + 1.10475 w_k1_FDX_s105
                                 + 0.788028 w_k1_IBM_s105
                                 + 1.10829 w_k1_INTC_s105
                                 + 0.867816 w_k1_KO_s105
                                 + 0.778745 w_k1_MAT_s105
                                 + 1.10729 w_k1_MSFT_s105
                                 + 0.929261 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.923092 w_k1_ADM_s105
                                 + 1.00585 w_k1_AXP_s105
                                 + 1.12025 w_k1_DIS_s105
                                 + 1.02812 w_k1_FDX_s105
                                 + 1.57108 w_k1_IBM_s105
                                 + 1.39361 w_k1_INTC_s105
                                 + 1.19462 w_k1_KO_s105 + 1.24409 w_k1_MAT_s105
                                 + 1.24354 w_k1_MSFT_s105
                                 + 1.09428 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.875331 w_k1_ADM_s105
                                 + 0.772057 w_k1_AXP_s105
                                 + 0.666565 w_k1_DIS_s105
                                 + 0.726591 w_k1_FDX_s105
                                 + 0.985231 w_k1_IBM_s105
                                 + 0.900337 w_k1_INTC_s105
                                 + 0.964033 w_k1_KO_s105
                                 + 0.827185 w_k1_MAT_s105
                                 + 0.839954 w_k1_MSFT_s105
                                 + 0.927024 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.912812 w_k1_ADM_s105
                                 + 1.10671 w_k1_AXP_s105
                                 + 0.993964 w_k1_DIS_s105
                                 + 1.0275 w_k1_FDX_s105 + 1.24039 w_k1_IBM_s105
                                 + 1.28133 w_k1_INTC_s105
                                 + 0.998822 w_k1_KO_s105
                                 + 1.46873 w_k1_MAT_s105 + 1.149 w_k1_MSFT_s105
                                 + 1.06244 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.16873 w_k1_ADM_s105
                                 + 1.55729 w_k1_AXP_s105
                                 + 1.33019 w_k1_DIS_s105
                                 + 1.24827 w_k1_FDX_s105
                                 + 1.15254 w_k1_IBM_s105
                                 + 1.22766 w_k1_INTC_s105
                                 + 1.03545 w_k1_KO_s105 + 1.17886 w_k1_MAT_s105
                                 + 1.26834 w_k1_MSFT_s105
                                 + 1.08793 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.930681 w_k1_ADM_s105
                                 + 0.720951 w_k1_AXP_s105
                                 + 0.776125 w_k1_DIS_s105
                                 + 0.898064 w_k1_FDX_s105
                                 + 0.605422 w_k1_IBM_s105
                                 + 1.01226 w_k1_INTC_s105
                                 + 0.783093 w_k1_KO_s105
                                 + 0.700515 w_k1_MAT_s105
                                 + 1.03355 w_k1_MSFT_s105
                                 + 0.925338 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.17494 w_k1_ADM_s105
                                 + 1.13044 w_k1_AXP_s105
                                 + 1.17761 w_k1_DIS_s105
                                 + 1.32336 w_k1_FDX_s105
                                 + 0.917459 w_k1_IBM_s105
                                 + 1.25476 w_k1_INTC_s105
                                 + 1.01975 w_k1_KO_s105
                                 + 0.865686 w_k1_MAT_s105
                                 + 1.42599 w_k1_MSFT_s105
                                 + 1.05828 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.998901 w_k1_ADM_s106
                                 + 0.612418 w_k1_AXP_s106
                                 + 0.669786 w_k1_DIS_s106
                                 + 0.790838 w_k1_FDX_s106
                                 + 0.919429 w_k1_IBM_s106
                                 + 0.786812 w_k1_INTC_s106
                                 + 0.951042 w_k1_KO_s106
                                 + 0.713107 w_k1_MAT_s106
                                 + 0.818722 w_k1_MSFT_s106
                                 + 0.888619 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.01489 w_k1_ADM_s106
                                 + 1.40983 w_k1_AXP_s106
                                 + 1.29362 w_k1_DIS_s106
                                 + 1.19055 w_k1_FDX_s106
                                 + 1.13403 w_k1_IBM_s106
                                 + 1.18178 w_k1_INTC_s106
                                 + 1.25402 w_k1_KO_s106 + 1.04047 w_k1_MAT_s106
                                 + 1.0882 w_k1_MSFT_s106 + 1.26512 w_k1_PG_s106
                                 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.15119 w_k1_ADM_s106
                                 + 0.919328 w_k1_AXP_s106
                                 + 0.78697 w_k1_DIS_s106
                                 + 0.993731 w_k1_FDX_s106
                                 + 0.952634 w_k1_IBM_s106
                                 + 0.961716 w_k1_INTC_s106
                                 + 0.898662 w_k1_KO_s106
                                 + 0.567715 w_k1_MAT_s106
                                 + 0.900294 w_k1_MSFT_s106
                                 + 0.863928 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.2036 w_k1_ADM_s106
                                 + 1.33223 w_k1_AXP_s106
                                 + 1.41013 w_k1_DIS_s106 + 1.1673 w_k1_FDX_s106
                                 + 1.67784 w_k1_IBM_s106
                                 + 1.4908 w_k1_INTC_s106 + 1.58119 w_k1_KO_s106
                                 + 0.809235 w_k1_MAT_s106
                                 + 1.20367 w_k1_MSFT_s106
                                 + 1.19354 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.719284 w_k1_ADM_s106
                                 + 0.9235 w_k1_AXP_s106 + 0.89968 w_k1_DIS_s106
                                 + 0.82999 w_k1_FDX_s106
                                 + 1.50006 w_k1_IBM_s106
                                 + 0.896734 w_k1_INTC_s106
                                 + 1.07476 w_k1_KO_s106 + 1.10142 w_k1_MAT_s106
                                 + 0.877078 w_k1_MSFT_s106
                                 + 0.967471 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.38003 w_k1_ADM_s106
                                 + 1.16994 w_k1_AXP_s106
                                 + 1.00634 w_k1_DIS_s106
                                 + 1.13544 w_k1_FDX_s106
                                 + 0.866139 w_k1_IBM_s106
                                 + 1.05259 w_k1_INTC_s106
                                 + 0.966925 w_k1_KO_s106
                                 + 0.906528 w_k1_MAT_s106
                                 + 1.17277 w_k1_MSFT_s106
                                 + 1.03699 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.913906 w_k1_ADM_s106
                                 + 0.988574 w_k1_AXP_s106
                                 + 0.748492 w_k1_DIS_s106
                                 + 0.818711 w_k1_FDX_s106
                                 + 1.1589 w_k1_IBM_s106
                                 + 0.953151 w_k1_INTC_s106
                                 + 0.912351 w_k1_KO_s106
                                 + 0.99724 w_k1_MAT_s106
                                 + 0.786453 w_k1_MSFT_s106
                                 + 0.924908 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.10701 w_k1_ADM_s106
                                 + 0.692419 w_k1_AXP_s106
                                 + 0.64266 w_k1_DIS_s106
                                 + 0.862864 w_k1_FDX_s106
                                 + 0.673199 w_k1_IBM_s106
                                 + 0.968362 w_k1_INTC_s106
                                 + 0.79394 w_k1_KO_s106
                                 + 0.581501 w_k1_MAT_s106
                                 + 1.00391 w_k1_MSFT_s106
                                 + 0.931968 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.809076 w_k1_ADM_s107
                                 + 0.533731 w_k1_AXP_s107
                                 + 0.626707 w_k1_DIS_s107
                                 + 0.682181 w_k1_FDX_s107
                                 + 0.800825 w_k1_IBM_s107
                                 + 0.824404 w_k1_INTC_s107
                                 + 0.857814 w_k1_KO_s107
                                 + 0.659243 w_k1_MAT_s107
                                 + 0.838618 w_k1_MSFT_s107
                                 + 0.76572 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.835896 w_k1_ADM_s107
                                 + 0.868072 w_k1_AXP_s107
                                 + 1.06686 w_k1_DIS_s107
                                 + 0.848425 w_k1_FDX_s107
                                 + 1.73368 w_k1_IBM_s107
                                 + 0.988087 w_k1_INTC_s107
                                 + 1.49902 w_k1_KO_s107 + 2.36962 w_k1_MAT_s107
                                 + 1.09229 w_k1_MSFT_s107
                                 + 1.12672 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.5773 w_k1_ADM_s107
                                 + 1.49757 w_k1_AXP_s107
                                 + 1.30479 w_k1_DIS_s107
                                 + 1.18221 w_k1_FDX_s107
                                 + 1.47506 w_k1_IBM_s107
                                 + 1.21694 w_k1_INTC_s107
                                 + 1.41742 w_k1_KO_s107 + 1.90554 w_k1_MAT_s107
                                 + 1.01918 w_k1_MSFT_s107
                                 + 1.09922 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.92959 w_k1_ADM_s107
                                 + 0.965601 w_k1_AXP_s107
                                 + 1.0034 w_k1_DIS_s107
                                 + 0.849136 w_k1_FDX_s107
                                 + 1.10476 w_k1_IBM_s107
                                 + 0.884081 w_k1_INTC_s107
                                 + 1.18913 w_k1_KO_s107 + 1.49573 w_k1_MAT_s107
                                 + 0.991962 w_k1_MSFT_s107
                                 + 1.00369 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.843689 w_k1_ADM_s107
                                 + 0.714066 w_k1_AXP_s107
                                 + 0.882719 w_k1_DIS_s107
                                 + 0.90155 w_k1_FDX_s107
                                 + 0.592034 w_k1_IBM_s107
                                 + 0.796116 w_k1_INTC_s107
                                 + 0.878273 w_k1_KO_s107
                                 + 0.622289 w_k1_MAT_s107
                                 + 0.895568 w_k1_MSFT_s107
                                 + 0.815091 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.828828 w_k1_ADM_s107
                                 + 0.701169 w_k1_AXP_s107
                                 + 0.697216 w_k1_DIS_s107
                                 + 0.646789 w_k1_FDX_s107
                                 + 1.2699 w_k1_IBM_s107
                                 + 0.988841 w_k1_INTC_s107
                                 + 1.32121 w_k1_KO_s107 + 1.67213 w_k1_MAT_s107
                                 + 0.850654 w_k1_MSFT_s107
                                 + 0.928918 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.03149 w_k1_ADM_s107
                                 + 0.940023 w_k1_AXP_s107
                                 + 1.09879 w_k1_DIS_s107
                                 + 0.982381 w_k1_FDX_s107
                                 + 0.812355 w_k1_IBM_s107
                                 + 1.06863 w_k1_INTC_s107
                                 + 1.12275 w_k1_KO_s107
                                 + 0.991361 w_k1_MAT_s107
                                 + 1.07237 w_k1_MSFT_s107
                                 + 0.983174 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.28986 w_k1_ADM_s107
                                 + 1.55858 w_k1_AXP_s107
                                 + 1.53895 w_k1_DIS_s107
                                 + 1.41683 w_k1_FDX_s107
                                 + 0.836425 w_k1_IBM_s107
                                 + 1.33976 w_k1_INTC_s107
                                 + 1.10516 w_k1_KO_s107 + 0.96513 w_k1_MAT_s107
                                 + 1.13975 w_k1_MSFT_s107
                                 + 1.09495 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.09635 w_k1_ADM_s110
                                 + 0.976615 w_k1_AXP_s110
                                 + 0.918828 w_k1_DIS_s110
                                 + 0.93115 w_k1_FDX_s110
                                 + 0.932267 w_k1_IBM_s110
                                 + 1.13059 w_k1_INTC_s110
                                 + 0.912225 w_k1_KO_s110
                                 + 0.780076 w_k1_MAT_s110
                                 + 0.923275 w_k1_MSFT_s110
                                 + 0.918282 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.16584 w_k1_ADM_s110
                                 + 1.06372 w_k1_AXP_s110
                                 + 1.11171 w_k1_DIS_s110
                                 + 0.998959 w_k1_FDX_s110
                                 + 1.19974 w_k1_IBM_s110
                                 + 1.17168 w_k1_INTC_s110
                                 + 1.25212 w_k1_KO_s110 + 1.08001 w_k1_MAT_s110
                                 + 1.00471 w_k1_MSFT_s110
                                 + 1.07041 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.16692 w_k1_ADM_s110
                                 + 1.78751 w_k1_AXP_s110
                                 + 1.42705 w_k1_DIS_s110
                                 + 1.28161 w_k1_FDX_s110
                                 + 1.68923 w_k1_IBM_s110
                                 + 1.45352 w_k1_INTC_s110
                                 + 1.37105 w_k1_KO_s110 + 2.34143 w_k1_MAT_s110
                                 + 1.28549 w_k1_MSFT_s110
                                 + 1.24373 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.31874 w_k1_ADM_s110
                                 + 1.51913 w_k1_AXP_s110
                                 + 1.47396 w_k1_DIS_s110
                                 + 1.29095 w_k1_FDX_s110
                                 + 1.16231 w_k1_IBM_s110
                                 + 0.928928 w_k1_INTC_s110
                                 + 1.20484 w_k1_KO_s110 + 1.3448 w_k1_MAT_s110
                                 + 1.07204 w_k1_MSFT_s110 + 1.1678 w_k1_PG_s110
                                 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.890738 w_k1_ADM_s110
                                 + 0.742995 w_k1_AXP_s110
                                 + 0.714442 w_k1_DIS_s110
                                 + 0.665695 w_k1_FDX_s110
                                 + 1.18378 w_k1_IBM_s110
                                 + 0.871521 w_k1_INTC_s110
                                 + 0.906119 w_k1_KO_s110
                                 + 0.899668 w_k1_MAT_s110
                                 + 0.706968 w_k1_MSFT_s110
                                 + 0.869412 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.27063 w_k1_ADM_s110
                                 + 1.39094 w_k1_AXP_s110
                                 + 1.73818 w_k1_DIS_s110
                                 + 1.33357 w_k1_FDX_s110
                                 + 0.996692 w_k1_IBM_s110
                                 + 1.42687 w_k1_INTC_s110
                                 + 1.41386 w_k1_KO_s110 + 1.07053 w_k1_MAT_s110
                                 + 1.38321 w_k1_MSFT_s110
                                 + 1.20616 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.995421 w_k1_ADM_s110
                                 + 0.955072 w_k1_AXP_s110
                                 + 0.99593 w_k1_DIS_s110
                                 + 0.865972 w_k1_FDX_s110
                                 + 1.75888 w_k1_IBM_s110
                                 + 0.877314 w_k1_INTC_s110
                                 + 1.39486 w_k1_KO_s110 + 1.48984 w_k1_MAT_s110
                                 + 0.918692 w_k1_MSFT_s110
                                 + 0.984647 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.864729 w_k1_ADM_s110
                                 + 0.903944 w_k1_AXP_s110
                                 + 0.763688 w_k1_DIS_s110
                                 + 0.922448 w_k1_FDX_s110
                                 + 0.654801 w_k1_IBM_s110
                                 + 0.85862 w_k1_INTC_s110
                                 + 0.747311 w_k1_KO_s110
                                 + 0.702631 w_k1_MAT_s110
                                 + 0.757894 w_k1_MSFT_s110
                                 + 0.848252 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.02921 w_k1_ADM_s111
                                 + 1.14978 w_k1_AXP_s111
                                 + 0.994732 w_k1_DIS_s111
                                 + 1.06168 w_k1_FDX_s111
                                 + 1.04302 w_k1_IBM_s111
                                 + 0.948788 w_k1_INTC_s111
                                 + 0.859791 w_k1_KO_s111
                                 + 1.19529 w_k1_MAT_s111
                                 + 0.975966 w_k1_MSFT_s111
                                 + 0.953078 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.935795 w_k1_ADM_s111
                                 + 0.802283 w_k1_AXP_s111
                                 + 0.980009 w_k1_DIS_s111
                                 + 0.983437 w_k1_FDX_s111
                                 + 0.787153 w_k1_IBM_s111
                                 + 0.975892 w_k1_INTC_s111
                                 + 0.915877 w_k1_KO_s111
                                 + 0.773588 w_k1_MAT_s111
                                 + 1.00812 w_k1_MSFT_s111
                                 + 0.853945 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.792987 w_k1_ADM_s111
                                 + 1.20708 w_k1_AXP_s111
                                 + 0.944307 w_k1_DIS_s111
                                 + 0.911361 w_k1_FDX_s111
                                 + 1.33963 w_k1_IBM_s111
                                 + 0.997149 w_k1_INTC_s111
                                 + 0.973456 w_k1_KO_s111
                                 + 1.19734 w_k1_MAT_s111
                                 + 0.949954 w_k1_MSFT_s111
                                 + 0.975041 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.831108 w_k1_ADM_s111
                                 + 0.847202 w_k1_AXP_s111
                                 + 0.946518 w_k1_DIS_s111
                                 + 0.889675 w_k1_FDX_s111
                                 + 1.0897 w_k1_IBM_s111
                                 + 0.833574 w_k1_INTC_s111
                                 + 1.10517 w_k1_KO_s111 + 1.26648 w_k1_MAT_s111
                                 + 1.04886 w_k1_MSFT_s111
                                 + 0.95449 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.03184 w_k1_ADM_s111
                                 + 1.26317 w_k1_AXP_s111
                                 + 1.33118 w_k1_DIS_s111
                                 + 1.24203 w_k1_FDX_s111
                                 + 1.14034 w_k1_IBM_s111
                                 + 1.13766 w_k1_INTC_s111
                                 + 1.13018 w_k1_KO_s111 + 1.27299 w_k1_MAT_s111
                                 + 1.16969 w_k1_MSFT_s111
                                 + 1.03155 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.14412 w_k1_ADM_s111
                                 + 1.65485 w_k1_AXP_s111
                                 + 1.57933 w_k1_DIS_s111
                                 + 1.23683 w_k1_FDX_s111
                                 + 1.78532 w_k1_IBM_s111
                                 + 1.06572 w_k1_INTC_s111
                                 + 1.31105 w_k1_KO_s111 + 2.36043 w_k1_MAT_s111
                                 + 1.12541 w_k1_MSFT_s111
                                 + 1.17113 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.18494 w_k1_ADM_s111
                                 + 1.79304 w_k1_AXP_s111
                                 + 1.67063 w_k1_DIS_s111
                                 + 1.56681 w_k1_FDX_s111 + 1.1666 w_k1_IBM_s111
                                 + 1.20989 w_k1_INTC_s111
                                 + 1.26179 w_k1_KO_s111 + 1.31902 w_k1_MAT_s111
                                 + 1.31818 w_k1_MSFT_s111
                                 + 1.25469 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.776273 w_k1_ADM_s111
                                 + 0.829945 w_k1_AXP_s111
                                 + 0.757919 w_k1_DIS_s111
                                 + 0.886117 w_k1_FDX_s111
                                 + 0.816465 w_k1_IBM_s111
                                 + 0.798383 w_k1_INTC_s111
                                 + 0.752298 w_k1_KO_s111
                                 + 0.607304 w_k1_MAT_s111
                                 + 0.822769 w_k1_MSFT_s111
                                 + 0.820119 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.03586 w_k1_ADM_s112
                                 + 0.954164 w_k1_AXP_s112
                                 + 1.02732 w_k1_DIS_s112
                                 + 1.00574 w_k1_FDX_s112
                                 + 0.843519 w_k1_IBM_s112
                                 + 1.12315 w_k1_INTC_s112
                                 + 1.02502 w_k1_KO_s112
                                 + 0.668712 w_k1_MAT_s112
                                 + 0.957825 w_k1_MSFT_s112
                                 + 0.966314 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.832737 w_k1_ADM_s112
                                 + 0.981079 w_k1_AXP_s112
                                 + 1.02514 w_k1_DIS_s112
                                 + 0.906138 w_k1_FDX_s112
                                 + 1.12788 w_k1_IBM_s112
                                 + 1.02693 w_k1_INTC_s112 + 1.0019 w_k1_KO_s112
                                 + 1.0859 w_k1_MAT_s112
                                 + 0.980394 w_k1_MSFT_s112
                                 + 1.00302 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.1255 w_k1_ADM_s112
                                 + 1.53724 w_k1_AXP_s112 + 1.2788 w_k1_DIS_s112
                                 + 1.14104 w_k1_FDX_s112
                                 + 1.50819 w_k1_IBM_s112
                                 + 1.05119 w_k1_INTC_s112
                                 + 1.27692 w_k1_KO_s112 + 1.91819 w_k1_MAT_s112
                                 + 0.932274 w_k1_MSFT_s112
                                 + 1.17401 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.967293 w_k1_ADM_s112
                                 + 0.917717 w_k1_AXP_s112
                                 + 0.808216 w_k1_DIS_s112
                                 + 0.946793 w_k1_FDX_s112
                                 + 0.633614 w_k1_IBM_s112
                                 + 0.974474 w_k1_INTC_s112
                                 + 0.889001 w_k1_KO_s112
                                 + 0.499278 w_k1_MAT_s112
                                 + 0.806566 w_k1_MSFT_s112
                                 + 0.814865 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.27909 w_k1_ADM_s112
                                 + 1.32396 w_k1_AXP_s112
                                 + 1.35352 w_k1_DIS_s112
                                 + 1.28481 w_k1_FDX_s112
                                 + 0.774787 w_k1_IBM_s112
                                 + 1.07177 w_k1_INTC_s112
                                 + 1.10444 w_k1_KO_s112
                                 + 0.842792 w_k1_MAT_s112
                                 + 1.04291 w_k1_MSFT_s112
                                 + 1.02126 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.20559 w_k1_ADM_s112
                                 + 1.32844 w_k1_AXP_s112
                                 + 1.22492 w_k1_DIS_s112
                                 + 1.24676 w_k1_FDX_s112
                                 + 1.07013 w_k1_IBM_s112
                                 + 1.21837 w_k1_INTC_s112
                                 + 1.19349 w_k1_KO_s112 + 1.29945 w_k1_MAT_s112
                                 + 1.13437 w_k1_MSFT_s112
                                 + 1.14245 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.812034 w_k1_ADM_s112
                                 + 0.839374 w_k1_AXP_s112
                                 + 0.639465 w_k1_DIS_s112
                                 + 0.800794 w_k1_FDX_s112
                                 + 0.886012 w_k1_IBM_s112
                                 + 0.884666 w_k1_INTC_s112
                                 + 0.857838 w_k1_KO_s112
                                 + 0.864538 w_k1_MAT_s112
                                 + 0.814439 w_k1_MSFT_s112
                                 + 0.875522 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.835433 w_k1_ADM_s112
                                 + 1.09476 w_k1_AXP_s112
                                 + 0.930458 w_k1_DIS_s112
                                 + 1.00603 w_k1_FDX_s112
                                 + 0.81779 w_k1_IBM_s112
                                 + 0.915407 w_k1_INTC_s112
                                 + 1.00109 w_k1_KO_s112 + 1.09832 w_k1_MAT_s112
                                 + 0.906277 w_k1_MSFT_s112
                                 + 0.931736 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.02182 w_k1_ADM_s113
                                 + 1.14808 w_k1_AXP_s113
                                 + 1.06375 w_k1_DIS_s113
                                 + 0.957499 w_k1_FDX_s113
                                 + 1.56848 w_k1_IBM_s113
                                 + 0.82199 w_k1_INTC_s113 + 1.2431 w_k1_KO_s113
                                 + 1.78349 w_k1_MAT_s113
                                 + 0.999723 w_k1_MSFT_s113
                                 + 1.01232 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.814972 w_k1_ADM_s113
                                 + 0.905714 w_k1_AXP_s113
                                 + 1.06471 w_k1_DIS_s113
                                 + 0.950178 w_k1_FDX_s113
                                 + 0.847014 w_k1_IBM_s113
                                 + 1.05706 w_k1_INTC_s113
                                 + 1.10232 w_k1_KO_s113
                                 + 0.897057 w_k1_MAT_s113
                                 + 1.15524 w_k1_MSFT_s113
                                 + 0.962946 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.08656 w_k1_ADM_s113
                                 + 1.46449 w_k1_AXP_s113
                                 + 1.78039 w_k1_DIS_s113
                                 + 1.37514 w_k1_FDX_s113
                                 + 1.42245 w_k1_IBM_s113
                                 + 1.30195 w_k1_INTC_s113
                                 + 1.57695 w_k1_KO_s113 + 2.23027 w_k1_MAT_s113
                                 + 1.4016 w_k1_MSFT_s113 + 1.23608 w_k1_PG_s113
                                 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.801867 w_k1_ADM_s113
                                 + 0.886827 w_k1_AXP_s113
                                 + 0.671425 w_k1_DIS_s113
                                 + 0.862257 w_k1_FDX_s113
                                 + 0.733291 w_k1_IBM_s113
                                 + 0.802615 w_k1_INTC_s113
                                 + 0.851882 w_k1_KO_s113
                                 + 0.802649 w_k1_MAT_s113
                                 + 0.892046 w_k1_MSFT_s113
                                 + 0.838993 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.654368 w_k1_ADM_s113
                                 + 0.618661 w_k1_AXP_s113
                                 + 0.511992 w_k1_DIS_s113
                                 + 0.653337 w_k1_FDX_s113
                                 + 0.757997 w_k1_IBM_s113
                                 + 0.748075 w_k1_INTC_s113
                                 + 0.621999 w_k1_KO_s113
                                 + 0.483491 w_k1_MAT_s113
                                 + 0.689251 w_k1_MSFT_s113
                                 + 0.765543 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.08646 w_k1_ADM_s113
                                 + 1.55173 w_k1_AXP_s113
                                 + 1.07157 w_k1_DIS_s113
                                 + 1.23935 w_k1_FDX_s113
                                 + 0.887399 w_k1_IBM_s113
                                 + 1.098 w_k1_INTC_s113 + 1.03013 w_k1_KO_s113
                                 + 1.4808 w_k1_MAT_s113
                                 + 1.04184 w_k1_MSFT_s113
                                 + 1.00251 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.06585 w_k1_ADM_s113
                                 + 1.01904 w_k1_AXP_s113
                                 + 1.28116 w_k1_DIS_s113 + 1.1781 w_k1_FDX_s113
                                 + 0.972787 w_k1_IBM_s113
                                 + 1.19144 w_k1_INTC_s113
                                 + 1.14671 w_k1_KO_s113 + 1.37429 w_k1_MAT_s113
                                 + 1.19828 w_k1_MSFT_s113 + 1.0398 w_k1_PG_s113
                                 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.847629 w_k1_ADM_s113
                                 + 0.883837 w_k1_AXP_s113
                                 + 0.884624 w_k1_DIS_s113
                                 + 0.887347 w_k1_FDX_s113
                                 + 1.10655 w_k1_IBM_s113
                                 + 0.942969 w_k1_INTC_s113
                                 + 0.983635 w_k1_KO_s113 + 1.134 w_k1_MAT_s113
                                 + 0.92201 w_k1_MSFT_s113
                                 + 0.88961 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.30998 w_k1_ADM_s114
                                 + 1.08 w_k1_AXP_s114 + 1.08729 w_k1_DIS_s114
                                 + 0.97929 w_k1_FDX_s114 + 1.3393 w_k1_IBM_s114
                                 + 1.07682 w_k1_INTC_s114
                                 + 1.18443 w_k1_KO_s114 + 1.374 w_k1_MAT_s114
                                 + 0.9817 w_k1_MSFT_s114 + 1.06048 w_k1_PG_s114
                                 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.08545 w_k1_ADM_s114
                                 + 1.32008 w_k1_AXP_s114
                                 + 1.54708 w_k1_DIS_s114
                                 + 1.32659 w_k1_FDX_s114
                                 + 1.00717 w_k1_IBM_s114
                                 + 1.19936 w_k1_INTC_s114
                                 + 1.16035 w_k1_KO_s114 + 1.06892 w_k1_MAT_s114
                                 + 1.22912 w_k1_MSFT_s114
                                 + 1.16681 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.28415 w_k1_ADM_s114
                                 + 1.51755 w_k1_AXP_s114
                                 + 1.56062 w_k1_DIS_s114
                                 + 1.26173 w_k1_FDX_s114
                                 + 1.58291 w_k1_IBM_s114
                                 + 1.28137 w_k1_INTC_s114
                                 + 1.60953 w_k1_KO_s114 + 2.11992 w_k1_MAT_s114
                                 + 1.32521 w_k1_MSFT_s114 + 1.0951 w_k1_PG_s114
                                 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.972896 w_k1_ADM_s114
                                 + 1.05403 w_k1_AXP_s114
                                 + 0.844863 w_k1_DIS_s114
                                 + 0.9311 w_k1_FDX_s114
                                 + 0.963577 w_k1_IBM_s114
                                 + 0.989907 w_k1_INTC_s114
                                 + 0.953634 w_k1_KO_s114
                                 + 1.19509 w_k1_MAT_s114
                                 + 1.10475 w_k1_MSFT_s114
                                 + 0.984874 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.809561 w_k1_ADM_s114
                                 + 0.745694 w_k1_AXP_s114
                                 + 0.922937 w_k1_DIS_s114
                                 + 0.83811 w_k1_FDX_s114
                                 + 1.13509 w_k1_IBM_s114
                                 + 1.06641 w_k1_INTC_s114
                                 + 1.06718 w_k1_KO_s114 + 1.15548 w_k1_MAT_s114
                                 + 1.04987 w_k1_MSFT_s114
                                 + 0.926488 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.953929 w_k1_ADM_s114
                                 + 0.919835 w_k1_AXP_s114
                                 + 1.20127 w_k1_DIS_s114
                                 + 1.02543 w_k1_FDX_s114
                                 + 1.00879 w_k1_IBM_s114
                                 + 1.10013 w_k1_INTC_s114
                                 + 1.08517 w_k1_KO_s114
                                 + 0.777108 w_k1_MAT_s114
                                 + 1.14153 w_k1_MSFT_s114
                                 + 1.06425 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.779951 w_k1_ADM_s114
                                 + 0.67202 w_k1_AXP_s114
                                 + 0.688191 w_k1_DIS_s114
                                 + 0.749703 w_k1_FDX_s114
                                 + 0.789677 w_k1_IBM_s114
                                 + 0.813632 w_k1_INTC_s114
                                 + 0.75544 w_k1_KO_s114
                                 + 0.683954 w_k1_MAT_s114
                                 + 0.914982 w_k1_MSFT_s114
                                 + 0.743191 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.13794 w_k1_ADM_s114
                                 + 0.957091 w_k1_AXP_s114
                                 + 0.913918 w_k1_DIS_s114
                                 + 1.01728 w_k1_FDX_s114
                                 + 0.778964 w_k1_IBM_s114
                                 + 0.949998 w_k1_INTC_s114
                                 + 0.858973 w_k1_KO_s114
                                 + 0.641394 w_k1_MAT_s114
                                 + 1.02758 w_k1_MSFT_s114
                                 + 0.917353 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.831667 w_k1_ADM_s115
                                 + 0.748599 w_k1_AXP_s115
                                 + 0.658168 w_k1_DIS_s115
                                 + 0.838182 w_k1_FDX_s115
                                 + 0.853248 w_k1_IBM_s115
                                 + 0.987855 w_k1_INTC_s115 + 0.808 w_k1_KO_s115
                                 + 0.913195 w_k1_MAT_s115
                                 + 0.894221 w_k1_MSFT_s115
                                 + 0.86226 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.06062 w_k1_ADM_s115
                                 + 0.956094 w_k1_AXP_s115
                                 + 0.73699 w_k1_DIS_s115
                                 + 0.943247 w_k1_FDX_s115
                                 + 0.67934 w_k1_IBM_s115
                                 + 1.01407 w_k1_INTC_s115
                                 + 0.746095 w_k1_KO_s115
                                 + 0.689373 w_k1_MAT_s115
                                 + 0.838733 w_k1_MSFT_s115
                                 + 0.838215 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.952864 w_k1_ADM_s115
                                 + 1.08957 w_k1_AXP_s115
                                 + 0.816956 w_k1_DIS_s115
                                 + 0.854312 w_k1_FDX_s115
                                 + 1.27239 w_k1_IBM_s115
                                 + 0.971399 w_k1_INTC_s115
                                 + 0.931704 w_k1_KO_s115
                                 + 1.26769 w_k1_MAT_s115
                                 + 0.841068 w_k1_MSFT_s115
                                 + 0.940692 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.02722 w_k1_ADM_s115
                                 + 1.26258 w_k1_AXP_s115
                                 + 1.01351 w_k1_DIS_s115
                                 + 1.11597 w_k1_FDX_s115
                                 + 0.992753 w_k1_IBM_s115
                                 + 1.08776 w_k1_INTC_s115
                                 + 0.930502 w_k1_KO_s115
                                 + 1.13368 w_k1_MAT_s115
                                 + 1.0392 w_k1_MSFT_s115 + 1.02151 w_k1_PG_s115
                                 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.827038 w_k1_ADM_s115
                                 + 0.687192 w_k1_AXP_s115
                                 + 0.581408 w_k1_DIS_s115
                                 + 0.683903 w_k1_FDX_s115
                                 + 0.656628 w_k1_IBM_s115
                                 + 0.873613 w_k1_INTC_s115
                                 + 0.757701 w_k1_KO_s115
                                 + 0.576698 w_k1_MAT_s115
                                 + 0.714375 w_k1_MSFT_s115
                                 + 0.812922 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.1678 w_k1_ADM_s115
                                 + 0.853336 w_k1_AXP_s115
                                 + 0.874454 w_k1_DIS_s115
                                 + 0.931156 w_k1_FDX_s115
                                 + 1.10623 w_k1_IBM_s115
                                 + 1.08782 w_k1_INTC_s115
                                 + 0.905394 w_k1_KO_s115
                                 + 0.862991 w_k1_MAT_s115
                                 + 0.998335 w_k1_MSFT_s115
                                 + 0.905531 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.26445 w_k1_ADM_s115
                                 + 1.38312 w_k1_AXP_s115
                                 + 1.56099 w_k1_DIS_s115
                                 + 1.33914 w_k1_FDX_s115
                                 + 1.27847 w_k1_IBM_s115
                                 + 1.47665 w_k1_INTC_s115
                                 + 1.17044 w_k1_KO_s115 + 1.38686 w_k1_MAT_s115
                                 + 1.19821 w_k1_MSFT_s115
                                 + 1.18938 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.02902 w_k1_ADM_s115
                                 + 0.857213 w_k1_AXP_s115
                                 + 0.8928 w_k1_DIS_s115
                                 + 0.956224 w_k1_FDX_s115
                                 + 0.850817 w_k1_IBM_s115
                                 + 0.971996 w_k1_INTC_s115
                                 + 0.853654 w_k1_KO_s115
                                 + 0.810469 w_k1_MAT_s115
                                 + 1.02308 w_k1_MSFT_s115
                                 + 1.04394 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.10725 w_k1_ADM_s116
                                 + 0.861118 w_k1_AXP_s116
                                 + 0.916198 w_k1_DIS_s116
                                 + 1.07008 w_k1_FDX_s116
                                 + 0.720971 w_k1_IBM_s116
                                 + 0.704142 w_k1_INTC_s116
                                 + 0.758989 w_k1_KO_s116
                                 + 0.873316 w_k1_MAT_s116
                                 + 1.02697 w_k1_MSFT_s116
                                 + 0.844025 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.04424 w_k1_ADM_s116
                                 + 0.951414 w_k1_AXP_s116
                                 + 0.773409 w_k1_DIS_s116
                                 + 0.875711 w_k1_FDX_s116
                                 + 0.819026 w_k1_IBM_s116
                                 + 0.851616 w_k1_INTC_s116
                                 + 0.843856 w_k1_KO_s116
                                 + 0.749486 w_k1_MAT_s116
                                 + 0.86363 w_k1_MSFT_s116
                                 + 0.815441 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.710529 w_k1_ADM_s116
                                 + 0.776235 w_k1_AXP_s116
                                 + 0.623039 w_k1_DIS_s116
                                 + 0.687956 w_k1_FDX_s116
                                 + 0.803381 w_k1_IBM_s116
                                 + 0.862741 w_k1_INTC_s116
                                 + 0.749462 w_k1_KO_s116
                                 + 0.685919 w_k1_MAT_s116
                                 + 0.745932 w_k1_MSFT_s116
                                 + 0.828165 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.943611 w_k1_ADM_s116
                                 + 1.12703 w_k1_AXP_s116
                                 + 1.23531 w_k1_DIS_s116
                                 + 1.04102 w_k1_FDX_s116
                                 + 1.34353 w_k1_IBM_s116
                                 + 0.840998 w_k1_INTC_s116
                                 + 1.05963 w_k1_KO_s116 + 1.44145 w_k1_MAT_s116
                                 + 0.891222 w_k1_MSFT_s116
                                 + 0.951447 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.746457 w_k1_ADM_s116
                                 + 0.866671 w_k1_AXP_s116
                                 + 0.812783 w_k1_DIS_s116
                                 + 0.813353 w_k1_FDX_s116
                                 + 1.0507 w_k1_IBM_s116
                                 + 0.992594 w_k1_INTC_s116
                                 + 0.945176 w_k1_KO_s116
                                 + 0.812125 w_k1_MAT_s116
                                 + 0.83384 w_k1_MSFT_s116
                                 + 0.828404 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.18232 w_k1_ADM_s116
                                 + 1.6256 w_k1_AXP_s116 + 1.65786 w_k1_DIS_s116
                                 + 1.34152 w_k1_FDX_s116
                                 + 1.16492 w_k1_IBM_s116
                                 + 1.05378 w_k1_INTC_s116
                                 + 1.13926 w_k1_KO_s116 + 1.58959 w_k1_MAT_s116
                                 + 1.25228 w_k1_MSFT_s116
                                 + 1.15313 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.26101 w_k1_ADM_s116
                                 + 1.2373 w_k1_AXP_s116 + 1.0971 w_k1_DIS_s116
                                 + 1.10165 w_k1_FDX_s116
                                 + 1.20374 w_k1_IBM_s116
                                 + 1.02306 w_k1_INTC_s116
                                 + 1.07904 w_k1_KO_s116 + 1.01241 w_k1_MAT_s116
                                 + 1.04354 w_k1_MSFT_s116
                                 + 0.980856 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.917134 w_k1_ADM_s116
                                 + 0.903923 w_k1_AXP_s116
                                 + 1.02126 w_k1_DIS_s116
                                 + 0.88085 w_k1_FDX_s116
                                 + 1.24808 w_k1_IBM_s116
                                 + 0.819101 w_k1_INTC_s116
                                 + 1.01456 w_k1_KO_s116 + 1.00722 w_k1_MAT_s116
                                 + 0.775003 w_k1_MSFT_s116
                                 + 0.889746 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.689055 w_k1_ADM_s117
                                 + 0.862158 w_k1_AXP_s117
                                 + 0.857288 w_k1_DIS_s117
                                 + 0.950289 w_k1_FDX_s117
                                 + 0.857416 w_k1_IBM_s117
                                 + 0.90085 w_k1_INTC_s117
                                 + 0.817214 w_k1_KO_s117
                                 + 0.720082 w_k1_MAT_s117
                                 + 1.0227 w_k1_MSFT_s117 + 1.02465 w_k1_PG_s117
                                 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.733853 w_k1_ADM_s117
                                 + 0.701145 w_k1_AXP_s117
                                 + 0.874689 w_k1_DIS_s117
                                 + 0.906043 w_k1_FDX_s117
                                 + 0.687962 w_k1_IBM_s117
                                 + 0.798644 w_k1_INTC_s117
                                 + 0.848155 w_k1_KO_s117
                                 + 0.71722 w_k1_MAT_s117
                                 + 1.12641 w_k1_MSFT_s117
                                 + 0.93402 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.767354 w_k1_ADM_s117
                                 + 1.06923 w_k1_AXP_s117
                                 + 0.92982 w_k1_DIS_s117
                                 + 0.902941 w_k1_FDX_s117
                                 + 1.06252 w_k1_IBM_s117
                                 + 0.930562 w_k1_INTC_s117
                                 + 1.00406 w_k1_KO_s117
                                 + 0.845853 w_k1_MAT_s117
                                 + 0.928211 w_k1_MSFT_s117
                                 + 1.06841 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.820833 w_k1_ADM_s117
                                 + 0.879424 w_k1_AXP_s117
                                 + 0.597442 w_k1_DIS_s117
                                 + 0.778416 w_k1_FDX_s117
                                 + 1.18599 w_k1_IBM_s117
                                 + 0.806735 w_k1_INTC_s117
                                 + 0.862139 w_k1_KO_s117
                                 + 0.917202 w_k1_MAT_s117
                                 + 0.767964 w_k1_MSFT_s117 + 1.044 w_k1_PG_s117
                                 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.37591 w_k1_ADM_s117
                                 + 1.76612 w_k1_AXP_s117
                                 + 1.64514 w_k1_DIS_s117
                                 + 1.36124 w_k1_FDX_s117
                                 + 1.54377 w_k1_IBM_s117
                                 + 1.36941 w_k1_INTC_s117
                                 + 1.37293 w_k1_KO_s117 + 1.49119 w_k1_MAT_s117
                                 + 1.26087 w_k1_MSFT_s117
                                 + 1.28499 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.18699 w_k1_ADM_s117
                                 + 1.3685 w_k1_AXP_s117
                                 + 0.985122 w_k1_DIS_s117
                                 + 1.16328 w_k1_FDX_s117
                                 + 0.802422 w_k1_IBM_s117
                                 + 1.07161 w_k1_INTC_s117
                                 + 0.849233 w_k1_KO_s117
                                 + 0.855102 w_k1_MAT_s117
                                 + 0.987932 w_k1_MSFT_s117
                                 + 1.17488 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.813305 w_k1_ADM_s117
                                 + 1.30012 w_k1_AXP_s117
                                 + 1.01461 w_k1_DIS_s117
                                 + 1.00212 w_k1_FDX_s117
                                 + 1.37497 w_k1_IBM_s117
                                 + 0.937204 w_k1_INTC_s117
                                 + 1.0868 w_k1_KO_s117 + 1.35403 w_k1_MAT_s117
                                 + 0.892462 w_k1_MSFT_s117
                                 + 1.22977 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.08127 w_k1_ADM_s117
                                 + 1.3492 w_k1_AXP_s117 + 1.42627 w_k1_DIS_s117
                                 + 1.28517 w_k1_FDX_s117 + 1.0509 w_k1_IBM_s117
                                 + 0.879817 w_k1_INTC_s117
                                 + 1.06547 w_k1_KO_s117 + 1.25962 w_k1_MAT_s117
                                 + 1.17061 w_k1_MSFT_s117
                                 + 1.15872 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0230551910864574 w_k1_ADM_s000
                                 + 0.0257595955009796 w_k1_ADM_s001
                                 + 0.0526086748766366 w_k1_ADM_s002
                                 + 0.0150110887577442 w_k1_ADM_s003
                                 + 0.036180080221337 w_k1_ADM_s004
                                 + 0.0166385684999554 w_k1_ADM_s005
                                 + 0.0267052017869494 w_k1_ADM_s006
                                 + 0.0543072524342887 w_k1_ADM_s007
                                 + 0.0295923593385029 w_k1_ADM_s010
                                 + 0.0333721883845182 w_k1_ADM_s011
                                 + 0.0166584546023575 w_k1_ADM_s012
                                 + 0.0248328963112858 w_k1_ADM_s013
                                 + 0.0534327803373099 w_k1_ADM_s014
                                 + 0.060025440623595 w_k1_ADM_s015
                                 + 0.0514828124112271 w_k1_ADM_s016
                                 + 0.0762813439581371 w_k1_ADM_s017
                                 + 0.0184518308061606 w_k1_ADM_s100
                                 + 0.0158296890657523 w_k1_ADM_s101
                                 + 0.0249047864311859 w_k1_ADM_s102
                                 + 0.00945706529245169 w_k1_ADM_s103
                                 + 0.0173984106151028 w_k1_ADM_s104
                                 + 0.0218744399816536 w_k1_ADM_s105
                                 + 0.00856334147173379 w_k1_ADM_s106
                                 + 0.0096665495751267 w_k1_ADM_s107
                                 + 0.0752478192170498 w_k1_ADM_s110
                                 + 0.0317725252254381 w_k1_ADM_s111
                                 + 0.0338706656024825 w_k1_ADM_s112
                                 + 0.0551893809878754 w_k1_ADM_s113
                                 + 0.0368565409090169 w_k1_ADM_s114
                                 + 0.0240760853546673 w_k1_ADM_s115
                                 + 0.0181117220543488 w_k1_ADM_s116
                                 + 0.0215883969520038 w_k1_ADM_s117
                                 + 0.0203832071391859 w_k1_AXP_s000
                                 + 0.0274580404433715 w_k1_AXP_s001
                                 + 0.05023494413747 w_k1_AXP_s002
                                 + 0.0144326482043384 w_k1_AXP_s003
                                 + 0.0301463604579271 w_k1_AXP_s004
                                 + 0.0170119019288464 w_k1_AXP_s005
                                 + 0.028978204966975 w_k1_AXP_s006
                                 + 0.0461978568678648 w_k1_AXP_s007
                                 + 0.0335809380984845 w_k1_AXP_s010
                                 + 0.0336423664513537 w_k1_AXP_s011
                                 + 0.0194482566700083 w_k1_AXP_s012
                                 + 0.0280396020914963 w_k1_AXP_s013
                                 + 0.051570519178456 w_k1_AXP_s014
                                 + 0.0605270607633068 w_k1_AXP_s015
                                 + 0.0552800351917629 w_k1_AXP_s016
                                 + 0.0734149264375441 w_k1_AXP_s017
                                 + 0.0176650434766703 w_k1_AXP_s100
                                 + 0.0144132319516299 w_k1_AXP_s101
                                 + 0.027391063911333 w_k1_AXP_s102
                                 + 0.00953855127042658 w_k1_AXP_s103
                                 + 0.0194173413880963 w_k1_AXP_s104
                                 + 0.0236167524445879 w_k1_AXP_s105
                                 + 0.00857706271301793 w_k1_AXP_s106
                                 + 0.00931265802987209 w_k1_AXP_s107
                                 + 0.0763202080128409 w_k1_AXP_s110
                                 + 0.0376381008980832 w_k1_AXP_s111
                                 + 0.0384228532313217 w_k1_AXP_s112
                                 + 0.0609750057175437 w_k1_AXP_s113
                                 + 0.0360197857984456 w_k1_AXP_s114
                                 + 0.0245014960922304 w_k1_AXP_s115
                                 + 0.019693000631146 w_k1_AXP_s116
                                 + 0.0264064968434087 w_k1_AXP_s117
                                 + 0.0217829696789233 w_k1_DIS_s000
                                 + 0.0311966942707678 w_k1_DIS_s001
                                 + 0.0525412188953655 w_k1_DIS_s002
                                 + 0.0138198094989961 w_k1_DIS_s003
                                 + 0.0349404325603321 w_k1_DIS_s004
                                 + 0.0172666613510517 w_k1_DIS_s005
                                 + 0.0288323667847246 w_k1_DIS_s006
                                 + 0.048525116485101 w_k1_DIS_s007
                                 + 0.0340503526824777 w_k1_DIS_s010
                                 + 0.0349517926458523 w_k1_DIS_s011
                                 + 0.0202363092832284 w_k1_DIS_s012
                                 + 0.0261138747704728 w_k1_DIS_s013
                                 + 0.0522049074638436 w_k1_DIS_s014
                                 + 0.0589855258659618 w_k1_DIS_s015
                                 + 0.0536692216250615 w_k1_DIS_s016
                                 + 0.0792954441024561 w_k1_DIS_s017
                                 + 0.0173871211852711 w_k1_DIS_s100
                                 + 0.0149649276800792 w_k1_DIS_s101
                                 + 0.0256800260120313 w_k1_DIS_s102
                                 + 0.00882761796655956 w_k1_DIS_s103
                                 + 0.0188850303088897 w_k1_DIS_s104
                                 + 0.0230438058741079 w_k1_DIS_s105
                                 + 0.00787388456044708 w_k1_DIS_s106
                                 + 0.00997554526752502 w_k1_DIS_s107
                                 + 0.073349626700977 w_k1_DIS_s110
                                 + 0.0366664239079195 w_k1_DIS_s111
                                 + 0.0351099766034362 w_k1_DIS_s112
                                 + 0.0616505130286821 w_k1_DIS_s113
                                 + 0.0383255234165282 w_k1_DIS_s114
                                 + 0.0226487071098349 w_k1_DIS_s115
                                 + 0.0193348354385264 w_k1_DIS_s116
                                 + 0.0242072974284414 w_k1_DIS_s117
                                 + 0.0208789702482257 w_k1_FDX_s000
                                 + 0.0293400952857079 w_k1_FDX_s001
                                 + 0.0495163915286259 w_k1_FDX_s002
                                 + 0.0139200463974737 w_k1_FDX_s003
                                 + 0.0331623356301394 w_k1_FDX_s004
                                 + 0.0171285719586832 w_k1_FDX_s005
                                 + 0.02733598367526 w_k1_FDX_s006
                                 + 0.0481683341031285 w_k1_FDX_s007
                                 + 0.0314426076078732 w_k1_FDX_s010
                                 + 0.0318163864600464 w_k1_FDX_s011
                                 + 0.0187399637720704 w_k1_FDX_s012
                                 + 0.0234680039418773 w_k1_FDX_s013
                                 + 0.0487196030754176 w_k1_FDX_s014
                                 + 0.058910095129455 w_k1_FDX_s015
                                 + 0.0503287732786972 w_k1_FDX_s016
                                 + 0.0748326186281254 w_k1_FDX_s017
                                 + 0.0180887170704227 w_k1_FDX_s100
                                 + 0.0144571755109018 w_k1_FDX_s101
                                 + 0.0269519382081027 w_k1_FDX_s102
                                 + 0.00857195779942346 w_k1_FDX_s103
                                 + 0.018866621577664 w_k1_FDX_s104
                                 + 0.0239439425383055 w_k1_FDX_s105
                                 + 0.00811357916598204 w_k1_FDX_s106
                                 + 0.00903539945627541 w_k1_FDX_s107
                                 + 0.0702354363102674 w_k1_FDX_s110
                                 + 0.0359247970534642 w_k1_FDX_s111
                                 + 0.0348814452528895 w_k1_FDX_s112
                                 + 0.0605731571480466 w_k1_FDX_s113
                                 + 0.035932789623499 w_k1_FDX_s114
                                 + 0.0234253181159417 w_k1_FDX_s115
                                 + 0.0181369937715057 w_k1_FDX_s116
                                 + 0.0245379459647066 w_k1_FDX_s117
                                 + 0.0223996789070169 w_k1_IBM_s000
                                 + 0.0297173914695789 w_k1_IBM_s001
                                 + 0.0520421328096549 w_k1_IBM_s002
                                 + 0.0142828287015589 w_k1_IBM_s003
                                 + 0.0331782806623631 w_k1_IBM_s004
                                 + 0.0170256715741949 w_k1_IBM_s005
                                 + 0.0258316714562868 w_k1_IBM_s006
                                 + 0.0498033435681615 w_k1_IBM_s007
                                 + 0.0311122531173625 w_k1_IBM_s010
                                 + 0.033141786739301 w_k1_IBM_s011
                                 + 0.0195119000794633 w_k1_IBM_s012
                                 + 0.0240322427104751 w_k1_IBM_s013
                                 + 0.0503526628882573 w_k1_IBM_s014
                                 + 0.057682440109528 w_k1_IBM_s015
                                 + 0.0535284104810057 w_k1_IBM_s016
                                 + 0.0736714203911266 w_k1_IBM_s017
                                 + 0.0170891142393894 w_k1_IBM_s100
                                 + 0.0148714599537237 w_k1_IBM_s101
                                 + 0.0269050840941032 w_k1_IBM_s102
                                 + 0.00873433056927016 w_k1_IBM_s103
                                 + 0.0193189680934354 w_k1_IBM_s104
                                 + 0.0239574739663439 w_k1_IBM_s105
                                 + 0.00855021579556068 w_k1_IBM_s106
                                 + 0.0100076827440526 w_k1_IBM_s107
                                 + 0.0750037290460667 w_k1_IBM_s110
                                 + 0.0382010989018841 w_k1_IBM_s111
                                 + 0.0360346773471363 w_k1_IBM_s112
                                 + 0.0607831783456176 w_k1_IBM_s113
                                 + 0.0368839093542129 w_k1_IBM_s114
                                 + 0.0239603421669699 w_k1_IBM_s115
                                 + 0.0192327161823951 w_k1_IBM_s116
                                 + 0.0249211838699265 w_k1_IBM_s117
                                 + 0.0206116051697201 w_k1_INTC_s000
                                 + 0.0327649982911511 w_k1_INTC_s001
                                 + 0.044111143544674 w_k1_INTC_s002
                                 + 0.0150483760838675 w_k1_INTC_s003
                                 + 0.0305344739054047 w_k1_INTC_s004
                                 + 0.0189756465274153 w_k1_INTC_s005
                                 + 0.0280463297926419 w_k1_INTC_s006
                                 + 0.0415889250505352 w_k1_INTC_s007
                                 + 0.0282333276100362 w_k1_INTC_s010
                                 + 0.0279408097262725 w_k1_INTC_s011
                                 + 0.0209411470167109 w_k1_INTC_s012
                                 + 0.0203129079291199 w_k1_INTC_s013
                                 + 0.0420450224574417 w_k1_INTC_s014
                                 + 0.0621225065252341 w_k1_INTC_s015
                                 + 0.0496205536367855 w_k1_INTC_s016
                                 + 0.0715533633611409 w_k1_INTC_s017
                                 + 0.0171325681574416 w_k1_INTC_s100
                                 + 0.0142819950457224 w_k1_INTC_s101
                                 + 0.0271732202268075 w_k1_INTC_s102
                                 + 0.00883060803936481 w_k1_INTC_s103
                                 + 0.018092799230306 w_k1_INTC_s104
                                 + 0.0265127883498656 w_k1_INTC_s105
                                 + 0.00833038142672861 w_k1_INTC_s106
                                 + 0.00988010287635381 w_k1_INTC_s107
                                 + 0.0744186459327388 w_k1_INTC_s110
                                 + 0.0336295313560005 w_k1_INTC_s111
                                 + 0.0356662168688395 w_k1_INTC_s112
                                 + 0.0603691521002263 w_k1_INTC_s113
                                 + 0.0372050660075497 w_k1_INTC_s114
                                 + 0.0253185470144073 w_k1_INTC_s115
                                 + 0.0163491401758174 w_k1_INTC_s116
                                 + 0.0224456374315659 w_k1_INTC_s117
                                 + 0.0222877028608599 w_k1_KO_s000
                                 + 0.0286046859904653 w_k1_KO_s001
                                 + 0.0485180678151726 w_k1_KO_s002
                                 + 0.0140269428662305 w_k1_KO_s003
                                 + 0.0337588923686897 w_k1_KO_s004
                                 + 0.0183086612289125 w_k1_KO_s005
                                 + 0.0269629190985036 w_k1_KO_s006
                                 + 0.0469525530799291 w_k1_KO_s007
                                 + 0.0302065548353075 w_k1_KO_s010
                                 + 0.0330184809115271 w_k1_KO_s011
                                 + 0.017625104886956 w_k1_KO_s012
                                 + 0.0252997968409397 w_k1_KO_s013
                                 + 0.0520754418306218 w_k1_KO_s014
                                 + 0.062452954059175 w_k1_KO_s015
                                 + 0.0494370191777143 w_k1_KO_s016
                                 + 0.0865450461943086 w_k1_KO_s017
                                 + 0.0179488576235961 w_k1_KO_s100
                                 + 0.0142346801975914 w_k1_KO_s101
                                 + 0.0269636321799998 w_k1_KO_s102
                                 + 0.00850348858917691 w_k1_KO_s103
                                 + 0.0200184823129073 w_k1_KO_s104
                                 + 0.0225058981365673 w_k1_KO_s105
                                 + 0.00852365818065373 w_k1_KO_s106
                                 + 0.0112806206256551 w_k1_KO_s107
                                 + 0.0741942594073825 w_k1_KO_s110
                                 + 0.0350702009367848 w_k1_KO_s111
                                 + 0.0361986652124549 w_k1_KO_s112
                                 + 0.0632808678416301 w_k1_KO_s113
                                 + 0.0373992153616795 w_k1_KO_s114
                                 + 0.0215419325904689 w_k1_KO_s115
                                 + 0.017475027384102 w_k1_KO_s116
                                 + 0.0232941053707777 w_k1_KO_s117
                                 + 0.0206126211638251 w_k1_MAT_s000
                                 + 0.0270757321669363 w_k1_MAT_s001
                                 + 0.0413150694546933 w_k1_MAT_s002
                                 + 0.0169887396198255 w_k1_MAT_s003
                                 + 0.0321696867255428 w_k1_MAT_s004
                                 + 0.0213542781009644 w_k1_MAT_s005
                                 + 0.0378481684205024 w_k1_MAT_s006
                                 + 0.0489493859397708 w_k1_MAT_s007
                                 + 0.0341078267144382 w_k1_MAT_s010
                                 + 0.039525176565715 w_k1_MAT_s011
                                 + 0.0174409855262426 w_k1_MAT_s012
                                 + 0.0251269615482745 w_k1_MAT_s013
                                 + 0.05582688127759 w_k1_MAT_s014
                                 + 0.0621846778642735 w_k1_MAT_s015
                                 + 0.0477595065276005 w_k1_MAT_s016
                                 + 0.0768199807180017 w_k1_MAT_s017
                                 + 0.0191912181750843 w_k1_MAT_s100
                                 + 0.0128217224270739 w_k1_MAT_s101
                                 + 0.0249193749775268 w_k1_MAT_s102
                                 + 0.00871557123561451 w_k1_MAT_s103
                                 + 0.0195798655814302 w_k1_MAT_s104
                                 + 0.0246336690380706 w_k1_MAT_s105
                                 + 0.00698224316017417 w_k1_MAT_s106
                                 + 0.0124724604640383 w_k1_MAT_s107
                                 + 0.0733999743052736 w_k1_MAT_s110
                                 + 0.0393974860717755 w_k1_MAT_s111
                                 + 0.038539783648596 w_k1_MAT_s112
                                 + 0.073664584031688 w_k1_MAT_s113
                                 + 0.0374087238874312 w_k1_MAT_s114
                                 + 0.0245193619063879 w_k1_MAT_s115
                                 + 0.0192374323480087 w_k1_MAT_s116
                                 + 0.0236119977030796 w_k1_MAT_s117
                                 + 0.0218650917265645 w_k1_MSFT_s000
                                 + 0.0294402396710594 w_k1_MSFT_s001
                                 + 0.0483160390166001 w_k1_MSFT_s002
                                 + 0.0133570156526422 w_k1_MSFT_s003
                                 + 0.0319024617770613 w_k1_MSFT_s004
                                 + 0.0193089305033197 w_k1_MSFT_s005
                                 + 0.029365693841565 w_k1_MSFT_s006
                                 + 0.0512658447051632 w_k1_MSFT_s007
                                 + 0.0304344890209185 w_k1_MSFT_s010
                                 + 0.0326706023911464 w_k1_MSFT_s011
                                 + 0.016935932305117 w_k1_MSFT_s012
                                 + 0.0242522442108262 w_k1_MSFT_s013
                                 + 0.0454445392974638 w_k1_MSFT_s014
                                 + 0.0589214684687147 w_k1_MSFT_s015
                                 + 0.0580923447056737 w_k1_MSFT_s016
                                 + 0.0837234382890746 w_k1_MSFT_s017
                                 + 0.0174140991096698 w_k1_MSFT_s100
                                 + 0.0133525472934224 w_k1_MSFT_s101
                                 + 0.0275070994664225 w_k1_MSFT_s102
                                 + 0.00881165583747211 w_k1_MSFT_s103
                                 + 0.0179068043363324 w_k1_MSFT_s104
                                 + 0.0259176296137691 w_k1_MSFT_s105
                                 + 0.00800653170385423 w_k1_MSFT_s106
                                 + 0.00962980989295797 w_k1_MSFT_s107
                                 + 0.0667084111704277 w_k1_MSFT_s110
                                 + 0.0356835406175685 w_k1_MSFT_s111
                                 + 0.0327294362050011 w_k1_MSFT_s112
                                 + 0.0622174489057034 w_k1_MSFT_s113
                                 + 0.0385774347143895 w_k1_MSFT_s114
                                 + 0.0228895326109814 w_k1_MSFT_s115
                                 + 0.0171620428277451 w_k1_MSFT_s116
                                 + 0.0245808045339634 w_k1_MSFT_s117
                                 + 0.0209535661147437 w_k1_PG_s000
                                 + 0.0300496118688046 w_k1_PG_s001
                                 + 0.0506627439823154 w_k1_PG_s002
                                 + 0.0146585769663126 w_k1_PG_s003
                                 + 0.0339699611824445 w_k1_PG_s004
                                 + 0.0176582788540426 w_k1_PG_s005
                                 + 0.0276609954340094 w_k1_PG_s006
                                 + 0.050274741843416 w_k1_PG_s007
                                 + 0.0328045449307459 w_k1_PG_s010
                                 + 0.0312900671989206 w_k1_PG_s011
                                 + 0.0180176032876392 w_k1_PG_s012
                                 + 0.0243069562997938 w_k1_PG_s013
                                 + 0.0514571544734294 w_k1_PG_s014
                                 + 0.0550353168670745 w_k1_PG_s015
                                 + 0.0540365587038212 w_k1_PG_s016
                                 + 0.0784832490244711 w_k1_PG_s017
                                 + 0.015637142260822 w_k1_PG_s100
                                 + 0.0147564878692942 w_k1_PG_s101
                                 + 0.0270374794756051 w_k1_PG_s102
                                 + 0.00804080583251093 w_k1_PG_s103
                                 + 0.0177049258193627 w_k1_PG_s104
                                 + 0.0229421554885083 w_k1_PG_s105
                                 + 0.00836548759336599 w_k1_PG_s106
                                 + 0.009448097839811 w_k1_PG_s107
                                 + 0.0699447034399869 w_k1_PG_s110
                                 + 0.0336787744649654 w_k1_PG_s111
                                 + 0.0348442449608931 w_k1_PG_s112
                                 + 0.0577184973196702 w_k1_PG_s113
                                 + 0.0352561591022598 w_k1_PG_s114
                                 + 0.0231456752008149 w_k1_PG_s115
                                 + 0.0166439587923593 w_k1_PG_s116
                                 + 0.0263818219548745 w_k1_PG_s117 >= 1.14
Bounds
 0 <= w_0_ADM <= 1
 0 <= w_0_AXP <= 1
 0 <= w_0_DIS <= 1
 0 <= w_0_FDX <= 1
 0 <= w_0_IBM <= 1
 0 <= w_0_INTC <= 1
 0 <= w_0_KO <= 1
 0 <= w_0_MAT <= 1
 0 <= w_0_MSFT <= 1
 0 <= w_0_PG <= 1
 0 <= w_k1_ADM_s0 <= 1
 0 <= w_k1_ADM_s1 <= 1
 0 <= w_k1_ADM_s00 <= 1
 0 <= w_k1_ADM_s01 <= 1
 0 <= w_k1_ADM_s10 <= 1
 0 <= w_k1_ADM_s11 <= 1
 0 <= w_k1_ADM_s000 <= 1
 0 <= w_k1_ADM_s001 <= 1
 0 <= w_k1_ADM_s002 <= 1
 0 <= w_k1_ADM_s003 <= 1
 0 <= w_k1_ADM_s004 <= 1
 0 <= w_k1_ADM_s005 <= 1
 0 <= w_k1_ADM_s006 <= 1
 0 <= w_k1_ADM_s007 <= 1
 0 <= w_k1_ADM_s010 <= 1
 0 <= w_k1_ADM_s011 <= 1
 0 <= w_k1_ADM_s012 <= 1
 0 <= w_k1_ADM_s013 <= 1
 0 <= w_k1_ADM_s014 <= 1
 0 <= w_k1_ADM_s015 <= 1
 0 <= w_k1_ADM_s016 <= 1
 0 <= w_k1_ADM_s017 <= 1
 0 <= w_k1_ADM_s100 <= 1
 0 <= w_k1_ADM_s101 <= 1
 0 <= w_k1_ADM_s102 <= 1
 0 <= w_k1_ADM_s103 <= 1
 0 <= w_k1_ADM_s104 <= 1
 0 <= w_k1_ADM_s105 <= 1
 0 <= w_k1_ADM_s106 <= 1
 0 <= w_k1_ADM_s107 <= 1
 0 <= w_k1_ADM_s110 <= 1
 0 <= w_k1_ADM_s111 <= 1
 0 <= w_k1_ADM_s112 <= 1
 0 <= w_k1_ADM_s113 <= 1
 0 <= w_k1_ADM_s114 <= 1
 0 <= w_k1_ADM_s115 <= 1
 0 <= w_k1_ADM_s116 <= 1
 0 <= w_k1_ADM_s117 <= 1
 0 <= w_k1_AXP_s0 <= 1
 0 <= w_k1_AXP_s1 <= 1
 0 <= w_k1_AXP_s00 <= 1
 0 <= w_k1_AXP_s01 <= 1
 0 <= w_k1_AXP_s10 <= 1
 0 <= w_k1_AXP_s11 <= 1
 0 <= w_k1_AXP_s000 <= 1
 0 <= w_k1_AXP_s001 <= 1
 0 <= w_k1_AXP_s002 <= 1
 0 <= w_k1_AXP_s003 <= 1
 0 <= w_k1_AXP_s004 <= 1
 0 <= w_k1_AXP_s005 <= 1
 0 <= w_k1_AXP_s006 <= 1
 0 <= w_k1_AXP_s007 <= 1
 0 <= w_k1_AXP_s010 <= 1
 0 <= w_k1_AXP_s011 <= 1
 0 <= w_k1_AXP_s012 <= 1
 0 <= w_k1_AXP_s013 <= 1
 0 <= w_k1_AXP_s014 <= 1
 0 <= w_k1_AXP_s015 <= 1
 0 <= w_k1_AXP_s016 <= 1
 0 <= w_k1_AXP_s017 <= 1
 0 <= w_k1_AXP_s100 <= 1
 0 <= w_k1_AXP_s101 <= 1
 0 <= w_k1_AXP_s102 <= 1
 0 <= w_k1_AXP_s103 <= 1
 0 <= w_k1_AXP_s104 <= 1
 0 <= w_k1_AXP_s105 <= 1
 0 <= w_k1_AXP_s106 <= 1
 0 <= w_k1_AXP_s107 <= 1
 0 <= w_k1_AXP_s110 <= 1
 0 <= w_k1_AXP_s111 <= 1
 0 <= w_k1_AXP_s112 <= 1
 0 <= w_k1_AXP_s113 <= 1
 0 <= w_k1_AXP_s114 <= 1
 0 <= w_k1_AXP_s115 <= 1
 0 <= w_k1_AXP_s116 <= 1
 0 <= w_k1_AXP_s117 <= 1
 0 <= w_k1_DIS_s0 <= 1
 0 <= w_k1_DIS_s1 <= 1
 0 <= w_k1_DIS_s00 <= 1
 0 <= w_k1_DIS_s01 <= 1
 0 <= w_k1_DIS_s10 <= 1
 0 <= w_k1_DIS_s11 <= 1
 0 <= w_k1_DIS_s000 <= 1
 0 <= w_k1_DIS_s001 <= 1
 0 <= w_k1_DIS_s002 <= 1
 0 <= w_k1_DIS_s003 <= 1
 0 <= w_k1_DIS_s004 <= 1
 0 <= w_k1_DIS_s005 <= 1
 0 <= w_k1_DIS_s006 <= 1
 0 <= w_k1_DIS_s007 <= 1
 0 <= w_k1_DIS_s010 <= 1
 0 <= w_k1_DIS_s011 <= 1
 0 <= w_k1_DIS_s012 <= 1
 0 <= w_k1_DIS_s013 <= 1
 0 <= w_k1_DIS_s014 <= 1
 0 <= w_k1_DIS_s015 <= 1
 0 <= w_k1_DIS_s016 <= 1
 0 <= w_k1_DIS_s017 <= 1
 0 <= w_k1_DIS_s100 <= 1
 0 <= w_k1_DIS_s101 <= 1
 0 <= w_k1_DIS_s102 <= 1
 0 <= w_k1_DIS_s103 <= 1
 0 <= w_k1_DIS_s104 <= 1
 0 <= w_k1_DIS_s105 <= 1
 0 <= w_k1_DIS_s106 <= 1
 0 <= w_k1_DIS_s107 <= 1
 0 <= w_k1_DIS_s110 <= 1
 0 <= w_k1_DIS_s111 <= 1
 0 <= w_k1_DIS_s112 <= 1
 0 <= w_k1_DIS_s113 <= 1
 0 <= w_k1_DIS_s114 <= 1
 0 <= w_k1_DIS_s115 <= 1
 0 <= w_k1_DIS_s116 <= 1
 0 <= w_k1_DIS_s117 <= 1
 0 <= w_k1_FDX_s0 <= 1
 0 <= w_k1_FDX_s1 <= 1
 0 <= w_k1_FDX_s00 <= 1
 0 <= w_k1_FDX_s01 <= 1
 0 <= w_k1_FDX_s10 <= 1
 0 <= w_k1_FDX_s11 <= 1
 0 <= w_k1_FDX_s000 <= 1
 0 <= w_k1_FDX_s001 <= 1
 0 <= w_k1_FDX_s002 <= 1
 0 <= w_k1_FDX_s003 <= 1
 0 <= w_k1_FDX_s004 <= 1
 0 <= w_k1_FDX_s005 <= 1
 0 <= w_k1_FDX_s006 <= 1
 0 <= w_k1_FDX_s007 <= 1
 0 <= w_k1_FDX_s010 <= 1
 0 <= w_k1_FDX_s011 <= 1
 0 <= w_k1_FDX_s012 <= 1
 0 <= w_k1_FDX_s013 <= 1
 0 <= w_k1_FDX_s014 <= 1
 0 <= w_k1_FDX_s015 <= 1
 0 <= w_k1_FDX_s016 <= 1
 0 <= w_k1_FDX_s017 <= 1
 0 <= w_k1_FDX_s100 <= 1
 0 <= w_k1_FDX_s101 <= 1
 0 <= w_k1_FDX_s102 <= 1
 0 <= w_k1_FDX_s103 <= 1
 0 <= w_k1_FDX_s104 <= 1
 0 <= w_k1_FDX_s105 <= 1
 0 <= w_k1_FDX_s106 <= 1
 0 <= w_k1_FDX_s107 <= 1
 0 <= w_k1_FDX_s110 <= 1
 0 <= w_k1_FDX_s111 <= 1
 0 <= w_k1_FDX_s112 <= 1
 0 <= w_k1_FDX_s113 <= 1
 0 <= w_k1_FDX_s114 <= 1
 0 <= w_k1_FDX_s115 <= 1
 0 <= w_k1_FDX_s116 <= 1
 0 <= w_k1_FDX_s117 <= 1
 0 <= w_k1_IBM_s0 <= 1
 0 <= w_k1_IBM_s1 <= 1
 0 <= w_k1_IBM_s00 <= 1
 0 <= w_k1_IBM_s01 <= 1
 0 <= w_k1_IBM_s10 <= 1
 0 <= w_k1_IBM_s11 <= 1
 0 <= w_k1_IBM_s000 <= 1
 0 <= w_k1_IBM_s001 <= 1
 0 <= w_k1_IBM_s002 <= 1
 0 <= w_k1_IBM_s003 <= 1
 0 <= w_k1_IBM_s004 <= 1
 0 <= w_k1_IBM_s005 <= 1
 0 <= w_k1_IBM_s006 <= 1
 0 <= w_k1_IBM_s007 <= 1
 0 <= w_k1_IBM_s010 <= 1
 0 <= w_k1_IBM_s011 <= 1
 0 <= w_k1_IBM_s012 <= 1
 0 <= w_k1_IBM_s013 <= 1
 0 <= w_k1_IBM_s014 <= 1
 0 <= w_k1_IBM_s015 <= 1
 0 <= w_k1_IBM_s016 <= 1
 0 <= w_k1_IBM_s017 <= 1
 0 <= w_k1_IBM_s100 <= 1
 0 <= w_k1_IBM_s101 <= 1
 0 <= w_k1_IBM_s102 <= 1
 0 <= w_k1_IBM_s103 <= 1
 0 <= w_k1_IBM_s104 <= 1
 0 <= w_k1_IBM_s105 <= 1
 0 <= w_k1_IBM_s106 <= 1
 0 <= w_k1_IBM_s107 <= 1
 0 <= w_k1_IBM_s110 <= 1
 0 <= w_k1_IBM_s111 <= 1
 0 <= w_k1_IBM_s112 <= 1
 0 <= w_k1_IBM_s113 <= 1
 0 <= w_k1_IBM_s114 <= 1
 0 <= w_k1_IBM_s115 <= 1
 0 <= w_k1_IBM_s116 <= 1
 0 <= w_k1_IBM_s117 <= 1
 0 <= w_k1_INTC_s0 <= 1
 0 <= w_k1_INTC_s1 <= 1
 0 <= w_k1_INTC_s00 <= 1
 0 <= w_k1_INTC_s01 <= 1
 0 <= w_k1_INTC_s10 <= 1
 0 <= w_k1_INTC_s11 <= 1
 0 <= w_k1_INTC_s000 <= 1
 0 <= w_k1_INTC_s001 <= 1
 0 <= w_k1_INTC_s002 <= 1
 0 <= w_k1_INTC_s003 <= 1
 0 <= w_k1_INTC_s004 <= 1
 0 <= w_k1_INTC_s005 <= 1
 0 <= w_k1_INTC_s006 <= 1
 0 <= w_k1_INTC_s007 <= 1
 0 <= w_k1_INTC_s010 <= 1
 0 <= w_k1_INTC_s011 <= 1
 0 <= w_k1_INTC_s012 <= 1
 0 <= w_k1_INTC_s013 <= 1
 0 <= w_k1_INTC_s014 <= 1
 0 <= w_k1_INTC_s015 <= 1
 0 <= w_k1_INTC_s016 <= 1
 0 <= w_k1_INTC_s017 <= 1
 0 <= w_k1_INTC_s100 <= 1
 0 <= w_k1_INTC_s101 <= 1
 0 <= w_k1_INTC_s102 <= 1
 0 <= w_k1_INTC_s103 <= 1
 0 <= w_k1_INTC_s104 <= 1
 0 <= w_k1_INTC_s105 <= 1
 0 <= w_k1_INTC_s106 <= 1
 0 <= w_k1_INTC_s107 <= 1
 0 <= w_k1_INTC_s110 <= 1
 0 <= w_k1_INTC_s111 <= 1
 0 <= w_k1_INTC_s112 <= 1
 0 <= w_k1_INTC_s113 <= 1
 0 <= w_k1_INTC_s114 <= 1
 0 <= w_k1_INTC_s115 <= 1
 0 <= w_k1_INTC_s116 <= 1
 0 <= w_k1_INTC_s117 <= 1
 0 <= w_k1_KO_s0 <= 1
 0 <= w_k1_KO_s1 <= 1
 0 <= w_k1_KO_s00 <= 1
 0 <= w_k1_KO_s01 <= 1
 0 <= w_k1_KO_s10 <= 1
 0 <= w_k1_KO_s11 <= 1
 0 <= w_k1_KO_s000 <= 1
 0 <= w_k1_KO_s001 <= 1
 0 <= w_k1_KO_s002 <= 1
 0 <= w_k1_KO_s003 <= 1
 0 <= w_k1_KO_s004 <= 1
 0 <= w_k1_KO_s005 <= 1
 0 <= w_k1_KO_s006 <= 1
 0 <= w_k1_KO_s007 <= 1
 0 <= w_k1_KO_s010 <= 1
 0 <= w_k1_KO_s011 <= 1
 0 <= w_k1_KO_s012 <= 1
 0 <= w_k1_KO_s013 <= 1
 0 <= w_k1_KO_s014 <= 1
 0 <= w_k1_KO_s015 <= 1
 0 <= w_k1_KO_s016 <= 1
 0 <= w_k1_KO_s017 <= 1
 0 <= w_k1_KO_s100 <= 1
 0 <= w_k1_KO_s101 <= 1
 0 <= w_k1_KO_s102 <= 1
 0 <= w_k1_KO_s103 <= 1
 0 <= w_k1_KO_s104 <= 1
 0 <= w_k1_KO_s105 <= 1
 0 <= w_k1_KO_s106 <= 1
 0 <= w_k1_KO_s107 <= 1
 0 <= w_k1_KO_s110 <= 1
 0 <= w_k1_KO_s111 <= 1
 0 <= w_k1_KO_s112 <= 1
 0 <= w_k1_KO_s113 <= 1
 0 <= w_k1_KO_s114 <= 1
 0 <= w_k1_KO_s115 <= 1
 0 <= w_k1_KO_s116 <= 1
 0 <= w_k1_KO_s117 <= 1
 0 <= w_k1_MAT_s0 <= 1
 0 <= w_k1_MAT_s1 <= 1
 0 <= w_k1_MAT_s00 <= 1
 0 <= w_k1_MAT_s01 <= 1
 0 <= w_k1_MAT_s10 <= 1
 0 <= w_k1_MAT_s11 <= 1
 0 <= w_k1_MAT_s000 <= 1
 0 <= w_k1_MAT_s001 <= 1
 0 <= w_k1_MAT_s002 <= 1
 0 <= w_k1_MAT_s003 <= 1
 0 <= w_k1_MAT_s004 <= 1
 0 <= w_k1_MAT_s005 <= 1
 0 <= w_k1_MAT_s006 <= 1
 0 <= w_k1_MAT_s007 <= 1
 0 <= w_k1_MAT_s010 <= 1
 0 <= w_k1_MAT_s011 <= 1
 0 <= w_k1_MAT_s012 <= 1
 0 <= w_k1_MAT_s013 <= 1
 0 <= w_k1_MAT_s014 <= 1
 0 <= w_k1_MAT_s015 <= 1
 0 <= w_k1_MAT_s016 <= 1
 0 <= w_k1_MAT_s017 <= 1
 0 <= w_k1_MAT_s100 <= 1
 0 <= w_k1_MAT_s101 <= 1
 0 <= w_k1_MAT_s102 <= 1
 0 <= w_k1_MAT_s103 <= 1
 0 <= w_k1_MAT_s104 <= 1
 0 <= w_k1_MAT_s105 <= 1
 0 <= w_k1_MAT_s106 <= 1
 0 <= w_k1_MAT_s107 <= 1
 0 <= w_k1_MAT_s110 <= 1
 0 <= w_k1_MAT_s111 <= 1
 0 <= w_k1_MAT_s112 <= 1
 0 <= w_k1_MAT_s113 <= 1
 0 <= w_k1_MAT_s114 <= 1
 0 <= w_k1_MAT_s115 <= 1
 0 <= w_k1_MAT_s116 <= 1
 0 <= w_k1_MAT_s117 <= 1
 0 <= w_k1_MSFT_s0 <= 1
 0 <= w_k1_MSFT_s1 <= 1
 0 <= w_k1_MSFT_s00 <= 1
 0 <= w_k1_MSFT_s01 <= 1
 0 <= w_k1_MSFT_s10 <= 1
 0 <= w_k1_MSFT_s11 <= 1
 0 <= w_k1_MSFT_s000 <= 1
 0 <= w_k1_MSFT_s001 <= 1
 0 <= w_k1_MSFT_s002 <= 1
 0 <= w_k1_MSFT_s003 <= 1
 0 <= w_k1_MSFT_s004 <= 1
 0 <= w_k1_MSFT_s005 <= 1
 0 <= w_k1_MSFT_s006 <= 1
 0 <= w_k1_MSFT_s007 <= 1
 0 <= w_k1_MSFT_s010 <= 1
 0 <= w_k1_MSFT_s011 <= 1
 0 <= w_k1_MSFT_s012 <= 1
 0 <= w_k1_MSFT_s013 <= 1
 0 <= w_k1_MSFT_s014 <= 1
 0 <= w_k1_MSFT_s015 <= 1
 0 <= w_k1_MSFT_s016 <= 1
 0 <= w_k1_MSFT_s017 <= 1
 0 <= w_k1_MSFT_s100 <= 1
 0 <= w_k1_MSFT_s101 <= 1
 0 <= w_k1_MSFT_s102 <= 1
 0 <= w_k1_MSFT_s103 <= 1
 0 <= w_k1_MSFT_s104 <= 1
 0 <= w_k1_MSFT_s105 <= 1
 0 <= w_k1_MSFT_s106 <= 1
 0 <= w_k1_MSFT_s107 <= 1
 0 <= w_k1_MSFT_s110 <= 1
 0 <= w_k1_MSFT_s111 <= 1
 0 <= w_k1_MSFT_s112 <= 1
 0 <= w_k1_MSFT_s113 <= 1
 0 <= w_k1_MSFT_s114 <= 1
 0 <= w_k1_MSFT_s115 <= 1
 0 <= w_k1_MSFT_s116 <= 1
 0 <= w_k1_MSFT_s117 <= 1
 0 <= w_k1_PG_s0 <= 1
 0 <= w_k1_PG_s1 <= 1
 0 <= w_k1_PG_s00 <= 1
 0 <= w_k1_PG_s01 <= 1
 0 <= w_k1_PG_s10 <= 1
 0 <= w_k1_PG_s11 <= 1
 0 <= w_k1_PG_s000 <= 1
 0 <= w_k1_PG_s001 <= 1
 0 <= w_k1_PG_s002 <= 1
 0 <= w_k1_PG_s003 <= 1
 0 <= w_k1_PG_s004 <= 1
 0 <= w_k1_PG_s005 <= 1
 0 <= w_k1_PG_s006 <= 1
 0 <= w_k1_PG_s007 <= 1
 0 <= w_k1_PG_s010 <= 1
 0 <= w_k1_PG_s011 <= 1
 0 <= w_k1_PG_s012 <= 1
 0 <= w_k1_PG_s013 <= 1
 0 <= w_k1_PG_s014 <= 1
 0 <= w_k1_PG_s015 <= 1
 0 <= w_k1_PG_s016 <= 1
 0 <= w_k1_PG_s017 <= 1
 0 <= w_k1_PG_s100 <= 1
 0 <= w_k1_PG_s101 <= 1
 0 <= w_k1_PG_s102 <= 1
 0 <= w_k1_PG_s103 <= 1
 0 <= w_k1_PG_s104 <= 1
 0 <= w_k1_PG_s105 <= 1
 0 <= w_k1_PG_s106 <= 1
 0 <= w_k1_PG_s107 <= 1
 0 <= w_k1_PG_s110 <= 1
 0 <= w_k1_PG_s111 <= 1
 0 <= w_k1_PG_s112 <= 1
 0 <= w_k1_PG_s113 <= 1
 0 <= w_k1_PG_s114 <= 1
 0 <= w_k1_PG_s115 <= 1
 0 <= w_k1_PG_s116 <= 1
 0 <= w_k1_PG_s117 <= 1
 0 <= b_0_ADM <= 0.2
 0 <= b_0_AXP <= 0.2
 0 <= b_0_DIS <= 0.2
 0 <= b_0_FDX <= 0.2
 0 <= b_0_IBM <= 0.2
 0 <= b_0_INTC <= 0.2
 0 <= b_0_KO <= 0.2
 0 <= b_0_MAT <= 0.2
 0 <= b_0_MSFT <= 0.2
 0 <= b_0_PG <= 0.2
 0 <= b_k1_ADM_s0 <= 0.2
 0 <= b_k1_ADM_s1 <= 0.2
 0 <= b_k1_ADM_s00 <= 0.2
 0 <= b_k1_ADM_s01 <= 0.2
 0 <= b_k1_ADM_s10 <= 0.2
 0 <= b_k1_ADM_s11 <= 0.2
 0 <= b_k1_ADM_s000 <= 0.2
 0 <= b_k1_ADM_s001 <= 0.2
 0 <= b_k1_ADM_s002 <= 0.2
 0 <= b_k1_ADM_s003 <= 0.2
 0 <= b_k1_ADM_s004 <= 0.2
 0 <= b_k1_ADM_s005 <= 0.2
 0 <= b_k1_ADM_s006 <= 0.2
 0 <= b_k1_ADM_s007 <= 0.2
 0 <= b_k1_ADM_s010 <= 0.2
 0 <= b_k1_ADM_s011 <= 0.2
 0 <= b_k1_ADM_s012 <= 0.2
 0 <= b_k1_ADM_s013 <= 0.2
 0 <= b_k1_ADM_s014 <= 0.2
 0 <= b_k1_ADM_s015 <= 0.2
 0 <= b_k1_ADM_s016 <= 0.2
 0 <= b_k1_ADM_s017 <= 0.2
 0 <= b_k1_ADM_s100 <= 0.2
 0 <= b_k1_ADM_s101 <= 0.2
 0 <= b_k1_ADM_s102 <= 0.2
 0 <= b_k1_ADM_s103 <= 0.2
 0 <= b_k1_ADM_s104 <= 0.2
 0 <= b_k1_ADM_s105 <= 0.2
 0 <= b_k1_ADM_s106 <= 0.2
 0 <= b_k1_ADM_s107 <= 0.2
 0 <= b_k1_ADM_s110 <= 0.2
 0 <= b_k1_ADM_s111 <= 0.2
 0 <= b_k1_ADM_s112 <= 0.2
 0 <= b_k1_ADM_s113 <= 0.2
 0 <= b_k1_ADM_s114 <= 0.2
 0 <= b_k1_ADM_s115 <= 0.2
 0 <= b_k1_ADM_s116 <= 0.2
 0 <= b_k1_ADM_s117 <= 0.2
 0 <= b_k1_AXP_s0 <= 0.2
 0 <= b_k1_AXP_s1 <= 0.2
 0 <= b_k1_AXP_s00 <= 0.2
 0 <= b_k1_AXP_s01 <= 0.2
 0 <= b_k1_AXP_s10 <= 0.2
 0 <= b_k1_AXP_s11 <= 0.2
 0 <= b_k1_AXP_s000 <= 0.2
 0 <= b_k1_AXP_s001 <= 0.2
 0 <= b_k1_AXP_s002 <= 0.2
 0 <= b_k1_AXP_s003 <= 0.2
 0 <= b_k1_AXP_s004 <= 0.2
 0 <= b_k1_AXP_s005 <= 0.2
 0 <= b_k1_AXP_s006 <= 0.2
 0 <= b_k1_AXP_s007 <= 0.2
 0 <= b_k1_AXP_s010 <= 0.2
 0 <= b_k1_AXP_s011 <= 0.2
 0 <= b_k1_AXP_s012 <= 0.2
 0 <= b_k1_AXP_s013 <= 0.2
 0 <= b_k1_AXP_s014 <= 0.2
 0 <= b_k1_AXP_s015 <= 0.2
 0 <= b_k1_AXP_s016 <= 0.2
 0 <= b_k1_AXP_s017 <= 0.2
 0 <= b_k1_AXP_s100 <= 0.2
 0 <= b_k1_AXP_s101 <= 0.2
 0 <= b_k1_AXP_s102 <= 0.2
 0 <= b_k1_AXP_s103 <= 0.2
 0 <= b_k1_AXP_s104 <= 0.2
 0 <= b_k1_AXP_s105 <= 0.2
 0 <= b_k1_AXP_s106 <= 0.2
 0 <= b_k1_AXP_s107 <= 0.2
 0 <= b_k1_AXP_s110 <= 0.2
 0 <= b_k1_AXP_s111 <= 0.2
 0 <= b_k1_AXP_s112 <= 0.2
 0 <= b_k1_AXP_s113 <= 0.2
 0 <= b_k1_AXP_s114 <= 0.2
 0 <= b_k1_AXP_s115 <= 0.2
 0 <= b_k1_AXP_s116 <= 0.2
 0 <= b_k1_AXP_s117 <= 0.2
 0 <= b_k1_DIS_s0 <= 0.2
 0 <= b_k1_DIS_s1 <= 0.2
 0 <= b_k1_DIS_s00 <= 0.2
 0 <= b_k1_DIS_s01 <= 0.2
 0 <= b_k1_DIS_s10 <= 0.2
 0 <= b_k1_DIS_s11 <= 0.2
 0 <= b_k1_DIS_s000 <= 0.2
 0 <= b_k1_DIS_s001 <= 0.2
 0 <= b_k1_DIS_s002 <= 0.2
 0 <= b_k1_DIS_s003 <= 0.2
 0 <= b_k1_DIS_s004 <= 0.2
 0 <= b_k1_DIS_s005 <= 0.2
 0 <= b_k1_DIS_s006 <= 0.2
 0 <= b_k1_DIS_s007 <= 0.2
 0 <= b_k1_DIS_s010 <= 0.2
 0 <= b_k1_DIS_s011 <= 0.2
 0 <= b_k1_DIS_s012 <= 0.2
 0 <= b_k1_DIS_s013 <= 0.2
 0 <= b_k1_DIS_s014 <= 0.2
 0 <= b_k1_DIS_s015 <= 0.2
 0 <= b_k1_DIS_s016 <= 0.2
 0 <= b_k1_DIS_s017 <= 0.2
 0 <= b_k1_DIS_s100 <= 0.2
 0 <= b_k1_DIS_s101 <= 0.2
 0 <= b_k1_DIS_s102 <= 0.2
 0 <= b_k1_DIS_s103 <= 0.2
 0 <= b_k1_DIS_s104 <= 0.2
 0 <= b_k1_DIS_s105 <= 0.2
 0 <= b_k1_DIS_s106 <= 0.2
 0 <= b_k1_DIS_s107 <= 0.2
 0 <= b_k1_DIS_s110 <= 0.2
 0 <= b_k1_DIS_s111 <= 0.2
 0 <= b_k1_DIS_s112 <= 0.2
 0 <= b_k1_DIS_s113 <= 0.2
 0 <= b_k1_DIS_s114 <= 0.2
 0 <= b_k1_DIS_s115 <= 0.2
 0 <= b_k1_DIS_s116 <= 0.2
 0 <= b_k1_DIS_s117 <= 0.2
 0 <= b_k1_FDX_s0 <= 0.2
 0 <= b_k1_FDX_s1 <= 0.2
 0 <= b_k1_FDX_s00 <= 0.2
 0 <= b_k1_FDX_s01 <= 0.2
 0 <= b_k1_FDX_s10 <= 0.2
 0 <= b_k1_FDX_s11 <= 0.2
 0 <= b_k1_FDX_s000 <= 0.2
 0 <= b_k1_FDX_s001 <= 0.2
 0 <= b_k1_FDX_s002 <= 0.2
 0 <= b_k1_FDX_s003 <= 0.2
 0 <= b_k1_FDX_s004 <= 0.2
 0 <= b_k1_FDX_s005 <= 0.2
 0 <= b_k1_FDX_s006 <= 0.2
 0 <= b_k1_FDX_s007 <= 0.2
 0 <= b_k1_FDX_s010 <= 0.2
 0 <= b_k1_FDX_s011 <= 0.2
 0 <= b_k1_FDX_s012 <= 0.2
 0 <= b_k1_FDX_s013 <= 0.2
 0 <= b_k1_FDX_s014 <= 0.2
 0 <= b_k1_FDX_s015 <= 0.2
 0 <= b_k1_FDX_s016 <= 0.2
 0 <= b_k1_FDX_s017 <= 0.2
 0 <= b_k1_FDX_s100 <= 0.2
 0 <= b_k1_FDX_s101 <= 0.2
 0 <= b_k1_FDX_s102 <= 0.2
 0 <= b_k1_FDX_s103 <= 0.2
 0 <= b_k1_FDX_s104 <= 0.2
 0 <= b_k1_FDX_s105 <= 0.2
 0 <= b_k1_FDX_s106 <= 0.2
 0 <= b_k1_FDX_s107 <= 0.2
 0 <= b_k1_FDX_s110 <= 0.2
 0 <= b_k1_FDX_s111 <= 0.2
 0 <= b_k1_FDX_s112 <= 0.2
 0 <= b_k1_FDX_s113 <= 0.2
 0 <= b_k1_FDX_s114 <= 0.2
 0 <= b_k1_FDX_s115 <= 0.2
 0 <= b_k1_FDX_s116 <= 0.2
 0 <= b_k1_FDX_s117 <= 0.2
 0 <= b_k1_IBM_s0 <= 0.2
 0 <= b_k1_IBM_s1 <= 0.2
 0 <= b_k1_IBM_s00 <= 0.2
 0 <= b_k1_IBM_s01 <= 0.2
 0 <= b_k1_IBM_s10 <= 0.2
 0 <= b_k1_IBM_s11 <= 0.2
 0 <= b_k1_IBM_s000 <= 0.2
 0 <= b_k1_IBM_s001 <= 0.2
 0 <= b_k1_IBM_s002 <= 0.2
 0 <= b_k1_IBM_s003 <= 0.2
 0 <= b_k1_IBM_s004 <= 0.2
 0 <= b_k1_IBM_s005 <= 0.2
 0 <= b_k1_IBM_s006 <= 0.2
 0 <= b_k1_IBM_s007 <= 0.2
 0 <= b_k1_IBM_s010 <= 0.2
 0 <= b_k1_IBM_s011 <= 0.2
 0 <= b_k1_IBM_s012 <= 0.2
 0 <= b_k1_IBM_s013 <= 0.2
 0 <= b_k1_IBM_s014 <= 0.2
 0 <= b_k1_IBM_s015 <= 0.2
 0 <= b_k1_IBM_s016 <= 0.2
 0 <= b_k1_IBM_s017 <= 0.2
 0 <= b_k1_IBM_s100 <= 0.2
 0 <= b_k1_IBM_s101 <= 0.2
 0 <= b_k1_IBM_s102 <= 0.2
 0 <= b_k1_IBM_s103 <= 0.2
 0 <= b_k1_IBM_s104 <= 0.2
 0 <= b_k1_IBM_s105 <= 0.2
 0 <= b_k1_IBM_s106 <= 0.2
 0 <= b_k1_IBM_s107 <= 0.2
 0 <= b_k1_IBM_s110 <= 0.2
 0 <= b_k1_IBM_s111 <= 0.2
 0 <= b_k1_IBM_s112 <= 0.2
 0 <= b_k1_IBM_s113 <= 0.2
 0 <= b_k1_IBM_s114 <= 0.2
 0 <= b_k1_IBM_s115 <= 0.2
 0 <= b_k1_IBM_s116 <= 0.2
 0 <= b_k1_IBM_s117 <= 0.2
 0 <= b_k1_INTC_s0 <= 0.2
 0 <= b_k1_INTC_s1 <= 0.2
 0 <= b_k1_INTC_s00 <= 0.2
 0 <= b_k1_INTC_s01 <= 0.2
 0 <= b_k1_INTC_s10 <= 0.2
 0 <= b_k1_INTC_s11 <= 0.2
 0 <= b_k1_INTC_s000 <= 0.2
 0 <= b_k1_INTC_s001 <= 0.2
 0 <= b_k1_INTC_s002 <= 0.2
 0 <= b_k1_INTC_s003 <= 0.2
 0 <= b_k1_INTC_s004 <= 0.2
 0 <= b_k1_INTC_s005 <= 0.2
 0 <= b_k1_INTC_s006 <= 0.2
 0 <= b_k1_INTC_s007 <= 0.2
 0 <= b_k1_INTC_s010 <= 0.2
 0 <= b_k1_INTC_s011 <= 0.2
 0 <= b_k1_INTC_s012 <= 0.2
 0 <= b_k1_INTC_s013 <= 0.2
 0 <= b_k1_INTC_s014 <= 0.2
 0 <= b_k1_INTC_s015 <= 0.2
 0 <= b_k1_INTC_s016 <= 0.2
 0 <= b_k1_INTC_s017 <= 0.2
 0 <= b_k1_INTC_s100 <= 0.2
 0 <= b_k1_INTC_s101 <= 0.2
 0 <= b_k1_INTC_s102 <= 0.2
 0 <= b_k1_INTC_s103 <= 0.2
 0 <= b_k1_INTC_s104 <= 0.2
 0 <= b_k1_INTC_s105 <= 0.2
 0 <= b_k1_INTC_s106 <= 0.2
 0 <= b_k1_INTC_s107 <= 0.2
 0 <= b_k1_INTC_s110 <= 0.2
 0 <= b_k1_INTC_s111 <= 0.2
 0 <= b_k1_INTC_s112 <= 0.2
 0 <= b_k1_INTC_s113 <= 0.2
 0 <= b_k1_INTC_s114 <= 0.2
 0 <= b_k1_INTC_s115 <= 0.2
 0 <= b_k1_INTC_s116 <= 0.2
 0 <= b_k1_INTC_s117 <= 0.2
 0 <= b_k1_KO_s0 <= 0.2
 0 <= b_k1_KO_s1 <= 0.2
 0 <= b_k1_KO_s00 <= 0.2
 0 <= b_k1_KO_s01 <= 0.2
 0 <= b_k1_KO_s10 <= 0.2
 0 <= b_k1_KO_s11 <= 0.2
 0 <= b_k1_KO_s000 <= 0.2
 0 <= b_k1_KO_s001 <= 0.2
 0 <= b_k1_KO_s002 <= 0.2
 0 <= b_k1_KO_s003 <= 0.2
 0 <= b_k1_KO_s004 <= 0.2
 0 <= b_k1_KO_s005 <= 0.2
 0 <= b_k1_KO_s006 <= 0.2
 0 <= b_k1_KO_s007 <= 0.2
 0 <= b_k1_KO_s010 <= 0.2
 0 <= b_k1_KO_s011 <= 0.2
 0 <= b_k1_KO_s012 <= 0.2
 0 <= b_k1_KO_s013 <= 0.2
 0 <= b_k1_KO_s014 <= 0.2
 0 <= b_k1_KO_s015 <= 0.2
 0 <= b_k1_KO_s016 <= 0.2
 0 <= b_k1_KO_s017 <= 0.2
 0 <= b_k1_KO_s100 <= 0.2
 0 <= b_k1_KO_s101 <= 0.2
 0 <= b_k1_KO_s102 <= 0.2
 0 <= b_k1_KO_s103 <= 0.2
 0 <= b_k1_KO_s104 <= 0.2
 0 <= b_k1_KO_s105 <= 0.2
 0 <= b_k1_KO_s106 <= 0.2
 0 <= b_k1_KO_s107 <= 0.2
 0 <= b_k1_KO_s110 <= 0.2
 0 <= b_k1_KO_s111 <= 0.2
 0 <= b_k1_KO_s112 <= 0.2
 0 <= b_k1_KO_s113 <= 0.2
 0 <= b_k1_KO_s114 <= 0.2
 0 <= b_k1_KO_s115 <= 0.2
 0 <= b_k1_KO_s116 <= 0.2
 0 <= b_k1_KO_s117 <= 0.2
 0 <= b_k1_MAT_s0 <= 0.2
 0 <= b_k1_MAT_s1 <= 0.2
 0 <= b_k1_MAT_s00 <= 0.2
 0 <= b_k1_MAT_s01 <= 0.2
 0 <= b_k1_MAT_s10 <= 0.2
 0 <= b_k1_MAT_s11 <= 0.2
 0 <= b_k1_MAT_s000 <= 0.2
 0 <= b_k1_MAT_s001 <= 0.2
 0 <= b_k1_MAT_s002 <= 0.2
 0 <= b_k1_MAT_s003 <= 0.2
 0 <= b_k1_MAT_s004 <= 0.2
 0 <= b_k1_MAT_s005 <= 0.2
 0 <= b_k1_MAT_s006 <= 0.2
 0 <= b_k1_MAT_s007 <= 0.2
 0 <= b_k1_MAT_s010 <= 0.2
 0 <= b_k1_MAT_s011 <= 0.2
 0 <= b_k1_MAT_s012 <= 0.2
 0 <= b_k1_MAT_s013 <= 0.2
 0 <= b_k1_MAT_s014 <= 0.2
 0 <= b_k1_MAT_s015 <= 0.2
 0 <= b_k1_MAT_s016 <= 0.2
 0 <= b_k1_MAT_s017 <= 0.2
 0 <= b_k1_MAT_s100 <= 0.2
 0 <= b_k1_MAT_s101 <= 0.2
 0 <= b_k1_MAT_s102 <= 0.2
 0 <= b_k1_MAT_s103 <= 0.2
 0 <= b_k1_MAT_s104 <= 0.2
 0 <= b_k1_MAT_s105 <= 0.2
 0 <= b_k1_MAT_s106 <= 0.2
 0 <= b_k1_MAT_s107 <= 0.2
 0 <= b_k1_MAT_s110 <= 0.2
 0 <= b_k1_MAT_s111 <= 0.2
 0 <= b_k1_MAT_s112 <= 0.2
 0 <= b_k1_MAT_s113 <= 0.2
 0 <= b_k1_MAT_s114 <= 0.2
 0 <= b_k1_MAT_s115 <= 0.2
 0 <= b_k1_MAT_s116 <= 0.2
 0 <= b_k1_MAT_s117 <= 0.2
 0 <= b_k1_MSFT_s0 <= 0.2
 0 <= b_k1_MSFT_s1 <= 0.2
 0 <= b_k1_MSFT_s00 <= 0.2
 0 <= b_k1_MSFT_s01 <= 0.2
 0 <= b_k1_MSFT_s10 <= 0.2
 0 <= b_k1_MSFT_s11 <= 0.2
 0 <= b_k1_MSFT_s000 <= 0.2
 0 <= b_k1_MSFT_s001 <= 0.2
 0 <= b_k1_MSFT_s002 <= 0.2
 0 <= b_k1_MSFT_s003 <= 0.2
 0 <= b_k1_MSFT_s004 <= 0.2
 0 <= b_k1_MSFT_s005 <= 0.2
 0 <= b_k1_MSFT_s006 <= 0.2
 0 <= b_k1_MSFT_s007 <= 0.2
 0 <= b_k1_MSFT_s010 <= 0.2
 0 <= b_k1_MSFT_s011 <= 0.2
 0 <= b_k1_MSFT_s012 <= 0.2
 0 <= b_k1_MSFT_s013 <= 0.2
 0 <= b_k1_MSFT_s014 <= 0.2
 0 <= b_k1_MSFT_s015 <= 0.2
 0 <= b_k1_MSFT_s016 <= 0.2
 0 <= b_k1_MSFT_s017 <= 0.2
 0 <= b_k1_MSFT_s100 <= 0.2
 0 <= b_k1_MSFT_s101 <= 0.2
 0 <= b_k1_MSFT_s102 <= 0.2
 0 <= b_k1_MSFT_s103 <= 0.2
 0 <= b_k1_MSFT_s104 <= 0.2
 0 <= b_k1_MSFT_s105 <= 0.2
 0 <= b_k1_MSFT_s106 <= 0.2
 0 <= b_k1_MSFT_s107 <= 0.2
 0 <= b_k1_MSFT_s110 <= 0.2
 0 <= b_k1_MSFT_s111 <= 0.2
 0 <= b_k1_MSFT_s112 <= 0.2
 0 <= b_k1_MSFT_s113 <= 0.2
 0 <= b_k1_MSFT_s114 <= 0.2
 0 <= b_k1_MSFT_s115 <= 0.2
 0 <= b_k1_MSFT_s116 <= 0.2
 0 <= b_k1_MSFT_s117 <= 0.2
 0 <= b_k1_PG_s0 <= 0.2
 0 <= b_k1_PG_s1 <= 0.2
 0 <= b_k1_PG_s00 <= 0.2
 0 <= b_k1_PG_s01 <= 0.2
 0 <= b_k1_PG_s10 <= 0.2
 0 <= b_k1_PG_s11 <= 0.2
 0 <= b_k1_PG_s000 <= 0.2
 0 <= b_k1_PG_s001 <= 0.2
 0 <= b_k1_PG_s002 <= 0.2
 0 <= b_k1_PG_s003 <= 0.2
 0 <= b_k1_PG_s004 <= 0.2
 0 <= b_k1_PG_s005 <= 0.2
 0 <= b_k1_PG_s006 <= 0.2
 0 <= b_k1_PG_s007 <= 0.2
 0 <= b_k1_PG_s010 <= 0.2
 0 <= b_k1_PG_s011 <= 0.2
 0 <= b_k1_PG_s012 <= 0.2
 0 <= b_k1_PG_s013 <= 0.2
 0 <= b_k1_PG_s014 <= 0.2
 0 <= b_k1_PG_s015 <= 0.2
 0 <= b_k1_PG_s016 <= 0.2
 0 <= b_k1_PG_s017 <= 0.2
 0 <= b_k1_PG_s100 <= 0.2
 0 <= b_k1_PG_s101 <= 0.2
 0 <= b_k1_PG_s102 <= 0.2
 0 <= b_k1_PG_s103 <= 0.2
 0 <= b_k1_PG_s104 <= 0.2
 0 <= b_k1_PG_s105 <= 0.2
 0 <= b_k1_PG_s106 <= 0.2
 0 <= b_k1_PG_s107 <= 0.2
 0 <= b_k1_PG_s110 <= 0.2
 0 <= b_k1_PG_s111 <= 0.2
 0 <= b_k1_PG_s112 <= 0.2
 0 <= b_k1_PG_s113 <= 0.2
 0 <= b_k1_PG_s114 <= 0.2
 0 <= b_k1_PG_s115 <= 0.2
 0 <= b_k1_PG_s116 <= 0.2
 0 <= b_k1_PG_s117 <= 0.2
 0 <= s_0_ADM <= 0.2
 0 <= s_0_AXP <= 0.2
 0 <= s_0_DIS <= 0.2
 0 <= s_0_FDX <= 0.2
 0 <= s_0_IBM <= 0.2
 0 <= s_0_INTC <= 0.2
 0 <= s_0_KO <= 0.2
 0 <= s_0_MAT <= 0.2
 0 <= s_0_MSFT <= 0.2
 0 <= s_0_PG <= 0.2
 0 <= s_k1_ADM_s0 <= 0.2
 0 <= s_k1_ADM_s1 <= 0.2
 0 <= s_k1_ADM_s00 <= 0.2
 0 <= s_k1_ADM_s01 <= 0.2
 0 <= s_k1_ADM_s10 <= 0.2
 0 <= s_k1_ADM_s11 <= 0.2
 0 <= s_k1_ADM_s000 <= 0.2
 0 <= s_k1_ADM_s001 <= 0.2
 0 <= s_k1_ADM_s002 <= 0.2
 0 <= s_k1_ADM_s003 <= 0.2
 0 <= s_k1_ADM_s004 <= 0.2
 0 <= s_k1_ADM_s005 <= 0.2
 0 <= s_k1_ADM_s006 <= 0.2
 0 <= s_k1_ADM_s007 <= 0.2
 0 <= s_k1_ADM_s010 <= 0.2
 0 <= s_k1_ADM_s011 <= 0.2
 0 <= s_k1_ADM_s012 <= 0.2
 0 <= s_k1_ADM_s013 <= 0.2
 0 <= s_k1_ADM_s014 <= 0.2
 0 <= s_k1_ADM_s015 <= 0.2
 0 <= s_k1_ADM_s016 <= 0.2
 0 <= s_k1_ADM_s017 <= 0.2
 0 <= s_k1_ADM_s100 <= 0.2
 0 <= s_k1_ADM_s101 <= 0.2
 0 <= s_k1_ADM_s102 <= 0.2
 0 <= s_k1_ADM_s103 <= 0.2
 0 <= s_k1_ADM_s104 <= 0.2
 0 <= s_k1_ADM_s105 <= 0.2
 0 <= s_k1_ADM_s106 <= 0.2
 0 <= s_k1_ADM_s107 <= 0.2
 0 <= s_k1_ADM_s110 <= 0.2
 0 <= s_k1_ADM_s111 <= 0.2
 0 <= s_k1_ADM_s112 <= 0.2
 0 <= s_k1_ADM_s113 <= 0.2
 0 <= s_k1_ADM_s114 <= 0.2
 0 <= s_k1_ADM_s115 <= 0.2
 0 <= s_k1_ADM_s116 <= 0.2
 0 <= s_k1_ADM_s117 <= 0.2
 0 <= s_k1_AXP_s0 <= 0.2
 0 <= s_k1_AXP_s1 <= 0.2
 0 <= s_k1_AXP_s00 <= 0.2
 0 <= s_k1_AXP_s01 <= 0.2
 0 <= s_k1_AXP_s10 <= 0.2
 0 <= s_k1_AXP_s11 <= 0.2
 0 <= s_k1_AXP_s000 <= 0.2
 0 <= s_k1_AXP_s001 <= 0.2
 0 <= s_k1_AXP_s002 <= 0.2
 0 <= s_k1_AXP_s003 <= 0.2
 0 <= s_k1_AXP_s004 <= 0.2
 0 <= s_k1_AXP_s005 <= 0.2
 0 <= s_k1_AXP_s006 <= 0.2
 0 <= s_k1_AXP_s007 <= 0.2
 0 <= s_k1_AXP_s010 <= 0.2
 0 <= s_k1_AXP_s011 <= 0.2
 0 <= s_k1_AXP_s012 <= 0.2
 0 <= s_k1_AXP_s013 <= 0.2
 0 <= s_k1_AXP_s014 <= 0.2
 0 <= s_k1_AXP_s015 <= 0.2
 0 <= s_k1_AXP_s016 <= 0.2
 0 <= s_k1_AXP_s017 <= 0.2
 0 <= s_k1_AXP_s100 <= 0.2
 0 <= s_k1_AXP_s101 <= 0.2
 0 <= s_k1_AXP_s102 <= 0.2
 0 <= s_k1_AXP_s103 <= 0.2
 0 <= s_k1_AXP_s104 <= 0.2
 0 <= s_k1_AXP_s105 <= 0.2
 0 <= s_k1_AXP_s106 <= 0.2
 0 <= s_k1_AXP_s107 <= 0.2
 0 <= s_k1_AXP_s110 <= 0.2
 0 <= s_k1_AXP_s111 <= 0.2
 0 <= s_k1_AXP_s112 <= 0.2
 0 <= s_k1_AXP_s113 <= 0.2
 0 <= s_k1_AXP_s114 <= 0.2
 0 <= s_k1_AXP_s115 <= 0.2
 0 <= s_k1_AXP_s116 <= 0.2
 0 <= s_k1_AXP_s117 <= 0.2
 0 <= s_k1_DIS_s0 <= 0.2
 0 <= s_k1_DIS_s1 <= 0.2
 0 <= s_k1_DIS_s00 <= 0.2
 0 <= s_k1_DIS_s01 <= 0.2
 0 <= s_k1_DIS_s10 <= 0.2
 0 <= s_k1_DIS_s11 <= 0.2
 0 <= s_k1_DIS_s000 <= 0.2
 0 <= s_k1_DIS_s001 <= 0.2
 0 <= s_k1_DIS_s002 <= 0.2
 0 <= s_k1_DIS_s003 <= 0.2
 0 <= s_k1_DIS_s004 <= 0.2
 0 <= s_k1_DIS_s005 <= 0.2
 0 <= s_k1_DIS_s006 <= 0.2
 0 <= s_k1_DIS_s007 <= 0.2
 0 <= s_k1_DIS_s010 <= 0.2
 0 <= s_k1_DIS_s011 <= 0.2
 0 <= s_k1_DIS_s012 <= 0.2
 0 <= s_k1_DIS_s013 <= 0.2
 0 <= s_k1_DIS_s014 <= 0.2
 0 <= s_k1_DIS_s015 <= 0.2
 0 <= s_k1_DIS_s016 <= 0.2
 0 <= s_k1_DIS_s017 <= 0.2
 0 <= s_k1_DIS_s100 <= 0.2
 0 <= s_k1_DIS_s101 <= 0.2
 0 <= s_k1_DIS_s102 <= 0.2
 0 <= s_k1_DIS_s103 <= 0.2
 0 <= s_k1_DIS_s104 <= 0.2
 0 <= s_k1_DIS_s105 <= 0.2
 0 <= s_k1_DIS_s106 <= 0.2
 0 <= s_k1_DIS_s107 <= 0.2
 0 <= s_k1_DIS_s110 <= 0.2
 0 <= s_k1_DIS_s111 <= 0.2
 0 <= s_k1_DIS_s112 <= 0.2
 0 <= s_k1_DIS_s113 <= 0.2
 0 <= s_k1_DIS_s114 <= 0.2
 0 <= s_k1_DIS_s115 <= 0.2
 0 <= s_k1_DIS_s116 <= 0.2
 0 <= s_k1_DIS_s117 <= 0.2
 0 <= s_k1_FDX_s0 <= 0.2
 0 <= s_k1_FDX_s1 <= 0.2
 0 <= s_k1_FDX_s00 <= 0.2
 0 <= s_k1_FDX_s01 <= 0.2
 0 <= s_k1_FDX_s10 <= 0.2
 0 <= s_k1_FDX_s11 <= 0.2
 0 <= s_k1_FDX_s000 <= 0.2
 0 <= s_k1_FDX_s001 <= 0.2
 0 <= s_k1_FDX_s002 <= 0.2
 0 <= s_k1_FDX_s003 <= 0.2
 0 <= s_k1_FDX_s004 <= 0.2
 0 <= s_k1_FDX_s005 <= 0.2
 0 <= s_k1_FDX_s006 <= 0.2
 0 <= s_k1_FDX_s007 <= 0.2
 0 <= s_k1_FDX_s010 <= 0.2
 0 <= s_k1_FDX_s011 <= 0.2
 0 <= s_k1_FDX_s012 <= 0.2
 0 <= s_k1_FDX_s013 <= 0.2
 0 <= s_k1_FDX_s014 <= 0.2
 0 <= s_k1_FDX_s015 <= 0.2
 0 <= s_k1_FDX_s016 <= 0.2
 0 <= s_k1_FDX_s017 <= 0.2
 0 <= s_k1_FDX_s100 <= 0.2
 0 <= s_k1_FDX_s101 <= 0.2
 0 <= s_k1_FDX_s102 <= 0.2
 0 <= s_k1_FDX_s103 <= 0.2
 0 <= s_k1_FDX_s104 <= 0.2
 0 <= s_k1_FDX_s105 <= 0.2
 0 <= s_k1_FDX_s106 <= 0.2
 0 <= s_k1_FDX_s107 <= 0.2
 0 <= s_k1_FDX_s110 <= 0.2
 0 <= s_k1_FDX_s111 <= 0.2
 0 <= s_k1_FDX_s112 <= 0.2
 0 <= s_k1_FDX_s113 <= 0.2
 0 <= s_k1_FDX_s114 <= 0.2
 0 <= s_k1_FDX_s115 <= 0.2
 0 <= s_k1_FDX_s116 <= 0.2
 0 <= s_k1_FDX_s117 <= 0.2
 0 <= s_k1_IBM_s0 <= 0.2
 0 <= s_k1_IBM_s1 <= 0.2
 0 <= s_k1_IBM_s00 <= 0.2
 0 <= s_k1_IBM_s01 <= 0.2
 0 <= s_k1_IBM_s10 <= 0.2
 0 <= s_k1_IBM_s11 <= 0.2
 0 <= s_k1_IBM_s000 <= 0.2
 0 <= s_k1_IBM_s001 <= 0.2
 0 <= s_k1_IBM_s002 <= 0.2
 0 <= s_k1_IBM_s003 <= 0.2
 0 <= s_k1_IBM_s004 <= 0.2
 0 <= s_k1_IBM_s005 <= 0.2
 0 <= s_k1_IBM_s006 <= 0.2
 0 <= s_k1_IBM_s007 <= 0.2
 0 <= s_k1_IBM_s010 <= 0.2
 0 <= s_k1_IBM_s011 <= 0.2
 0 <= s_k1_IBM_s012 <= 0.2
 0 <= s_k1_IBM_s013 <= 0.2
 0 <= s_k1_IBM_s014 <= 0.2
 0 <= s_k1_IBM_s015 <= 0.2
 0 <= s_k1_IBM_s016 <= 0.2
 0 <= s_k1_IBM_s017 <= 0.2
 0 <= s_k1_IBM_s100 <= 0.2
 0 <= s_k1_IBM_s101 <= 0.2
 0 <= s_k1_IBM_s102 <= 0.2
 0 <= s_k1_IBM_s103 <= 0.2
 0 <= s_k1_IBM_s104 <= 0.2
 0 <= s_k1_IBM_s105 <= 0.2
 0 <= s_k1_IBM_s106 <= 0.2
 0 <= s_k1_IBM_s107 <= 0.2
 0 <= s_k1_IBM_s110 <= 0.2
 0 <= s_k1_IBM_s111 <= 0.2
 0 <= s_k1_IBM_s112 <= 0.2
 0 <= s_k1_IBM_s113 <= 0.2
 0 <= s_k1_IBM_s114 <= 0.2
 0 <= s_k1_IBM_s115 <= 0.2
 0 <= s_k1_IBM_s116 <= 0.2
 0 <= s_k1_IBM_s117 <= 0.2
 0 <= s_k1_INTC_s0 <= 0.2
 0 <= s_k1_INTC_s1 <= 0.2
 0 <= s_k1_INTC_s00 <= 0.2
 0 <= s_k1_INTC_s01 <= 0.2
 0 <= s_k1_INTC_s10 <= 0.2
 0 <= s_k1_INTC_s11 <= 0.2
 0 <= s_k1_INTC_s000 <= 0.2
 0 <= s_k1_INTC_s001 <= 0.2
 0 <= s_k1_INTC_s002 <= 0.2
 0 <= s_k1_INTC_s003 <= 0.2
 0 <= s_k1_INTC_s004 <= 0.2
 0 <= s_k1_INTC_s005 <= 0.2
 0 <= s_k1_INTC_s006 <= 0.2
 0 <= s_k1_INTC_s007 <= 0.2
 0 <= s_k1_INTC_s010 <= 0.2
 0 <= s_k1_INTC_s011 <= 0.2
 0 <= s_k1_INTC_s012 <= 0.2
 0 <= s_k1_INTC_s013 <= 0.2
 0 <= s_k1_INTC_s014 <= 0.2
 0 <= s_k1_INTC_s015 <= 0.2
 0 <= s_k1_INTC_s016 <= 0.2
 0 <= s_k1_INTC_s017 <= 0.2
 0 <= s_k1_INTC_s100 <= 0.2
 0 <= s_k1_INTC_s101 <= 0.2
 0 <= s_k1_INTC_s102 <= 0.2
 0 <= s_k1_INTC_s103 <= 0.2
 0 <= s_k1_INTC_s104 <= 0.2
 0 <= s_k1_INTC_s105 <= 0.2
 0 <= s_k1_INTC_s106 <= 0.2
 0 <= s_k1_INTC_s107 <= 0.2
 0 <= s_k1_INTC_s110 <= 0.2
 0 <= s_k1_INTC_s111 <= 0.2
 0 <= s_k1_INTC_s112 <= 0.2
 0 <= s_k1_INTC_s113 <= 0.2
 0 <= s_k1_INTC_s114 <= 0.2
 0 <= s_k1_INTC_s115 <= 0.2
 0 <= s_k1_INTC_s116 <= 0.2
 0 <= s_k1_INTC_s117 <= 0.2
 0 <= s_k1_KO_s0 <= 0.2
 0 <= s_k1_KO_s1 <= 0.2
 0 <= s_k1_KO_s00 <= 0.2
 0 <= s_k1_KO_s01 <= 0.2
 0 <= s_k1_KO_s10 <= 0.2
 0 <= s_k1_KO_s11 <= 0.2
 0 <= s_k1_KO_s000 <= 0.2
 0 <= s_k1_KO_s001 <= 0.2
 0 <= s_k1_KO_s002 <= 0.2
 0 <= s_k1_KO_s003 <= 0.2
 0 <= s_k1_KO_s004 <= 0.2
 0 <= s_k1_KO_s005 <= 0.2
 0 <= s_k1_KO_s006 <= 0.2
 0 <= s_k1_KO_s007 <= 0.2
 0 <= s_k1_KO_s010 <= 0.2
 0 <= s_k1_KO_s011 <= 0.2
 0 <= s_k1_KO_s012 <= 0.2
 0 <= s_k1_KO_s013 <= 0.2
 0 <= s_k1_KO_s014 <= 0.2
 0 <= s_k1_KO_s015 <= 0.2
 0 <= s_k1_KO_s016 <= 0.2
 0 <= s_k1_KO_s017 <= 0.2
 0 <= s_k1_KO_s100 <= 0.2
 0 <= s_k1_KO_s101 <= 0.2
 0 <= s_k1_KO_s102 <= 0.2
 0 <= s_k1_KO_s103 <= 0.2
 0 <= s_k1_KO_s104 <= 0.2
 0 <= s_k1_KO_s105 <= 0.2
 0 <= s_k1_KO_s106 <= 0.2
 0 <= s_k1_KO_s107 <= 0.2
 0 <= s_k1_KO_s110 <= 0.2
 0 <= s_k1_KO_s111 <= 0.2
 0 <= s_k1_KO_s112 <= 0.2
 0 <= s_k1_KO_s113 <= 0.2
 0 <= s_k1_KO_s114 <= 0.2
 0 <= s_k1_KO_s115 <= 0.2
 0 <= s_k1_KO_s116 <= 0.2
 0 <= s_k1_KO_s117 <= 0.2
 0 <= s_k1_MAT_s0 <= 0.2
 0 <= s_k1_MAT_s1 <= 0.2
 0 <= s_k1_MAT_s00 <= 0.2
 0 <= s_k1_MAT_s01 <= 0.2
 0 <= s_k1_MAT_s10 <= 0.2
 0 <= s_k1_MAT_s11 <= 0.2
 0 <= s_k1_MAT_s000 <= 0.2
 0 <= s_k1_MAT_s001 <= 0.2
 0 <= s_k1_MAT_s002 <= 0.2
 0 <= s_k1_MAT_s003 <= 0.2
 0 <= s_k1_MAT_s004 <= 0.2
 0 <= s_k1_MAT_s005 <= 0.2
 0 <= s_k1_MAT_s006 <= 0.2
 0 <= s_k1_MAT_s007 <= 0.2
 0 <= s_k1_MAT_s010 <= 0.2
 0 <= s_k1_MAT_s011 <= 0.2
 0 <= s_k1_MAT_s012 <= 0.2
 0 <= s_k1_MAT_s013 <= 0.2
 0 <= s_k1_MAT_s014 <= 0.2
 0 <= s_k1_MAT_s015 <= 0.2
 0 <= s_k1_MAT_s016 <= 0.2
 0 <= s_k1_MAT_s017 <= 0.2
 0 <= s_k1_MAT_s100 <= 0.2
 0 <= s_k1_MAT_s101 <= 0.2
 0 <= s_k1_MAT_s102 <= 0.2
 0 <= s_k1_MAT_s103 <= 0.2
 0 <= s_k1_MAT_s104 <= 0.2
 0 <= s_k1_MAT_s105 <= 0.2
 0 <= s_k1_MAT_s106 <= 0.2
 0 <= s_k1_MAT_s107 <= 0.2
 0 <= s_k1_MAT_s110 <= 0.2
 0 <= s_k1_MAT_s111 <= 0.2
 0 <= s_k1_MAT_s112 <= 0.2
 0 <= s_k1_MAT_s113 <= 0.2
 0 <= s_k1_MAT_s114 <= 0.2
 0 <= s_k1_MAT_s115 <= 0.2
 0 <= s_k1_MAT_s116 <= 0.2
 0 <= s_k1_MAT_s117 <= 0.2
 0 <= s_k1_MSFT_s0 <= 0.2
 0 <= s_k1_MSFT_s1 <= 0.2
 0 <= s_k1_MSFT_s00 <= 0.2
 0 <= s_k1_MSFT_s01 <= 0.2
 0 <= s_k1_MSFT_s10 <= 0.2
 0 <= s_k1_MSFT_s11 <= 0.2
 0 <= s_k1_MSFT_s000 <= 0.2
 0 <= s_k1_MSFT_s001 <= 0.2
 0 <= s_k1_MSFT_s002 <= 0.2
 0 <= s_k1_MSFT_s003 <= 0.2
 0 <= s_k1_MSFT_s004 <= 0.2
 0 <= s_k1_MSFT_s005 <= 0.2
 0 <= s_k1_MSFT_s006 <= 0.2
 0 <= s_k1_MSFT_s007 <= 0.2
 0 <= s_k1_MSFT_s010 <= 0.2
 0 <= s_k1_MSFT_s011 <= 0.2
 0 <= s_k1_MSFT_s012 <= 0.2
 0 <= s_k1_MSFT_s013 <= 0.2
 0 <= s_k1_MSFT_s014 <= 0.2
 0 <= s_k1_MSFT_s015 <= 0.2
 0 <= s_k1_MSFT_s016 <= 0.2
 0 <= s_k1_MSFT_s017 <= 0.2
 0 <= s_k1_MSFT_s100 <= 0.2
 0 <= s_k1_MSFT_s101 <= 0.2
 0 <= s_k1_MSFT_s102 <= 0.2
 0 <= s_k1_MSFT_s103 <= 0.2
 0 <= s_k1_MSFT_s104 <= 0.2
 0 <= s_k1_MSFT_s105 <= 0.2
 0 <= s_k1_MSFT_s106 <= 0.2
 0 <= s_k1_MSFT_s107 <= 0.2
 0 <= s_k1_MSFT_s110 <= 0.2
 0 <= s_k1_MSFT_s111 <= 0.2
 0 <= s_k1_MSFT_s112 <= 0.2
 0 <= s_k1_MSFT_s113 <= 0.2
 0 <= s_k1_MSFT_s114 <= 0.2
 0 <= s_k1_MSFT_s115 <= 0.2
 0 <= s_k1_MSFT_s116 <= 0.2
 0 <= s_k1_MSFT_s117 <= 0.2
 0 <= s_k1_PG_s0 <= 0.2
 0 <= s_k1_PG_s1 <= 0.2
 0 <= s_k1_PG_s00 <= 0.2
 0 <= s_k1_PG_s01 <= 0.2
 0 <= s_k1_PG_s10 <= 0.2
 0 <= s_k1_PG_s11 <= 0.2
 0 <= s_k1_PG_s000 <= 0.2
 0 <= s_k1_PG_s001 <= 0.2
 0 <= s_k1_PG_s002 <= 0.2
 0 <= s_k1_PG_s003 <= 0.2
 0 <= s_k1_PG_s004 <= 0.2
 0 <= s_k1_PG_s005 <= 0.2
 0 <= s_k1_PG_s006 <= 0.2
 0 <= s_k1_PG_s007 <= 0.2
 0 <= s_k1_PG_s010 <= 0.2
 0 <= s_k1_PG_s011 <= 0.2
 0 <= s_k1_PG_s012 <= 0.2
 0 <= s_k1_PG_s013 <= 0.2
 0 <= s_k1_PG_s014 <= 0.2
 0 <= s_k1_PG_s015 <= 0.2
 0 <= s_k1_PG_s016 <= 0.2
 0 <= s_k1_PG_s017 <= 0.2
 0 <= s_k1_PG_s100 <= 0.2
 0 <= s_k1_PG_s101 <= 0.2
 0 <= s_k1_PG_s102 <= 0.2
 0 <= s_k1_PG_s103 <= 0.2
 0 <= s_k1_PG_s104 <= 0.2
 0 <= s_k1_PG_s105 <= 0.2
 0 <= s_k1_PG_s106 <= 0.2
 0 <= s_k1_PG_s107 <= 0.2
 0 <= s_k1_PG_s110 <= 0.2
 0 <= s_k1_PG_s111 <= 0.2
 0 <= s_k1_PG_s112 <= 0.2
 0 <= s_k1_PG_s113 <= 0.2
 0 <= s_k1_PG_s114 <= 0.2
 0 <= s_k1_PG_s115 <= 0.2
 0 <= s_k1_PG_s116 <= 0.2
 0 <= s_k1_PG_s117 <= 0.2
End
