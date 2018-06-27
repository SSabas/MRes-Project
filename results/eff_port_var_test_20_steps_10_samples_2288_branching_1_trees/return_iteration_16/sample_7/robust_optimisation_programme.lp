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
 w_k1_ADM_s0_return_constraint:  0.990606 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.09509 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.13888 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.834674 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.90832 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.05725 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.03565 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.913605 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.30886 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.33784 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.734993 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.885131 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.00859 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.0313 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.734984 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.39516 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.20628 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.926633 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.05793 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.78287 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.08287 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.903296 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.802818 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.764901 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.01331 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.03261 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.811551 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.05433 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.32314 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.2289 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.04613 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.1311 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.19333 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.892835 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.06654 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.20959 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.53613 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.47274 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  0.987247 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.19884 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.24267 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.761209 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.841764 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.232 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.06431 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.877437 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.5828 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.24013 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.704417 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.81575 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.855622 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.17168 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.672157 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.42713 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.04152 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.714635 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.966376 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.861953 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.42037 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.933436 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.669739 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.959118 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.18397 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.10837 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.896154 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.12683 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.50842 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.52972 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.19472 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.995235 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.04482 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.655893 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.89701 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.12373 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.37985 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.71974 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  0.963122 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.23135 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.36406 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.772015 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.831781 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.33562 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.30057 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.886511 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.48179 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.03958 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.554864 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.724039 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.873697 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.27994 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.632194 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.63513 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.970773 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.615869 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.949471 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.638968 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.2135 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.905291 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.643303 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.952554 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.09967 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.1046 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.837547 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.877132 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.57794 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.37412 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.15835 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.736392 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.939756 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.650383 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.09742 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.21935 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.73156 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.6694 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.00278 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  1.08409 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.22597 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.823009 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.864341 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.23484 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.12683 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.889364 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.366 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.22799 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.717699 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.733576 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.983599 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.02968 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.635519 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.23917 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.05421 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.814305 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.922234 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.79009 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.23742 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.00411 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.726668 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.801955 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.09406 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.17465 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.848276 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.03786 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.37092 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.19419 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.01215 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.891052 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.01966 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.752615 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.913139 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.28253 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.41116 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.38644 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.899959 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.34498 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.08127 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.946508 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.0038 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.07911 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.19688 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.918561 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.2078 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.697244 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.692886 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.24602 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.558026 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.82207 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.985702 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.53253 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.835227 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.492991 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.17965 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.863717 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.911714 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.637744 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.745162 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.58546 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.12416 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.695046 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.09579 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.812367 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.15355 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.70195 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.66979 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.79234 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.906445 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.720534 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.17547 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.664699 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.971313 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.33994 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 0.967329 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.989165 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.17151 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.894745 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.959549 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.31047 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.27634 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.999632 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.22882 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.1441 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.721147 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.791487 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.96543 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.06385 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.850078 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.19886 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.01521 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.811698 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.02979 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.789611 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.14476 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.97083 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.84242 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.952944 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.09425 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.09412 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.884746 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.10243 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.42729 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.20972 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.02704 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.857179 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.90955 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.764619 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.05981 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.13337 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.30014 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.16093 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.91114 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.20406 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.27328 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.91064 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.891317 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.26385 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.15801 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.855246 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.23398 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.891803 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.59833 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.867523 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.659458 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.27865 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.759604 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.5228 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.879643 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.636528 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.07902 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.813083 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.02338 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.737963 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.754453 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.12416 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.07549 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.00402 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.06241 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.96255 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.45271 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.56823 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.314 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.813004 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.888731 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.683833 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.01153 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.839588 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.18287 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.16551 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.897505 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.44786 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.46018 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.881913 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.812511 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.28568 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.836927 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.659529 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.31568 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.744945 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.579751 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.884619 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.360231 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.42496 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.760774 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.30553 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.796926 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.585381 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.04742 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.996944 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.26231 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.514358 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.715609 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.57519 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.10364 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.833315 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.19023 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.01381 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.36772 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.93429 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.32936 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.682659 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.703985 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.572483 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.792297 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.546766 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.739222 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.10042 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.01137 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.06798 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.16577 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.852997 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.916523 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.28268 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.12087 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.935839 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.16244 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.1017 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.744021 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.805343 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.949915 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.08392 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.710569 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.19696 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.982256 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.771797 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.01981 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.748578 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.16417 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.948834 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.753317 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.881943 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.971557 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.15289 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.846898 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.00031 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.34293 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.1795 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.937771 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.772246 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.932197 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.79772 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.03376 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.17195 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.26543 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.13668 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   0.962034 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.10201 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.13177 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.905606 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.959205 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.15677 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.12227 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.928093 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.1503 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.09624 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.811114 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.907404 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.896164 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.13775 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.830869 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.20502 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.07634 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.745229 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.05172 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.94291 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.13594 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.94114 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.738214 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.898886 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.970058 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.962139 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.86628 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.938708 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.17274 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.19087 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.13776 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.879076 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.03226 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.79417 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.00516 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.06126 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.20711 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.20162 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.367999728943082 z_k1_s0000
                                 - 1.11700152744916 z_k1_s0001
                                 - 0.456999704620052 z_k1_s0002
                                 - 0.532999507275776 z_k1_s0003
                                 - 1.13200131024317 z_k1_s0004
                                 - 0.699001799993184 z_k1_s0005
                                 - 0.697002190220429 z_k1_s0006
                                 - 0.415999575783557 z_k1_s0007
                                 - 1.46699697262947 z_k1_s0010
                                 - 1.39600066014873 z_k1_s0011
                                 - 1.30599646968113 z_k1_s0012
                                 - 0.536000340031881 z_k1_s0013
                                 - 0.638999541123173 z_k1_s0014
                                 - 1.37999711784367 z_k1_s0015
                                 - 1.19899848101335 z_k1_s0016
                                 - 0.469000297527896 z_k1_s0017
                                 - 0.263999641126094 z_k1_s0020
                                 - 0.300999642453529 z_k1_s0021
                                 - 0.317999412847596 z_k1_s0022
                                 - 0.314999378207591 z_k1_s0023
                                 - 0.278999388426824 z_k1_s0024
                                 - 1.03999923155627 z_k1_s0025
                                 - 1.1089991764778 z_k1_s0026
                                 - 0.632997087458004 z_k1_s0027
                                 - 0.206000375193867 z_k1_s0030
                                 - 0.0899999747623921 z_k1_s0031
                                 - 0.341998998817199 z_k1_s0032
                                 - 0.320000697466188 z_k1_s0033
                                 - 0.222999957146814 z_k1_s0034
                                 - 0.23999953909976 z_k1_s0035
                                 - 0.386999281398359 z_k1_s0036
                                 - 0.159999955133141 z_k1_s0037
                                 - 0.594001233233785 z_k1_s0040
                                 - 0.205999933175182 z_k1_s0041
                                 - 0.178999961446403 z_k1_s0042
                                 - 0.229000092321983 z_k1_s0043
                                 - 0.488001252609657 z_k1_s0044
                                 - 0.761998927278786 z_k1_s0045
                                 - 0.377001506258056 z_k1_s0046
                                 - 0.257000017196257 z_k1_s0047
                                 - 0.130000136466063 z_k1_s0050
                                 - 0.234001212900119 z_k1_s0051
                                 - 0.285001211519045 z_k1_s0052
                                 - 0.60799966743699 z_k1_s0053
                                 - 0.173000194005622 z_k1_s0054
                                 - 0.324000262167628 z_k1_s0055
                                 - 0.287000218012665 z_k1_s0056
                                 - 0.262001121828044 z_k1_s0057
                                 - 0.268999957026567 z_k1_s0060
                                 - 0.247000059002104 z_k1_s0061
                                 - 0.832000165943924 z_k1_s0062
                                 - 0.247000059002104 z_k1_s0063
                                 - 0.247000059002104 z_k1_s0064
                                 - 0.328999792220574 z_k1_s0065
                                 - 0.594000051060917 z_k1_s0066
                                 - 0.3509987554449 z_k1_s0067
                                 - 1.00299777339408 z_k1_s0070
                                 - 0.36700007195643 z_k1_s0071
                                 - 0.438999936632372 z_k1_s0072
                                 - 0.719001038905245 z_k1_s0073
                                 - 0.575997971807679 z_k1_s0074
                                 - 0.478998337985896 z_k1_s0075
                                 - 0.668997718214117 z_k1_s0076
                                 - 0.476000786424042 z_k1_s0077
                                 - 0.281999915806071 z_k1_s0100
                                 - 0.222999875827155 z_k1_s0101
                                 - 0.341998660145957 z_k1_s0102
                                 - 0.369998999444349 z_k1_s0103
                                 - 0.469999367510237 z_k1_s0104
                                 - 0.64500001162554 z_k1_s0105
                                 - 0.292999838173347 z_k1_s0106
                                 - 0.327998490496761 z_k1_s0107
                                 - 0.635999149843754 z_k1_s0110
                                 - 1.23700111873416 z_k1_s0111
                                 - 1.59700032962715 z_k1_s0112
                                 - 1.313002130666 z_k1_s0113
                                 - 0.453999775419859 z_k1_s0114
                                 - 0.718998945246881 z_k1_s0115
                                 - 0.731999483348028 z_k1_s0116
                                 - 1.47100101376421 z_k1_s0117
                                 - 0.216000012373806 z_k1_s0120
                                 - 0.472999083714653 z_k1_s0121
                                 - 0.121999926569486 z_k1_s0122
                                 - 0.416998956784441 z_k1_s0123
                                 - 0.249000802993203 z_k1_s0124
                                 - 0.331998843658203 z_k1_s0125
                                 - 0.16700001265984 z_k1_s0126
                                 - 0.251000648455039 z_k1_s0127
                                 - 0.477998390196913 z_k1_s0130
                                 - 0.399999913260836 z_k1_s0131
                                 - 0.992000454286778 z_k1_s0132
                                 - 0.485000313203704 z_k1_s0133
                                 - 0.255999944486935 z_k1_s0134
                                 - 0.211999954028243 z_k1_s0135
                                 - 0.290000104464082 z_k1_s0136
                                 - 0.234000116607565 z_k1_s0137
                                 - 1.24900059581877 z_k1_s0140
                                 - 1.0039967393507 z_k1_s0141
                                 - 1.12499703393702 z_k1_s0142
                                 - 0.746997743654916 z_k1_s0143
                                 - 2.31899749585291 z_k1_s0144
                                 - 1.10999862741387 z_k1_s0145
                                 - 0.56899812529444 z_k1_s0146
                                 - 0.841997811942114 z_k1_s0147
                                 - 0.152999788602432 z_k1_s0150
                                 - 0.375000956289188 z_k1_s0151
                                 - 0.372000785502995 z_k1_s0152
                                 - 0.415999042036962 z_k1_s0153
                                 - 0.229999923784524 z_k1_s0154
                                 - 0.28299996717613 z_k1_s0155
                                 - 0.404000907890309 z_k1_s0156
                                 - 0.315999296826136 z_k1_s0157
                                 - 0.633995378743013 z_k1_s0160
                                 - 1.01999568197832 z_k1_s0161
                                 - 0.65299632945824 z_k1_s0162
                                 - 0.276999014842408 z_k1_s0163
                                 - 0.669999975876871 z_k1_s0164
                                 - 0.458998056890242 z_k1_s0165
                                 - 0.874994577233125 z_k1_s0166
                                 - 0.723996247618632 z_k1_s0167
                                 - 0.598000621452162 z_k1_s0170
                                 - 0.266000966735244 z_k1_s0171
                                 - 0.258000875585101 z_k1_s0172
                                 - 0.38800111837072 z_k1_s0173
                                 - 0.853004558866467 z_k1_s0174
                                 - 0.726002079854448 z_k1_s0175
                                 - 0.319000951402839 z_k1_s0176
                                 - 0.720004075498845 z_k1_s0177
                                 - 0.271999837648544 z_k1_s1000
                                 - 0.570997638503659 z_k1_s1001
                                 - 0.596001857914077 z_k1_s1002
                                 - 0.258999725794078 z_k1_s1003
                                 - 0.627998929803606 z_k1_s1004
                                 - 0.731998089440205 z_k1_s1005
                                 - 0.765001576823006 z_k1_s1006
                                 - 0.514998424699168 z_k1_s1007
                                 - 0.0880000132344553 z_k1_s1010
                                 - 0.100999815651973 z_k1_s1011
                                 - 0.227000515376554 z_k1_s1012
                                 - 0.0760000583797739 z_k1_s1013
                                 - 0.0609999974363795 z_k1_s1014
                                 - 0.0910000254231342 z_k1_s1015
                                 - 0.197000393489765 z_k1_s1016
                                 - 0.0979997095632599 z_k1_s1017
                                 - 0.18899946202665 z_k1_s1020
                                 - 0.165999608662325 z_k1_s1021
                                 - 0.446998032947583 z_k1_s1022
                                 - 0.160999424748967 z_k1_s1023
                                 - 0.208999295282699 z_k1_s1024
                                 - 0.753998135500212 z_k1_s1025
                                 - 0.15499961013176 z_k1_s1026
                                 - 0.174999443387809 z_k1_s1027
                                 - 0.371000015184589 z_k1_s1030
                                 - 0.204999618957016 z_k1_s1031
                                 - 0.102999812744251 z_k1_s1032
                                 - 0.0789998279883505 z_k1_s1033
                                 - 0.195998872174659 z_k1_s1034
                                 - 0.270999362036058 z_k1_s1035
                                 - 0.237000171964041 z_k1_s1036
                                 - 0.257999621126243 z_k1_s1037
                                 - 0.0749998774574163 z_k1_s1040
                                 - 0.103999938703526 z_k1_s1041
                                 - 0.236999302153697 z_k1_s1042
                                 - 0.181999383531599 z_k1_s1043
                                 - 0.0869999291385428 z_k1_s1044
                                 - 0.227999199742906 z_k1_s1045
                                 - 0.184999842001505 z_k1_s1046
                                 - 0.175000435433363 z_k1_s1047
                                 - 0.290000994201048 z_k1_s1050
                                 - 0.164000030038032 z_k1_s1051
                                 - 0.208999918210707 z_k1_s1052
                                 - 0.781000548687061 z_k1_s1053
                                 - 0.433000420045408 z_k1_s1054
                                 - 0.243000034772209 z_k1_s1055
                                 - 0.175000067749106 z_k1_s1056
                                 - 0.365999125951079 z_k1_s1057
                                 - 0.281000260558299 z_k1_s1060
                                 - 0.100999865186643 z_k1_s1061
                                 - 0.180999211463688 z_k1_s1062
                                 - 0.142999857906786 z_k1_s1063
                                 - 0.112999841778132 z_k1_s1064
                                 - 0.180000395371657 z_k1_s1065
                                 - 0.102999886168535 z_k1_s1066
                                 - 0.390999921563728 z_k1_s1067
                                 - 0.167000232315985 z_k1_s1070
                                 - 0.345001604241409 z_k1_s1071
                                 - 0.341999677436247 z_k1_s1072
                                 - 0.100000159069484 z_k1_s1073
                                 - 0.272999800988677 z_k1_s1074
                                 - 0.367001661456731 z_k1_s1075
                                 - 0.319999620220925 z_k1_s1076
                                 - 0.308000801014508 z_k1_s1077
                                 - 0.366999028393605 z_k1_s1100
                                 - 0.119999775361183 z_k1_s1101
                                 - 0.297998675002034 z_k1_s1102
                                 - 0.362999357587493 z_k1_s1103
                                 - 0.203999915073328 z_k1_s1104
                                 - 0.0889998210195724 z_k1_s1105
                                 - 0.24999891333722 z_k1_s1106
                                 - 0.164999691121626 z_k1_s1107
                                 - 0.0890002669565081 z_k1_s1110
                                 - 0.142999929836198 z_k1_s1111
                                 - 0.223001046182264 z_k1_s1112
                                 - 0.24100146261072 z_k1_s1113
                                 - 0.202001001017309 z_k1_s1114
                                 - 0.208000258490307 z_k1_s1115
                                 - 0.114000434039003 z_k1_s1116
                                 - 0.10200033268733 z_k1_s1117
                                 - 0.203000006383309 z_k1_s1120
                                 - 0.319999362614095 z_k1_s1121
                                 - 0.593999873002396 z_k1_s1122
                                 - 0.3420001402438 z_k1_s1123
                                 - 0.163000086056548 z_k1_s1124
                                 - 0.134999907127818 z_k1_s1125
                                 - 0.332999395395291 z_k1_s1126
                                 - 0.256998842674456 z_k1_s1127
                                 - 0.340001125865004 z_k1_s1130
                                 - 0.303999660204758 z_k1_s1131
                                 - 0.257001202269578 z_k1_s1132
                                 - 0.332002182389006 z_k1_s1133
                                 - 0.557002634739987 z_k1_s1134
                                 - 0.285001236446524 z_k1_s1135
                                 - 0.218000692607475 z_k1_s1136
                                 - 0.195000557904671 z_k1_s1137
                                 - 0.169999435312286 z_k1_s1140
                                 - 0.404000867519952 z_k1_s1141
                                 - 0.187000043643919 z_k1_s1142
                                 - 0.384000347365209 z_k1_s1143
                                 - 0.085000050056345 z_k1_s1144
                                 - 0.0930000919181412 z_k1_s1145
                                 - 0.143000062704191 z_k1_s1146
                                 - 0.19599977911325 z_k1_s1147
                                 - 0.0879998736244026 z_k1_s1150
                                 - 0.051000071182468 z_k1_s1151
                                 - 0.0600000501440466 z_k1_s1152
                                 - 0.117000440501232 z_k1_s1153
                                 - 0.114000209513802 z_k1_s1154
                                 - 0.0769999786714412 z_k1_s1155
                                 - 0.0620000375355006 z_k1_s1156
                                 - 0.144999978381304 z_k1_s1157
                                 - 0.0409999894613557 z_k1_s1160
                                 - 0.0429999954155686 z_k1_s1161
                                 - 0.10800005632748 z_k1_s1162
                                 - 0.0470000073239946 z_k1_s1163
                                 - 0.0340000349216154 z_k1_s1164
                                 - 0.117000282021454 z_k1_s1165
                                 - 0.127000311792518 z_k1_s1166
                                 - 0.145999904257506 z_k1_s1167
                                 - 0.0850000634977212 z_k1_s1170
                                 - 0.10900001368647 z_k1_s1171
                                 - 0.0880000431588108 z_k1_s1172
                                 - 0.181000202952814 z_k1_s1173
                                 - 0.20500049184166 z_k1_s1174
                                 - 0.213999753424734 z_k1_s1175
                                 - 0.0539999725997093 z_k1_s1176
                                 - 0.193000347397237 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.15662 w_k1_ADM_s000
                                 + 1.33929 w_k1_AXP_s000
                                 + 1.51843 w_k1_DIS_s000
                                 + 1.17905 w_k1_FDX_s000
                                 + 1.54124 w_k1_IBM_s000
                                 + 1.09279 w_k1_INTC_s000
                                 + 1.35081 w_k1_KO_s000 + 1.96275 w_k1_MAT_s000
                                 + 1.20418 w_k1_MSFT_s000
                                 + 1.32294 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.16773 w_k1_ADM_s000
                                 + 0.921961 w_k1_AXP_s000
                                 + 1.13229 w_k1_DIS_s000
                                 + 1.08233 w_k1_FDX_s000
                                 + 0.814477 w_k1_IBM_s000
                                 + 0.869285 w_k1_INTC_s000
                                 + 0.984387 w_k1_KO_s000
                                 + 1.06498 w_k1_MAT_s000
                                 + 1.19151 w_k1_MSFT_s000 + 1.0245 w_k1_PG_s000
                                 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.880523 w_k1_ADM_s000
                                 + 1.07194 w_k1_AXP_s000
                                 + 0.794609 w_k1_DIS_s000
                                 + 0.954945 w_k1_FDX_s000
                                 + 0.869008 w_k1_IBM_s000
                                 + 0.950759 w_k1_INTC_s000
                                 + 0.880393 w_k1_KO_s000
                                 + 1.19108 w_k1_MAT_s000
                                 + 1.01697 w_k1_MSFT_s000
                                 + 0.940851 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.4753 w_k1_ADM_s000
                                 + 1.15702 w_k1_AXP_s000
                                 + 1.55783 w_k1_DIS_s000
                                 + 1.36838 w_k1_FDX_s000 + 0.914 w_k1_IBM_s000
                                 + 0.849899 w_k1_INTC_s000
                                 + 1.07012 w_k1_KO_s000 + 1.1522 w_k1_MAT_s000
                                 + 1.40486 w_k1_MSFT_s000 + 1.1396 w_k1_PG_s000
                                 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.08719 w_k1_ADM_s000
                                 + 0.751467 w_k1_AXP_s000
                                 + 0.670843 w_k1_DIS_s000
                                 + 0.825545 w_k1_FDX_s000
                                 + 0.714602 w_k1_IBM_s000
                                 + 0.756216 w_k1_INTC_s000
                                 + 0.650215 w_k1_KO_s000
                                 + 0.714404 w_k1_MAT_s000
                                 + 0.814052 w_k1_MSFT_s000
                                 + 0.830339 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.869267 w_k1_ADM_s000
                                 + 0.731733 w_k1_AXP_s000
                                 + 0.810042 w_k1_DIS_s000
                                 + 0.831693 w_k1_FDX_s000
                                 + 1.02881 w_k1_IBM_s000
                                 + 0.925281 w_k1_INTC_s000
                                 + 1.03059 w_k1_KO_s000
                                 + 0.908252 w_k1_MAT_s000
                                 + 0.92818 w_k1_MSFT_s000
                                 + 0.948812 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.01945 w_k1_ADM_s000
                                 + 1.08818 w_k1_AXP_s000
                                 + 0.94548 w_k1_DIS_s000
                                 + 1.01852 w_k1_FDX_s000
                                 + 1.14602 w_k1_IBM_s000
                                 + 0.846853 w_k1_INTC_s000
                                 + 0.972945 w_k1_KO_s000 + 1.3798 w_k1_MAT_s000
                                 + 1.04406 w_k1_MSFT_s000
                                 + 0.991428 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.08267 w_k1_ADM_s000
                                 + 1.2368 w_k1_AXP_s000 + 1.26599 w_k1_DIS_s000
                                 + 1.15838 w_k1_FDX_s000
                                 + 1.17026 w_k1_IBM_s000
                                 + 0.882034 w_k1_INTC_s000
                                 + 1.02368 w_k1_KO_s000 + 1.53095 w_k1_MAT_s000
                                 + 1.14967 w_k1_MSFT_s000
                                 + 0.976519 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.05821 w_k1_ADM_s001
                                 + 0.691942 w_k1_AXP_s001
                                 + 0.860382 w_k1_DIS_s001
                                 + 0.861972 w_k1_FDX_s001
                                 + 0.79546 w_k1_IBM_s001
                                 + 0.689096 w_k1_INTC_s001
                                 + 0.8248 w_k1_KO_s001 + 0.853546 w_k1_MAT_s001
                                 + 0.833643 w_k1_MSFT_s001
                                 + 0.946227 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.00066 w_k1_ADM_s001
                                 + 0.805773 w_k1_AXP_s001
                                 + 0.718454 w_k1_DIS_s001
                                 + 0.822246 w_k1_FDX_s001
                                 + 1.02104 w_k1_IBM_s001
                                 + 0.711432 w_k1_INTC_s001
                                 + 0.846677 w_k1_KO_s001
                                 + 1.02819 w_k1_MAT_s001
                                 + 0.746362 w_k1_MSFT_s001
                                 + 0.914647 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.07619 w_k1_ADM_s001
                                 + 1.13237 w_k1_AXP_s001
                                 + 0.947025 w_k1_DIS_s001
                                 + 1.07245 w_k1_FDX_s001
                                 + 1.10351 w_k1_IBM_s001
                                 + 1.02289 w_k1_INTC_s001
                                 + 1.12946 w_k1_KO_s001 + 1.19787 w_k1_MAT_s001
                                 + 1.00139 w_k1_MSFT_s001
                                 + 1.05126 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.3395 w_k1_ADM_s001
                                 + 1.22655 w_k1_AXP_s001
                                 + 1.54409 w_k1_DIS_s001
                                 + 1.20753 w_k1_FDX_s001
                                 + 1.51066 w_k1_IBM_s001
                                 + 0.97822 w_k1_INTC_s001 + 1.5678 w_k1_KO_s001
                                 + 2.01614 w_k1_MAT_s001
                                 + 1.10036 w_k1_MSFT_s001
                                 + 1.26677 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.17794 w_k1_ADM_s001
                                 + 1.11376 w_k1_AXP_s001
                                 + 1.36774 w_k1_DIS_s001 + 1.1497 w_k1_FDX_s001
                                 + 1.01454 w_k1_IBM_s001
                                 + 1.20208 w_k1_INTC_s001
                                 + 1.30726 w_k1_KO_s001
                                 + 0.801192 w_k1_MAT_s001
                                 + 1.2348 w_k1_MSFT_s001 + 1.06781 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.839249 w_k1_ADM_s001
                                 + 0.819591 w_k1_AXP_s001
                                 + 0.79947 w_k1_DIS_s001
                                 + 0.969855 w_k1_FDX_s001
                                 + 0.80894 w_k1_IBM_s001
                                 + 0.918155 w_k1_INTC_s001
                                 + 0.755221 w_k1_KO_s001
                                 + 1.01254 w_k1_MAT_s001
                                 + 0.938868 w_k1_MSFT_s001
                                 + 0.922169 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.38346 w_k1_ADM_s001
                                 + 1.18701 w_k1_AXP_s001
                                 + 1.36273 w_k1_DIS_s001 + 1.3656 w_k1_FDX_s001
                                 + 0.988948 w_k1_IBM_s001
                                 + 1.22655 w_k1_INTC_s001 + 1.1174 w_k1_KO_s001
                                 + 1.11709 w_k1_MAT_s001
                                 + 1.11614 w_k1_MSFT_s001 + 1.1111 w_k1_PG_s001
                                 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.09992 w_k1_ADM_s001
                                 + 1.27427 w_k1_AXP_s001
                                 + 0.95964 w_k1_DIS_s001
                                 + 1.00176 w_k1_FDX_s001
                                 + 1.66283 w_k1_IBM_s001
                                 + 0.877674 w_k1_INTC_s001
                                 + 1.12166 w_k1_KO_s001 + 2.06291 w_k1_MAT_s001
                                 + 0.894928 w_k1_MSFT_s001
                                 + 0.999131 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.967899 w_k1_ADM_s002
                                 + 0.766421 w_k1_AXP_s002
                                 + 0.60733 w_k1_DIS_s002
                                 + 0.651848 w_k1_FDX_s002
                                 + 0.969502 w_k1_IBM_s002
                                 + 0.727233 w_k1_INTC_s002
                                 + 0.707755 w_k1_KO_s002
                                 + 0.791327 w_k1_MAT_s002
                                 + 0.811692 w_k1_MSFT_s002
                                 + 0.953385 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.794578 w_k1_ADM_s002
                                 + 0.696137 w_k1_AXP_s002
                                 + 0.666504 w_k1_DIS_s002
                                 + 0.7334 w_k1_FDX_s002
                                 + 0.654761 w_k1_IBM_s002
                                 + 0.914737 w_k1_INTC_s002
                                 + 0.64043 w_k1_KO_s002
                                 + 0.454237 w_k1_MAT_s002
                                 + 0.795669 w_k1_MSFT_s002
                                 + 0.702826 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.13374 w_k1_ADM_s002
                                 + 1.40194 w_k1_AXP_s002
                                 + 1.79004 w_k1_DIS_s002 + 1.2572 w_k1_FDX_s002
                                 + 1.68602 w_k1_IBM_s002
                                 + 1.19994 w_k1_INTC_s002
                                 + 1.58033 w_k1_KO_s002 + 2.01644 w_k1_MAT_s002
                                 + 1.45704 w_k1_MSFT_s002
                                 + 1.37416 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.873707 w_k1_ADM_s002
                                 + 0.947243 w_k1_AXP_s002
                                 + 0.622244 w_k1_DIS_s002
                                 + 0.760012 w_k1_FDX_s002
                                 + 0.891598 w_k1_IBM_s002
                                 + 0.84299 w_k1_INTC_s002
                                 + 0.72184 w_k1_KO_s002
                                 + 0.720936 w_k1_MAT_s002
                                 + 0.865864 w_k1_MSFT_s002
                                 + 0.971144 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.08669 w_k1_ADM_s002
                                 + 0.874876 w_k1_AXP_s002
                                 + 0.688566 w_k1_DIS_s002
                                 + 0.963574 w_k1_FDX_s002
                                 + 0.684127 w_k1_IBM_s002
                                 + 0.812093 w_k1_INTC_s002
                                 + 0.725632 w_k1_KO_s002
                                 + 0.709516 w_k1_MAT_s002
                                 + 0.898902 w_k1_MSFT_s002
                                 + 1.01941 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.38 w_k1_ADM_s002
                                 + 1.21682 w_k1_AXP_s002
                                 + 1.62063 w_k1_DIS_s002
                                 + 1.29727 w_k1_FDX_s002
                                 + 1.07677 w_k1_IBM_s002
                                 + 0.965641 w_k1_INTC_s002
                                 + 1.23524 w_k1_KO_s002
                                 + 0.929154 w_k1_MAT_s002
                                 + 1.47009 w_k1_MSFT_s002
                                 + 1.26208 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.1694 w_k1_ADM_s002
                                 + 0.984175 w_k1_AXP_s002
                                 + 0.905437 w_k1_DIS_s002
                                 + 1.01335 w_k1_FDX_s002
                                 + 0.841478 w_k1_IBM_s002
                                 + 0.961215 w_k1_INTC_s002
                                 + 0.851086 w_k1_KO_s002
                                 + 0.797031 w_k1_MAT_s002
                                 + 1.0448 w_k1_MSFT_s002 + 1.14743 w_k1_PG_s002
                                 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.20596 w_k1_ADM_s002
                                 + 1.05712 w_k1_AXP_s002
                                 + 0.88908 w_k1_DIS_s002
                                 + 0.927352 w_k1_FDX_s002
                                 + 1.36385 w_k1_IBM_s002
                                 + 0.890423 w_k1_INTC_s002
                                 + 1.11517 w_k1_KO_s002 + 1.29605 w_k1_MAT_s002
                                 + 0.958656 w_k1_MSFT_s002
                                 + 1.13626 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.634339 w_k1_ADM_s003
                                 + 0.786217 w_k1_AXP_s003
                                 + 0.540305 w_k1_DIS_s003
                                 + 0.689735 w_k1_FDX_s003
                                 + 0.963963 w_k1_IBM_s003
                                 + 0.744875 w_k1_INTC_s003
                                 + 0.64329 w_k1_KO_s003
                                 + 0.864764 w_k1_MAT_s003
                                 + 0.65921 w_k1_MSFT_s003
                                 + 0.788756 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.07198 w_k1_ADM_s003
                                 + 1.24991 w_k1_AXP_s003
                                 + 1.56448 w_k1_DIS_s003
                                 + 1.14464 w_k1_FDX_s003
                                 + 1.46951 w_k1_IBM_s003
                                 + 1.43921 w_k1_INTC_s003
                                 + 1.43646 w_k1_KO_s003
                                 + 0.829584 w_k1_MAT_s003
                                 + 1.33429 w_k1_MSFT_s003
                                 + 1.10577 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.921705 w_k1_ADM_s003
                                 + 0.841948 w_k1_AXP_s003
                                 + 1.01073 w_k1_DIS_s003
                                 + 0.993414 w_k1_FDX_s003
                                 + 0.813548 w_k1_IBM_s003
                                 + 1.10702 w_k1_INTC_s003
                                 + 0.968303 w_k1_KO_s003
                                 + 0.726459 w_k1_MAT_s003
                                 + 1.08078 w_k1_MSFT_s003
                                 + 0.889393 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.08156 w_k1_ADM_s003
                                 + 1.35025 w_k1_AXP_s003
                                 + 1.18535 w_k1_DIS_s003
                                 + 1.16591 w_k1_FDX_s003
                                 + 1.18191 w_k1_IBM_s003
                                 + 1.00764 w_k1_INTC_s003
                                 + 1.09873 w_k1_KO_s003 + 1.07443 w_k1_MAT_s003
                                 + 1.09067 w_k1_MSFT_s003
                                 + 1.04956 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.936543 w_k1_ADM_s003
                                 + 0.903432 w_k1_AXP_s003
                                 + 1.16819 w_k1_DIS_s003 + 1.1671 w_k1_FDX_s003
                                 + 0.830609 w_k1_IBM_s003
                                 + 1.18324 w_k1_INTC_s003
                                 + 1.03454 w_k1_KO_s003
                                 + 0.639427 w_k1_MAT_s003
                                 + 1.2094 w_k1_MSFT_s003
                                 + 0.968996 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.833376 w_k1_ADM_s003
                                 + 0.718165 w_k1_AXP_s003
                                 + 0.695751 w_k1_DIS_s003
                                 + 0.863699 w_k1_FDX_s003
                                 + 0.698569 w_k1_IBM_s003
                                 + 0.913457 w_k1_INTC_s003
                                 + 0.796657 w_k1_KO_s003
                                 + 0.484301 w_k1_MAT_s003
                                 + 0.847669 w_k1_MSFT_s003
                                 + 0.854022 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.837352 w_k1_ADM_s003
                                 + 0.968344 w_k1_AXP_s003
                                 + 0.785922 w_k1_DIS_s003
                                 + 0.895388 w_k1_FDX_s003
                                 + 1.15658 w_k1_IBM_s003
                                 + 0.978067 w_k1_INTC_s003
                                 + 0.921402 w_k1_KO_s003
                                 + 0.996508 w_k1_MAT_s003
                                 + 0.835837 w_k1_MSFT_s003
                                 + 0.9531 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.01133 w_k1_ADM_s003
                                 + 1.54408 w_k1_AXP_s003 + 1.643 w_k1_DIS_s003
                                 + 1.45342 w_k1_FDX_s003 + 1.1876 w_k1_IBM_s003
                                 + 1.43844 w_k1_INTC_s003
                                 + 1.13922 w_k1_KO_s003
                                 + 0.888301 w_k1_MAT_s003
                                 + 1.29841 w_k1_MSFT_s003
                                 + 0.920447 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.07299 w_k1_ADM_s004
                                 + 0.849472 w_k1_AXP_s004
                                 + 0.816506 w_k1_DIS_s004
                                 + 0.868638 w_k1_FDX_s004
                                 + 0.824474 w_k1_IBM_s004
                                 + 0.892952 w_k1_INTC_s004
                                 + 0.873914 w_k1_KO_s004
                                 + 0.72368 w_k1_MAT_s004
                                 + 0.815894 w_k1_MSFT_s004
                                 + 0.918428 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.846599 w_k1_ADM_s004
                                 + 0.65613 w_k1_AXP_s004
                                 + 0.654233 w_k1_DIS_s004
                                 + 0.720398 w_k1_FDX_s004
                                 + 1.02783 w_k1_IBM_s004
                                 + 0.800413 w_k1_INTC_s004
                                 + 0.957845 w_k1_KO_s004
                                 + 1.02331 w_k1_MAT_s004
                                 + 0.827629 w_k1_MSFT_s004
                                 + 0.87494 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.27589 w_k1_ADM_s004
                                 + 0.899036 w_k1_AXP_s004
                                 + 1.00916 w_k1_DIS_s004
                                 + 1.14902 w_k1_FDX_s004
                                 + 0.48987 w_k1_IBM_s004
                                 + 1.0169 w_k1_INTC_s004 + 0.73129 w_k1_KO_s004
                                 + 0.460974 w_k1_MAT_s004
                                 + 0.938251 w_k1_MSFT_s004
                                 + 0.909336 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.924637 w_k1_ADM_s004
                                 + 0.615167 w_k1_AXP_s004
                                 + 0.629158 w_k1_DIS_s004
                                 + 0.752289 w_k1_FDX_s004
                                 + 0.648658 w_k1_IBM_s004
                                 + 0.75302 w_k1_INTC_s004
                                 + 0.740531 w_k1_KO_s004
                                 + 0.481767 w_k1_MAT_s004
                                 + 0.727867 w_k1_MSFT_s004
                                 + 0.734787 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.11933 w_k1_ADM_s004
                                 + 1.11593 w_k1_AXP_s004
                                 + 0.928404 w_k1_DIS_s004
                                 + 0.922918 w_k1_FDX_s004
                                 + 1.44804 w_k1_IBM_s004
                                 + 0.981947 w_k1_INTC_s004
                                 + 1.21904 w_k1_KO_s004 + 1.21727 w_k1_MAT_s004
                                 + 0.88263 w_k1_MSFT_s004
                                 + 1.09787 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.13979 w_k1_ADM_s004
                                 + 1.08715 w_k1_AXP_s004
                                 + 1.19501 w_k1_DIS_s004
                                 + 1.12945 w_k1_FDX_s004
                                 + 0.95334 w_k1_IBM_s004
                                 + 1.13177 w_k1_INTC_s004
                                 + 1.08364 w_k1_KO_s004 + 1.02455 w_k1_MAT_s004
                                 + 1.21286 w_k1_MSFT_s004
                                 + 1.09766 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.5935 w_k1_ADM_s004
                                 + 1.3767 w_k1_AXP_s004 + 1.63265 w_k1_DIS_s004
                                 + 1.30444 w_k1_FDX_s004
                                 + 1.31837 w_k1_IBM_s004
                                 + 1.30221 w_k1_INTC_s004
                                 + 1.61438 w_k1_KO_s004 + 1.26034 w_k1_MAT_s004
                                 + 1.2688 w_k1_MSFT_s004 + 1.13322 w_k1_PG_s004
                                 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.01847 w_k1_ADM_s004
                                 + 0.908758 w_k1_AXP_s004
                                 + 1.15685 w_k1_DIS_s004
                                 + 0.965174 w_k1_FDX_s004
                                 + 0.863756 w_k1_IBM_s004
                                 + 1.11829 w_k1_INTC_s004
                                 + 1.11796 w_k1_KO_s004
                                 + 0.557189 w_k1_MAT_s004
                                 + 1.06782 w_k1_MSFT_s004
                                 + 0.995448 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.16213 w_k1_ADM_s005
                                 + 1.41638 w_k1_AXP_s005
                                 + 1.07797 w_k1_DIS_s005
                                 + 1.22195 w_k1_FDX_s005 + 1.0663 w_k1_IBM_s005
                                 + 1.14393 w_k1_INTC_s005
                                 + 1.03226 w_k1_KO_s005 + 1.63926 w_k1_MAT_s005
                                 + 0.98595 w_k1_MSFT_s005
                                 + 1.12811 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.675461 w_k1_ADM_s005
                                 + 0.870772 w_k1_AXP_s005
                                 + 0.687344 w_k1_DIS_s005
                                 + 0.920139 w_k1_FDX_s005
                                 + 0.693382 w_k1_IBM_s005
                                 + 0.961867 w_k1_INTC_s005
                                 + 0.717886 w_k1_KO_s005
                                 + 0.752303 w_k1_MAT_s005
                                 + 0.74432 w_k1_MSFT_s005
                                 + 0.892615 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.8015 w_k1_ADM_s005
                                 + 0.850266 w_k1_AXP_s005
                                 + 1.05481 w_k1_DIS_s005
                                 + 0.901983 w_k1_FDX_s005
                                 + 1.41475 w_k1_IBM_s005
                                 + 0.968013 w_k1_INTC_s005
                                 + 1.3101 w_k1_KO_s005 + 1.68969 w_k1_MAT_s005
                                 + 1.02168 w_k1_MSFT_s005
                                 + 1.04032 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.01693 w_k1_ADM_s005
                                 + 0.902634 w_k1_AXP_s005
                                 + 0.821845 w_k1_DIS_s005
                                 + 0.85318 w_k1_FDX_s005
                                 + 0.982939 w_k1_IBM_s005
                                 + 0.971066 w_k1_INTC_s005
                                 + 0.907946 w_k1_KO_s005
                                 + 0.920709 w_k1_MAT_s005
                                 + 0.883871 w_k1_MSFT_s005
                                 + 0.998507 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.03499 w_k1_ADM_s005
                                 + 1.34914 w_k1_AXP_s005
                                 + 1.40552 w_k1_DIS_s005
                                 + 1.31674 w_k1_FDX_s005
                                 + 1.12841 w_k1_IBM_s005
                                 + 1.57763 w_k1_INTC_s005
                                 + 1.27354 w_k1_KO_s005 + 1.36464 w_k1_MAT_s005
                                 + 1.29238 w_k1_MSFT_s005
                                 + 1.16245 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.20616 w_k1_ADM_s005
                                 + 1.1111 w_k1_AXP_s005 + 1.08358 w_k1_DIS_s005
                                 + 1.12765 w_k1_FDX_s005
                                 + 0.861406 w_k1_IBM_s005
                                 + 0.960627 w_k1_INTC_s005
                                 + 0.941408 w_k1_KO_s005
                                 + 0.83509 w_k1_MAT_s005
                                 + 0.960737 w_k1_MSFT_s005
                                 + 0.954325 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.697287 w_k1_ADM_s005
                                 + 0.630698 w_k1_AXP_s005
                                 + 0.659551 w_k1_DIS_s005
                                 + 0.790924 w_k1_FDX_s005
                                 + 0.677844 w_k1_IBM_s005
                                 + 1.08187 w_k1_INTC_s005
                                 + 0.721219 w_k1_KO_s005
                                 + 0.487481 w_k1_MAT_s005
                                 + 0.841131 w_k1_MSFT_s005
                                 + 0.887206 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.781187 w_k1_ADM_s005
                                 + 1.2713 w_k1_AXP_s005 + 1.28548 w_k1_DIS_s005
                                 + 1.15412 w_k1_FDX_s005 + 1.4156 w_k1_IBM_s005
                                 + 1.45905 w_k1_INTC_s005
                                 + 1.29829 w_k1_KO_s005 + 1.39256 w_k1_MAT_s005
                                 + 1.17809 w_k1_MSFT_s005
                                 + 1.20481 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.932502 w_k1_ADM_s006
                                 + 0.780615 w_k1_AXP_s006
                                 + 1.13924 w_k1_DIS_s006
                                 + 0.987765 w_k1_FDX_s006
                                 + 1.01304 w_k1_IBM_s006
                                 + 1.10109 w_k1_INTC_s006
                                 + 1.19033 w_k1_KO_s006
                                 + 0.939994 w_k1_MAT_s006
                                 + 1.05743 w_k1_MSFT_s006
                                 + 1.09815 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.941935 w_k1_ADM_s006
                                 + 1.10283 w_k1_AXP_s006
                                 + 0.868429 w_k1_DIS_s006
                                 + 0.89136 w_k1_FDX_s006
                                 + 1.31779 w_k1_IBM_s006
                                 + 0.923988 w_k1_INTC_s006
                                 + 1.11967 w_k1_KO_s006 + 2.05676 w_k1_MAT_s006
                                 + 0.89851 w_k1_MSFT_s006
                                 + 0.928802 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.759481 w_k1_ADM_s006
                                 + 0.777935 w_k1_AXP_s006
                                 + 0.702063 w_k1_DIS_s006
                                 + 0.848616 w_k1_FDX_s006
                                 + 0.845362 w_k1_IBM_s006
                                 + 0.793302 w_k1_INTC_s006
                                 + 0.860489 w_k1_KO_s006
                                 + 0.95466 w_k1_MAT_s006
                                 + 0.817272 w_k1_MSFT_s006
                                 + 0.942959 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.09311 w_k1_ADM_s006
                                 + 0.872008 w_k1_AXP_s006
                                 + 0.700187 w_k1_DIS_s006
                                 + 0.990764 w_k1_FDX_s006
                                 + 0.603883 w_k1_IBM_s006
                                 + 0.706076 w_k1_INTC_s006
                                 + 0.772072 w_k1_KO_s006
                                 + 1.06248 w_k1_MAT_s006
                                 + 0.984535 w_k1_MSFT_s006
                                 + 0.851082 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.749663 w_k1_ADM_s006
                                 + 0.733721 w_k1_AXP_s006
                                 + 0.877616 w_k1_DIS_s006
                                 + 0.843552 w_k1_FDX_s006
                                 + 1.3204 w_k1_IBM_s006
                                 + 0.792767 w_k1_INTC_s006
                                 + 1.18073 w_k1_KO_s006 + 1.63111 w_k1_MAT_s006
                                 + 0.922843 w_k1_MSFT_s006
                                 + 0.921423 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.40537 w_k1_ADM_s006
                                 + 1.37792 w_k1_AXP_s006
                                 + 1.57143 w_k1_DIS_s006
                                 + 1.42854 w_k1_FDX_s006
                                 + 1.18527 w_k1_IBM_s006
                                 + 1.02869 w_k1_INTC_s006
                                 + 1.26862 w_k1_KO_s006 + 1.6498 w_k1_MAT_s006
                                 + 1.30397 w_k1_MSFT_s006 + 1.2999 w_k1_PG_s006
                                 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.896869 w_k1_ADM_s006
                                 + 1.18181 w_k1_AXP_s006
                                 + 1.09949 w_k1_DIS_s006
                                 + 1.09816 w_k1_FDX_s006
                                 + 1.11624 w_k1_IBM_s006
                                 + 1.00558 w_k1_INTC_s006
                                 + 1.01704 w_k1_KO_s006 + 1.53189 w_k1_MAT_s006
                                 + 1.09184 w_k1_MSFT_s006
                                 + 1.09325 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.19973 w_k1_ADM_s006
                                 + 1.12661 w_k1_AXP_s006
                                 + 1.43943 w_k1_DIS_s006
                                 + 1.22123 w_k1_FDX_s006
                                 + 1.63877 w_k1_IBM_s006
                                 + 1.10971 w_k1_INTC_s006
                                 + 1.63484 w_k1_KO_s006 + 2.13555 w_k1_MAT_s006
                                 + 1.21165 w_k1_MSFT_s006
                                 + 1.18972 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.21746 w_k1_ADM_s007
                                 + 1.36468 w_k1_AXP_s007
                                 + 1.40408 w_k1_DIS_s007
                                 + 1.13796 w_k1_FDX_s007
                                 + 1.41426 w_k1_IBM_s007
                                 + 0.875219 w_k1_INTC_s007
                                 + 1.26853 w_k1_KO_s007 + 1.56378 w_k1_MAT_s007
                                 + 1.05396 w_k1_MSFT_s007
                                 + 1.11937 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.729777 w_k1_ADM_s007
                                 + 0.789551 w_k1_AXP_s007
                                 + 0.406866 w_k1_DIS_s007
                                 + 0.684332 w_k1_FDX_s007
                                 + 0.774773 w_k1_IBM_s007
                                 + 0.679649 w_k1_INTC_s007
                                 + 0.641811 w_k1_KO_s007
                                 + 0.725325 w_k1_MAT_s007
                                 + 0.636729 w_k1_MSFT_s007
                                 + 0.700839 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.45329 w_k1_ADM_s007
                                 + 1.70118 w_k1_AXP_s007
                                 + 1.55381 w_k1_DIS_s007
                                 + 1.46407 w_k1_FDX_s007 + 1.0209 w_k1_IBM_s007
                                 + 1.19489 w_k1_INTC_s007
                                 + 1.10255 w_k1_KO_s007 + 1.4247 w_k1_MAT_s007
                                 + 1.37097 w_k1_MSFT_s007
                                 + 1.10347 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.22105 w_k1_ADM_s007
                                 + 1.11953 w_k1_AXP_s007
                                 + 1.07893 w_k1_DIS_s007
                                 + 1.10018 w_k1_FDX_s007
                                 + 0.833129 w_k1_IBM_s007
                                 + 0.983298 w_k1_INTC_s007
                                 + 1.1258 w_k1_KO_s007 + 0.967995 w_k1_MAT_s007
                                 + 1.09457 w_k1_MSFT_s007
                                 + 1.12996 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.938074 w_k1_ADM_s007
                                 + 0.750989 w_k1_AXP_s007
                                 + 0.730352 w_k1_DIS_s007
                                 + 0.829284 w_k1_FDX_s007
                                 + 0.715336 w_k1_IBM_s007
                                 + 0.785774 w_k1_INTC_s007
                                 + 0.844688 w_k1_KO_s007
                                 + 0.62216 w_k1_MAT_s007
                                 + 0.815642 w_k1_MSFT_s007
                                 + 0.86783 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.01833 w_k1_ADM_s007
                                 + 0.828445 w_k1_AXP_s007
                                 + 0.787763 w_k1_DIS_s007
                                 + 0.865098 w_k1_FDX_s007
                                 + 0.933949 w_k1_IBM_s007
                                 + 0.95854 w_k1_INTC_s007
                                 + 0.914463 w_k1_KO_s007
                                 + 0.959233 w_k1_MAT_s007
                                 + 0.784167 w_k1_MSFT_s007
                                 + 0.957316 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.879067 w_k1_ADM_s007
                                 + 0.985555 w_k1_AXP_s007
                                 + 0.820487 w_k1_DIS_s007
                                 + 0.937672 w_k1_FDX_s007
                                 + 1.11201 w_k1_IBM_s007
                                 + 1.07004 w_k1_INTC_s007
                                 + 0.993288 w_k1_KO_s007
                                 + 1.15652 w_k1_MAT_s007
                                 + 0.875475 w_k1_MSFT_s007
                                 + 1.02436 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.18472 w_k1_ADM_s007
                                 + 0.888243 w_k1_AXP_s007
                                 + 0.822781 w_k1_DIS_s007
                                 + 0.968327 w_k1_FDX_s007
                                 + 0.673477 w_k1_IBM_s007
                                 + 0.891741 w_k1_INTC_s007
                                 + 0.822182 w_k1_KO_s007
                                 + 0.745498 w_k1_MAT_s007
                                 + 0.945972 w_k1_MSFT_s007
                                 + 0.932439 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.20765 w_k1_ADM_s010
                                 + 1.02161 w_k1_AXP_s010
                                 + 1.36501 w_k1_DIS_s010
                                 + 1.21618 w_k1_FDX_s010
                                 + 1.00595 w_k1_IBM_s010
                                 + 0.887084 w_k1_INTC_s010
                                 + 1.28059 w_k1_KO_s010 + 1.3459 w_k1_MAT_s010
                                 + 1.18534 w_k1_MSFT_s010
                                 + 1.02718 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.807514 w_k1_ADM_s010
                                 + 0.601365 w_k1_AXP_s010
                                 + 0.567117 w_k1_DIS_s010
                                 + 0.765274 w_k1_FDX_s010
                                 + 0.596771 w_k1_IBM_s010
                                 + 0.619195 w_k1_INTC_s010
                                 + 0.628947 w_k1_KO_s010
                                 + 0.556891 w_k1_MAT_s010
                                 + 0.664019 w_k1_MSFT_s010
                                 + 0.74428 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.951653 w_k1_ADM_s010
                                 + 0.796637 w_k1_AXP_s010
                                 + 0.679107 w_k1_DIS_s010
                                 + 0.797798 w_k1_FDX_s010
                                 + 1.03104 w_k1_IBM_s010
                                 + 0.719325 w_k1_INTC_s010
                                 + 1.03397 w_k1_KO_s010 + 1.24039 w_k1_MAT_s010
                                 + 0.707783 w_k1_MSFT_s010
                                 + 0.915672 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.10393 w_k1_ADM_s010
                                 + 0.958092 w_k1_AXP_s010
                                 + 1.04738 w_k1_DIS_s010
                                 + 1.15627 w_k1_FDX_s010
                                 + 0.668023 w_k1_IBM_s010
                                 + 1.09576 w_k1_INTC_s010
                                 + 0.848915 w_k1_KO_s010
                                 + 0.779008 w_k1_MAT_s010
                                 + 1.05757 w_k1_MSFT_s010
                                 + 0.919085 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.2892 w_k1_ADM_s010
                                 + 1.17972 w_k1_AXP_s010
                                 + 0.992398 w_k1_DIS_s010
                                 + 1.05093 w_k1_FDX_s010
                                 + 1.29037 w_k1_IBM_s010
                                 + 0.884765 w_k1_INTC_s010
                                 + 1.16531 w_k1_KO_s010 + 1.23297 w_k1_MAT_s010
                                 + 0.994758 w_k1_MSFT_s010
                                 + 1.01914 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.01046 w_k1_ADM_s010
                                 + 0.902446 w_k1_AXP_s010
                                 + 0.944738 w_k1_DIS_s010
                                 + 1.06074 w_k1_FDX_s010
                                 + 0.848409 w_k1_IBM_s010
                                 + 0.813574 w_k1_INTC_s010
                                 + 1.09677 w_k1_KO_s010 + 1.06748 w_k1_MAT_s010
                                 + 1.00313 w_k1_MSFT_s010
                                 + 0.984244 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.25539 w_k1_ADM_s010
                                 + 1.6247 w_k1_AXP_s010 + 1.94717 w_k1_DIS_s010
                                 + 1.53566 w_k1_FDX_s010
                                 + 1.23276 w_k1_IBM_s010
                                 + 1.2371 w_k1_INTC_s010 + 1.58468 w_k1_KO_s010
                                 + 1.38796 w_k1_MAT_s010
                                 + 1.25107 w_k1_MSFT_s010
                                 + 1.25755 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.02943 w_k1_ADM_s010
                                 + 1.01773 w_k1_AXP_s010
                                 + 1.05875 w_k1_DIS_s010
                                 + 1.09688 w_k1_FDX_s010
                                 + 1.76541 w_k1_IBM_s010
                                 + 0.901821 w_k1_INTC_s010
                                 + 1.63005 w_k1_KO_s010 + 2.02141 w_k1_MAT_s010
                                 + 1.01859 w_k1_MSFT_s010
                                 + 1.18436 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.843443 w_k1_ADM_s011
                                 + 0.823013 w_k1_AXP_s011
                                 + 1.01174 w_k1_DIS_s011
                                 + 0.913797 w_k1_FDX_s011
                                 + 1.05818 w_k1_IBM_s011
                                 + 1.00645 w_k1_INTC_s011
                                 + 1.00657 w_k1_KO_s011 + 1.17826 w_k1_MAT_s011
                                 + 1.00476 w_k1_MSFT_s011
                                 + 0.960218 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.761837 w_k1_ADM_s011
                                 + 0.890578 w_k1_AXP_s011
                                 + 0.68271 w_k1_DIS_s011
                                 + 0.927205 w_k1_FDX_s011
                                 + 0.908234 w_k1_IBM_s011
                                 + 1.02373 w_k1_INTC_s011
                                 + 0.734889 w_k1_KO_s011
                                 + 1.14193 w_k1_MAT_s011
                                 + 0.910047 w_k1_MSFT_s011
                                 + 0.959341 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.93343 w_k1_ADM_s011
                                 + 1.24745 w_k1_AXP_s011
                                 + 1.43432 w_k1_DIS_s011
                                 + 1.35992 w_k1_FDX_s011
                                 + 0.833023 w_k1_IBM_s011
                                 + 1.48113 w_k1_INTC_s011
                                 + 0.984114 w_k1_KO_s011
                                 + 0.851789 w_k1_MAT_s011
                                 + 1.26714 w_k1_MSFT_s011
                                 + 1.17684 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.03173 w_k1_ADM_s011
                                 + 1.21973 w_k1_AXP_s011 + 1.0227 w_k1_DIS_s011
                                 + 1.18302 w_k1_FDX_s011
                                 + 0.974098 w_k1_IBM_s011
                                 + 0.934807 w_k1_INTC_s011
                                 + 1.01899 w_k1_KO_s011 + 1.53382 w_k1_MAT_s011
                                 + 0.968689 w_k1_MSFT_s011
                                 + 1.07237 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.731311 w_k1_ADM_s011
                                 + 1.02183 w_k1_AXP_s011
                                 + 0.701123 w_k1_DIS_s011
                                 + 0.743245 w_k1_FDX_s011
                                 + 1.32866 w_k1_IBM_s011
                                 + 0.660829 w_k1_INTC_s011
                                 + 0.824502 w_k1_KO_s011
                                 + 1.77275 w_k1_MAT_s011
                                 + 0.679861 w_k1_MSFT_s011
                                 + 0.91511 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.10127 w_k1_ADM_s011
                                 + 1.56696 w_k1_AXP_s011
                                 + 1.29406 w_k1_DIS_s011
                                 + 1.14842 w_k1_FDX_s011
                                 + 1.83573 w_k1_IBM_s011
                                 + 1.3981 w_k1_INTC_s011 + 1.25751 w_k1_KO_s011
                                 + 2.21061 w_k1_MAT_s011
                                 + 1.31312 w_k1_MSFT_s011
                                 + 1.29885 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.76093 w_k1_ADM_s011
                                 + 1.10709 w_k1_AXP_s011
                                 + 1.01659 w_k1_DIS_s011
                                 + 0.941645 w_k1_FDX_s011
                                 + 1.40842 w_k1_IBM_s011
                                 + 1.24699 w_k1_INTC_s011
                                 + 1.07819 w_k1_KO_s011 + 1.76703 w_k1_MAT_s011
                                 + 1.10731 w_k1_MSFT_s011
                                 + 1.08351 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.765097 w_k1_ADM_s011
                                 + 0.716066 w_k1_AXP_s011
                                 + 0.701156 w_k1_DIS_s011
                                 + 0.81102 w_k1_FDX_s011
                                 + 0.64558 w_k1_IBM_s011
                                 + 0.901441 w_k1_INTC_s011
                                 + 0.695681 w_k1_KO_s011
                                 + 1.05068 w_k1_MAT_s011
                                 + 0.822231 w_k1_MSFT_s011
                                 + 0.919137 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.722587 w_k1_ADM_s012
                                 + 0.847765 w_k1_AXP_s012
                                 + 0.717388 w_k1_DIS_s012
                                 + 0.827808 w_k1_FDX_s012
                                 + 1.13219 w_k1_IBM_s012
                                 + 0.818619 w_k1_INTC_s012
                                 + 1.08287 w_k1_KO_s012 + 1.2401 w_k1_MAT_s012
                                 + 0.82233 w_k1_MSFT_s012
                                 + 0.899602 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.706989 w_k1_ADM_s012
                                 + 0.712355 w_k1_AXP_s012
                                 + 0.756091 w_k1_DIS_s012
                                 + 0.838431 w_k1_FDX_s012
                                 + 0.819248 w_k1_IBM_s012
                                 + 0.851961 w_k1_INTC_s012
                                 + 0.895554 w_k1_KO_s012
                                 + 0.775794 w_k1_MAT_s012
                                 + 0.947658 w_k1_MSFT_s012
                                 + 0.928735 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.06892 w_k1_ADM_s012
                                 + 0.917315 w_k1_AXP_s012
                                 + 1.32656 w_k1_DIS_s012 + 1.0141 w_k1_FDX_s012
                                 + 1.12124 w_k1_IBM_s012
                                 + 0.970376 w_k1_INTC_s012
                                 + 1.29659 w_k1_KO_s012 + 1.0159 w_k1_MAT_s012
                                 + 1.1832 w_k1_MSFT_s012 + 1.10499 w_k1_PG_s012
                                 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.01249 w_k1_ADM_s012
                                 + 1.0441 w_k1_AXP_s012 + 1.10188 w_k1_DIS_s012
                                 + 1.12612 w_k1_FDX_s012
                                 + 0.892213 w_k1_IBM_s012
                                 + 0.976952 w_k1_INTC_s012
                                 + 1.11713 w_k1_KO_s012 + 1.28199 w_k1_MAT_s012
                                 + 1.15288 w_k1_MSFT_s012
                                 + 1.02158 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.07048 w_k1_ADM_s012
                                 + 1.22713 w_k1_AXP_s012
                                 + 1.04726 w_k1_DIS_s012 + 1.044 w_k1_FDX_s012
                                 + 1.28897 w_k1_IBM_s012
                                 + 1.00606 w_k1_INTC_s012
                                 + 1.29346 w_k1_KO_s012 + 1.48476 w_k1_MAT_s012
                                 + 1.0427 w_k1_MSFT_s012 + 1.05022 w_k1_PG_s012
                                 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.918071 w_k1_ADM_s012
                                 + 1.05162 w_k1_AXP_s012
                                 + 0.761865 w_k1_DIS_s012
                                 + 0.926184 w_k1_FDX_s012
                                 + 0.791481 w_k1_IBM_s012
                                 + 1.03979 w_k1_INTC_s012
                                 + 0.806199 w_k1_KO_s012
                                 + 1.10645 w_k1_MAT_s012
                                 + 0.884205 w_k1_MSFT_s012
                                 + 0.844672 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.945185 w_k1_ADM_s012
                                 + 1.08381 w_k1_AXP_s012 + 1.3297 w_k1_DIS_s012
                                 + 1.02885 w_k1_FDX_s012
                                 + 1.43206 w_k1_IBM_s012
                                 + 1.16903 w_k1_INTC_s012
                                 + 1.55279 w_k1_KO_s012 + 1.58112 w_k1_MAT_s012
                                 + 1.17543 w_k1_MSFT_s012
                                 + 1.11498 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.2474 w_k1_ADM_s012
                                 + 1.47873 w_k1_AXP_s012
                                 + 1.50073 w_k1_DIS_s012
                                 + 1.33686 w_k1_FDX_s012
                                 + 1.30995 w_k1_IBM_s012
                                 + 1.24842 w_k1_INTC_s012
                                 + 1.52094 w_k1_KO_s012 + 1.41609 w_k1_MAT_s012
                                 + 1.25225 w_k1_MSFT_s012
                                 + 1.22835 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.19277 w_k1_ADM_s013
                                 + 1.60523 w_k1_AXP_s013
                                 + 1.46865 w_k1_DIS_s013
                                 + 1.45834 w_k1_FDX_s013
                                 + 1.10584 w_k1_IBM_s013
                                 + 1.44071 w_k1_INTC_s013
                                 + 1.24246 w_k1_KO_s013 + 1.12809 w_k1_MAT_s013
                                 + 1.37386 w_k1_MSFT_s013
                                 + 1.31205 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.02174 w_k1_ADM_s013
                                 + 1.03077 w_k1_AXP_s013
                                 + 0.851962 w_k1_DIS_s013
                                 + 1.01665 w_k1_FDX_s013
                                 + 0.82896 w_k1_IBM_s013
                                 + 1.12651 w_k1_INTC_s013
                                 + 0.84211 w_k1_KO_s013
                                 + 0.565526 w_k1_MAT_s013
                                 + 1.04272 w_k1_MSFT_s013
                                 + 0.951722 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.07339 w_k1_ADM_s013
                                 + 0.985092 w_k1_AXP_s013
                                 + 1.12173 w_k1_DIS_s013
                                 + 1.06707 w_k1_FDX_s013 + 1.2917 w_k1_IBM_s013
                                 + 0.998751 w_k1_INTC_s013
                                 + 1.04723 w_k1_KO_s013 + 1.47056 w_k1_MAT_s013
                                 + 1.03434 w_k1_MSFT_s013
                                 + 1.12319 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.720063 w_k1_ADM_s013
                                 + 0.796269 w_k1_AXP_s013
                                 + 0.65402 w_k1_DIS_s013
                                 + 0.772436 w_k1_FDX_s013
                                 + 1.08693 w_k1_IBM_s013
                                 + 0.785898 w_k1_INTC_s013
                                 + 0.79946 w_k1_KO_s013
                                 + 0.918607 w_k1_MAT_s013
                                 + 0.755855 w_k1_MSFT_s013
                                 + 0.983819 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.32921 w_k1_ADM_s013
                                 + 1.18017 w_k1_AXP_s013
                                 + 1.48018 w_k1_DIS_s013
                                 + 1.42813 w_k1_FDX_s013
                                 + 0.943879 w_k1_IBM_s013
                                 + 1.05617 w_k1_INTC_s013
                                 + 1.10405 w_k1_KO_s013 + 1.20764 w_k1_MAT_s013
                                 + 1.41851 w_k1_MSFT_s013 + 1.2983 w_k1_PG_s013
                                 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.997882 w_k1_ADM_s013
                                 + 0.970475 w_k1_AXP_s013
                                 + 1.10741 w_k1_DIS_s013
                                 + 1.13058 w_k1_FDX_s013
                                 + 0.674206 w_k1_IBM_s013
                                 + 1.10138 w_k1_INTC_s013
                                 + 0.801506 w_k1_KO_s013
                                 + 0.720081 w_k1_MAT_s013
                                 + 1.22014 w_k1_MSFT_s013
                                 + 1.07714 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.612128 w_k1_ADM_s013
                                 + 0.584874 w_k1_AXP_s013
                                 + 0.552558 w_k1_DIS_s013
                                 + 0.737964 w_k1_FDX_s013
                                 + 0.673132 w_k1_IBM_s013
                                 + 0.836258 w_k1_INTC_s013
                                 + 0.665009 w_k1_KO_s013
                                 + 0.555031 w_k1_MAT_s013
                                 + 0.786904 w_k1_MSFT_s013
                                 + 0.80461 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.808119 w_k1_ADM_s013
                                 + 0.697748 w_k1_AXP_s013
                                 + 0.755753 w_k1_DIS_s013
                                 + 0.814074 w_k1_FDX_s013
                                 + 0.936187 w_k1_IBM_s013
                                 + 0.922674 w_k1_INTC_s013
                                 + 0.897497 w_k1_KO_s013
                                 + 0.57622 w_k1_MAT_s013
                                 + 0.952832 w_k1_MSFT_s013
                                 + 1.07966 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.927347 w_k1_ADM_s014
                                 + 1.35469 w_k1_AXP_s014
                                 + 1.58566 w_k1_DIS_s014
                                 + 1.26421 w_k1_FDX_s014
                                 + 1.13159 w_k1_IBM_s014
                                 + 1.09483 w_k1_INTC_s014
                                 + 1.14396 w_k1_KO_s014 + 1.17104 w_k1_MAT_s014
                                 + 1.0922 w_k1_MSFT_s014 + 1.0528 w_k1_PG_s014
                                 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.2522 w_k1_ADM_s014
                                 + 1.70672 w_k1_AXP_s014
                                 + 1.05567 w_k1_DIS_s014
                                 + 1.09573 w_k1_FDX_s014
                                 + 1.26175 w_k1_IBM_s014
                                 + 1.05675 w_k1_INTC_s014
                                 + 1.16886 w_k1_KO_s014 + 2.19581 w_k1_MAT_s014
                                 + 0.955156 w_k1_MSFT_s014
                                 + 1.08901 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.921264 w_k1_ADM_s014
                                 + 0.992706 w_k1_AXP_s014
                                 + 0.799111 w_k1_DIS_s014
                                 + 0.932378 w_k1_FDX_s014
                                 + 0.698667 w_k1_IBM_s014
                                 + 0.805923 w_k1_INTC_s014
                                 + 0.764741 w_k1_KO_s014
                                 + 0.950251 w_k1_MAT_s014
                                 + 0.820432 w_k1_MSFT_s014
                                 + 0.86909 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.751885 w_k1_ADM_s014
                                 + 0.819945 w_k1_AXP_s014
                                 + 0.899178 w_k1_DIS_s014
                                 + 0.88106 w_k1_FDX_s014
                                 + 0.747628 w_k1_IBM_s014
                                 + 0.840556 w_k1_INTC_s014
                                 + 0.826023 w_k1_KO_s014
                                 + 0.837977 w_k1_MAT_s014
                                 + 0.894894 w_k1_MSFT_s014
                                 + 0.8763 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.0378 w_k1_ADM_s014
                                 + 1.11482 w_k1_AXP_s014
                                 + 1.13728 w_k1_DIS_s014
                                 + 0.980424 w_k1_FDX_s014
                                 + 1.23598 w_k1_IBM_s014
                                 + 0.876259 w_k1_INTC_s014
                                 + 1.13386 w_k1_KO_s014 + 1.15971 w_k1_MAT_s014
                                 + 0.988565 w_k1_MSFT_s014
                                 + 0.913551 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.938378 w_k1_ADM_s014
                                 + 0.885511 w_k1_AXP_s014
                                 + 0.917641 w_k1_DIS_s014
                                 + 0.856172 w_k1_FDX_s014
                                 + 1.08086 w_k1_IBM_s014
                                 + 0.773525 w_k1_INTC_s014
                                 + 0.817699 w_k1_KO_s014
                                 + 0.94786 w_k1_MAT_s014
                                 + 0.870479 w_k1_MSFT_s014
                                 + 0.901014 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.24971 w_k1_ADM_s014
                                 + 1.22224 w_k1_AXP_s014
                                 + 1.24716 w_k1_DIS_s014
                                 + 1.25397 w_k1_FDX_s014
                                 + 0.768545 w_k1_IBM_s014
                                 + 0.867728 w_k1_INTC_s014
                                 + 0.935505 w_k1_KO_s014
                                 + 1.05961 w_k1_MAT_s014
                                 + 1.13466 w_k1_MSFT_s014
                                 + 0.982163 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.619447 w_k1_ADM_s014
                                 + 0.844063 w_k1_AXP_s014
                                 + 0.559983 w_k1_DIS_s014
                                 + 0.747927 w_k1_FDX_s014
                                 + 0.893925 w_k1_IBM_s014
                                 + 0.7787 w_k1_INTC_s014
                                 + 0.732501 w_k1_KO_s014
                                 + 1.11661 w_k1_MAT_s014
                                 + 0.733743 w_k1_MSFT_s014
                                 + 0.769301 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.54381 w_k1_ADM_s015
                                 + 1.87606 w_k1_AXP_s015
                                 + 1.53291 w_k1_DIS_s015
                                 + 1.30403 w_k1_FDX_s015
                                 + 1.55534 w_k1_IBM_s015
                                 + 1.45013 w_k1_INTC_s015
                                 + 1.41062 w_k1_KO_s015 + 1.71247 w_k1_MAT_s015
                                 + 1.41438 w_k1_MSFT_s015 + 1.1857 w_k1_PG_s015
                                 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.05029 w_k1_ADM_s015
                                 + 0.913626 w_k1_AXP_s015
                                 + 1.04217 w_k1_DIS_s015
                                 + 0.963349 w_k1_FDX_s015
                                 + 1.00632 w_k1_IBM_s015
                                 + 1.06846 w_k1_INTC_s015
                                 + 0.975348 w_k1_KO_s015
                                 + 0.976636 w_k1_MAT_s015
                                 + 1.10894 w_k1_MSFT_s015
                                 + 0.954413 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.94077 w_k1_ADM_s015
                                 + 0.707255 w_k1_AXP_s015
                                 + 0.59088 w_k1_DIS_s015
                                 + 0.750269 w_k1_FDX_s015
                                 + 0.882473 w_k1_IBM_s015
                                 + 0.788132 w_k1_INTC_s015
                                 + 0.7236 w_k1_KO_s015 + 0.93879 w_k1_MAT_s015
                                 + 0.792811 w_k1_MSFT_s015
                                 + 0.718736 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.03577 w_k1_ADM_s015
                                 + 0.822334 w_k1_AXP_s015
                                 + 0.813475 w_k1_DIS_s015
                                 + 0.99315 w_k1_FDX_s015
                                 + 0.753386 w_k1_IBM_s015
                                 + 0.991625 w_k1_INTC_s015
                                 + 0.787938 w_k1_KO_s015
                                 + 0.694458 w_k1_MAT_s015
                                 + 0.935132 w_k1_MSFT_s015
                                 + 0.788546 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.37741 w_k1_ADM_s015
                                 + 1.13008 w_k1_AXP_s015
                                 + 1.40724 w_k1_DIS_s015
                                 + 1.22335 w_k1_FDX_s015
                                 + 1.05566 w_k1_IBM_s015
                                 + 1.14301 w_k1_INTC_s015
                                 + 1.07002 w_k1_KO_s015 + 1.13645 w_k1_MAT_s015
                                 + 1.13 w_k1_MSFT_s015 + 0.950512 w_k1_PG_s015
                                 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.05284 w_k1_ADM_s015
                                 + 1.31591 w_k1_AXP_s015
                                 + 0.914127 w_k1_DIS_s015
                                 + 0.931853 w_k1_FDX_s015
                                 + 1.54235 w_k1_IBM_s015
                                 + 1.10069 w_k1_INTC_s015
                                 + 1.02029 w_k1_KO_s015 + 1.40241 w_k1_MAT_s015
                                 + 1.03326 w_k1_MSFT_s015
                                 + 0.954086 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.35735 w_k1_ADM_s015
                                 + 1.12338 w_k1_AXP_s015
                                 + 1.03864 w_k1_DIS_s015
                                 + 0.998329 w_k1_FDX_s015
                                 + 1.27246 w_k1_IBM_s015
                                 + 1.11863 w_k1_INTC_s015
                                 + 1.21397 w_k1_KO_s015
                                 + 0.964485 w_k1_MAT_s015
                                 + 1.13969 w_k1_MSFT_s015
                                 + 1.01194 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.10114 w_k1_ADM_s015
                                 + 1.23011 w_k1_AXP_s015
                                 + 0.984596 w_k1_DIS_s015
                                 + 1.1603 w_k1_FDX_s015
                                 + 0.932299 w_k1_IBM_s015
                                 + 1.08678 w_k1_INTC_s015
                                 + 0.879632 w_k1_KO_s015
                                 + 0.89967 w_k1_MAT_s015
                                 + 1.04206 w_k1_MSFT_s015
                                 + 0.858351 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.27932 w_k1_ADM_s016
                                 + 1.42441 w_k1_AXP_s016
                                 + 1.46009 w_k1_DIS_s016
                                 + 1.25336 w_k1_FDX_s016 + 1.5471 w_k1_IBM_s016
                                 + 1.09289 w_k1_INTC_s016
                                 + 1.20294 w_k1_KO_s016 + 1.3357 w_k1_MAT_s016
                                 + 1.04391 w_k1_MSFT_s016
                                 + 1.16494 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.19706 w_k1_ADM_s016
                                 + 0.845894 w_k1_AXP_s016
                                 + 0.891417 w_k1_DIS_s016
                                 + 0.982373 w_k1_FDX_s016
                                 + 0.792568 w_k1_IBM_s016
                                 + 0.825901 w_k1_INTC_s016
                                 + 0.771212 w_k1_KO_s016
                                 + 0.708631 w_k1_MAT_s016
                                 + 0.889423 w_k1_MSFT_s016
                                 + 0.922681 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.907249 w_k1_ADM_s016
                                 + 0.713446 w_k1_AXP_s016
                                 + 0.605549 w_k1_DIS_s016
                                 + 0.773769 w_k1_FDX_s016
                                 + 0.752044 w_k1_IBM_s016
                                 + 0.761748 w_k1_INTC_s016
                                 + 0.727005 w_k1_KO_s016
                                 + 0.520209 w_k1_MAT_s016
                                 + 0.717064 w_k1_MSFT_s016
                                 + 0.80466 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.29415 w_k1_ADM_s016
                                 + 0.922681 w_k1_AXP_s016
                                 + 1.30101 w_k1_DIS_s016
                                 + 1.16853 w_k1_FDX_s016
                                 + 0.785264 w_k1_IBM_s016
                                 + 0.92258 w_k1_INTC_s016 + 1.0371 w_k1_KO_s016
                                 + 0.81231 w_k1_MAT_s016
                                 + 1.13776 w_k1_MSFT_s016
                                 + 1.04403 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.29938 w_k1_ADM_s016
                                 + 1.23654 w_k1_AXP_s016
                                 + 1.28948 w_k1_DIS_s016
                                 + 1.26466 w_k1_FDX_s016
                                 + 0.845564 w_k1_IBM_s016
                                 + 1.40616 w_k1_INTC_s016 + 1.0787 w_k1_KO_s016
                                 + 0.885851 w_k1_MAT_s016
                                 + 1.27481 w_k1_MSFT_s016
                                 + 1.10609 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.16203 w_k1_ADM_s016
                                 + 1.28266 w_k1_AXP_s016
                                 + 0.947152 w_k1_DIS_s016
                                 + 0.928485 w_k1_FDX_s016
                                 + 1.32164 w_k1_IBM_s016
                                 + 1.0213 w_k1_INTC_s016
                                 + 0.933871 w_k1_KO_s016
                                 + 1.39308 w_k1_MAT_s016
                                 + 0.890471 w_k1_MSFT_s016
                                 + 0.87754 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.40739 w_k1_ADM_s016
                                 + 0.93221 w_k1_AXP_s016
                                 + 1.14101 w_k1_DIS_s016
                                 + 0.966822 w_k1_FDX_s016
                                 + 1.17134 w_k1_IBM_s016
                                 + 1.12328 w_k1_INTC_s016
                                 + 1.20054 w_k1_KO_s016
                                 + 0.786803 w_k1_MAT_s016
                                 + 0.947489 w_k1_MSFT_s016
                                 + 1.0303 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.837323 w_k1_ADM_s016
                                 + 0.793463 w_k1_AXP_s016
                                 + 0.765454 w_k1_DIS_s016
                                 + 0.803018 w_k1_FDX_s016
                                 + 1.21454 w_k1_IBM_s016
                                 + 0.877088 w_k1_INTC_s016
                                 + 0.900756 w_k1_KO_s016
                                 + 1.03826 w_k1_MAT_s016
                                 + 0.80671 w_k1_MSFT_s016
                                 + 0.825247 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.42813 w_k1_ADM_s017
                                 + 1.42623 w_k1_AXP_s017
                                 + 1.62203 w_k1_DIS_s017
                                 + 1.31883 w_k1_FDX_s017
                                 + 1.05379 w_k1_IBM_s017
                                 + 1.45266 w_k1_INTC_s017
                                 + 1.45392 w_k1_KO_s017 + 1.28471 w_k1_MAT_s017
                                 + 1.37048 w_k1_MSFT_s017
                                 + 1.13901 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.905972 w_k1_ADM_s017
                                 + 0.689193 w_k1_AXP_s017
                                 + 0.8143 w_k1_DIS_s017
                                 + 0.901227 w_k1_FDX_s017
                                 + 0.650671 w_k1_IBM_s017
                                 + 1.00744 w_k1_INTC_s017
                                 + 0.863556 w_k1_KO_s017
                                 + 0.564147 w_k1_MAT_s017
                                 + 0.941123 w_k1_MSFT_s017
                                 + 0.865127 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.10126 w_k1_ADM_s017
                                 + 0.790346 w_k1_AXP_s017
                                 + 0.729946 w_k1_DIS_s017
                                 + 0.814679 w_k1_FDX_s017
                                 + 1.11851 w_k1_IBM_s017
                                 + 0.843232 w_k1_INTC_s017
                                 + 1.01053 w_k1_KO_s017 + 1.39624 w_k1_MAT_s017
                                 + 0.755877 w_k1_MSFT_s017
                                 + 0.89956 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.79091 w_k1_ADM_s017
                                 + 0.732965 w_k1_AXP_s017
                                 + 0.595422 w_k1_DIS_s017
                                 + 0.732821 w_k1_FDX_s017
                                 + 0.82071 w_k1_IBM_s017
                                 + 0.763228 w_k1_INTC_s017
                                 + 0.802951 w_k1_KO_s017
                                 + 0.960667 w_k1_MAT_s017
                                 + 0.722952 w_k1_MSFT_s017
                                 + 0.837588 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.08204 w_k1_ADM_s017
                                 + 0.984682 w_k1_AXP_s017
                                 + 1.16993 w_k1_DIS_s017
                                 + 1.09429 w_k1_FDX_s017
                                 + 0.978043 w_k1_IBM_s017
                                 + 1.22762 w_k1_INTC_s017
                                 + 1.25116 w_k1_KO_s017 + 1.18316 w_k1_MAT_s017
                                 + 1.13025 w_k1_MSFT_s017
                                 + 0.958886 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.927702 w_k1_ADM_s017
                                 + 1.00621 w_k1_AXP_s017
                                 + 1.29645 w_k1_DIS_s017
                                 + 1.00674 w_k1_FDX_s017
                                 + 1.50388 w_k1_IBM_s017
                                 + 1.0186 w_k1_INTC_s017 + 1.39825 w_k1_KO_s017
                                 + 2.37051 w_k1_MAT_s017
                                 + 1.03389 w_k1_MSFT_s017
                                 + 1.13421 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.790055 w_k1_ADM_s017
                                 + 0.713215 w_k1_AXP_s017
                                 + 0.63367 w_k1_DIS_s017
                                 + 0.68947 w_k1_FDX_s017
                                 + 1.10702 w_k1_IBM_s017
                                 + 0.913403 w_k1_INTC_s017
                                 + 0.87486 w_k1_KO_s017 + 1.09376 w_k1_MAT_s017
                                 + 0.917337 w_k1_MSFT_s017
                                 + 0.85002 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.03616 w_k1_ADM_s017
                                 + 0.996338 w_k1_AXP_s017
                                 + 0.920105 w_k1_DIS_s017
                                 + 0.905654 w_k1_FDX_s017
                                 + 1.05313 w_k1_IBM_s017
                                 + 1.0468 w_k1_INTC_s017 + 1.13722 w_k1_KO_s017
                                 + 1.91592 w_k1_MAT_s017
                                 + 0.976724 w_k1_MSFT_s017
                                 + 0.977099 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.23548 w_k1_ADM_s100
                                 + 1.04131 w_k1_AXP_s100
                                 + 0.881774 w_k1_DIS_s100
                                 + 1.10861 w_k1_FDX_s100
                                 + 0.637712 w_k1_IBM_s100
                                 + 0.869117 w_k1_INTC_s100
                                 + 0.775158 w_k1_KO_s100
                                 + 0.533113 w_k1_MAT_s100
                                 + 1.01138 w_k1_MSFT_s100
                                 + 0.988187 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.894503 w_k1_ADM_s100
                                 + 0.794351 w_k1_AXP_s100
                                 + 0.633207 w_k1_DIS_s100
                                 + 0.814191 w_k1_FDX_s100
                                 + 0.873684 w_k1_IBM_s100
                                 + 1.0075 w_k1_INTC_s100
                                 + 0.727471 w_k1_KO_s100
                                 + 0.811168 w_k1_MAT_s100
                                 + 0.786688 w_k1_MSFT_s100
                                 + 0.865036 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.0312 w_k1_ADM_s100
                                 + 1.10716 w_k1_AXP_s100 + 0.9561 w_k1_DIS_s100
                                 + 1.05851 w_k1_FDX_s100
                                 + 0.95085 w_k1_IBM_s100
                                 + 1.04622 w_k1_INTC_s100
                                 + 1.02233 w_k1_KO_s100 + 1.27038 w_k1_MAT_s100
                                 + 1.117 w_k1_MSFT_s100 + 1.07126 w_k1_PG_s100
                                 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.967768 w_k1_ADM_s100
                                 + 1.12629 w_k1_AXP_s100
                                 + 1.17267 w_k1_DIS_s100
                                 + 1.30725 w_k1_FDX_s100
                                 + 0.723065 w_k1_IBM_s100
                                 + 1.05045 w_k1_INTC_s100
                                 + 0.87547 w_k1_KO_s100
                                 + 0.847466 w_k1_MAT_s100
                                 + 1.13969 w_k1_MSFT_s100
                                 + 1.18723 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.18963 w_k1_ADM_s100
                                 + 1.37367 w_k1_AXP_s100 + 1.2461 w_k1_DIS_s100
                                 + 1.10029 w_k1_FDX_s100
                                 + 1.70134 w_k1_IBM_s100
                                 + 1.11385 w_k1_INTC_s100
                                 + 1.50707 w_k1_KO_s100 + 2.05454 w_k1_MAT_s100
                                 + 1.1558 w_k1_MSFT_s100 + 1.32588 w_k1_PG_s100
                                 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.866549 w_k1_ADM_s100
                                 + 1.07314 w_k1_AXP_s100
                                 + 0.797056 w_k1_DIS_s100
                                 + 0.837876 w_k1_FDX_s100
                                 + 1.24311 w_k1_IBM_s100
                                 + 0.959452 w_k1_INTC_s100
                                 + 0.92997 w_k1_KO_s100 + 1.11293 w_k1_MAT_s100
                                 + 0.963013 w_k1_MSFT_s100
                                 + 1.1033 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.977107 w_k1_ADM_s100
                                 + 1.30244 w_k1_AXP_s100
                                 + 1.34346 w_k1_DIS_s100 + 1.3225 w_k1_FDX_s100
                                 + 1.14818 w_k1_IBM_s100
                                 + 1.12438 w_k1_INTC_s100
                                 + 1.30643 w_k1_KO_s100 + 1.07764 w_k1_MAT_s100
                                 + 1.21461 w_k1_MSFT_s100
                                 + 1.27342 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.649641 w_k1_ADM_s100
                                 + 0.781893 w_k1_AXP_s100
                                 + 0.564386 w_k1_DIS_s100
                                 + 0.878172 w_k1_FDX_s100
                                 + 0.614664 w_k1_IBM_s100
                                 + 0.85063 w_k1_INTC_s100
                                 + 0.675492 w_k1_KO_s100
                                 + 0.810447 w_k1_MAT_s100
                                 + 0.936815 w_k1_MSFT_s100
                                 + 0.816539 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.726109 w_k1_ADM_s101
                                 + 0.807731 w_k1_AXP_s101
                                 + 1.00951 w_k1_DIS_s101
                                 + 0.888252 w_k1_FDX_s101
                                 + 1.05384 w_k1_IBM_s101
                                 + 1.10518 w_k1_INTC_s101 + 1.0801 w_k1_KO_s101
                                 + 0.910822 w_k1_MAT_s101
                                 + 1.0763 w_k1_MSFT_s101
                                 + 0.983635 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.92762 w_k1_ADM_s101
                                 + 0.616587 w_k1_AXP_s101
                                 + 0.620436 w_k1_DIS_s101
                                 + 0.895749 w_k1_FDX_s101
                                 + 0.636622 w_k1_IBM_s101
                                 + 0.955641 w_k1_INTC_s101
                                 + 0.772398 w_k1_KO_s101
                                 + 0.682427 w_k1_MAT_s101
                                 + 0.86183 w_k1_MSFT_s101
                                 + 0.868131 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.20434 w_k1_ADM_s101
                                 + 1.04834 w_k1_AXP_s101
                                 + 1.15345 w_k1_DIS_s101
                                 + 1.01757 w_k1_FDX_s101
                                 + 1.46157 w_k1_IBM_s101
                                 + 0.91438 w_k1_INTC_s101
                                 + 1.41482 w_k1_KO_s101 + 1.50736 w_k1_MAT_s101
                                 + 1.07569 w_k1_MSFT_s101 + 1.2472 w_k1_PG_s101
                                 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.39463 w_k1_ADM_s101
                                 + 1.09451 w_k1_AXP_s101
                                 + 1.03151 w_k1_DIS_s101
                                 + 1.18988 w_k1_FDX_s101
                                 + 1.07841 w_k1_IBM_s101
                                 + 1.07354 w_k1_INTC_s101
                                 + 1.22777 w_k1_KO_s101
                                 + 0.955464 w_k1_MAT_s101
                                 + 1.14773 w_k1_MSFT_s101
                                 + 1.04114 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.14117 w_k1_ADM_s101
                                 + 1.23063 w_k1_AXP_s101
                                 + 1.50197 w_k1_DIS_s101
                                 + 1.33446 w_k1_FDX_s101
                                 + 1.44452 w_k1_IBM_s101
                                 + 1.1637 w_k1_INTC_s101 + 1.45934 w_k1_KO_s101
                                 + 1.24284 w_k1_MAT_s101
                                 + 1.26627 w_k1_MSFT_s101
                                 + 1.44315 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.19725 w_k1_ADM_s101
                                 + 1.04328 w_k1_AXP_s101 + 1.0537 w_k1_DIS_s101
                                 + 1.22824 w_k1_FDX_s101
                                 + 0.920875 w_k1_IBM_s101
                                 + 1.19158 w_k1_INTC_s101
                                 + 1.12916 w_k1_KO_s101 + 1.27528 w_k1_MAT_s101
                                 + 1.09906 w_k1_MSFT_s101
                                 + 1.18232 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.06769 w_k1_ADM_s101
                                 + 0.774378 w_k1_AXP_s101
                                 + 0.822501 w_k1_DIS_s101
                                 + 0.869026 w_k1_FDX_s101
                                 + 0.754855 w_k1_IBM_s101
                                 + 0.984601 w_k1_INTC_s101
                                 + 0.944922 w_k1_KO_s101
                                 + 0.795944 w_k1_MAT_s101
                                 + 0.967476 w_k1_MSFT_s101
                                 + 0.993921 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.947857 w_k1_ADM_s101
                                 + 0.893381 w_k1_AXP_s101
                                 + 0.790418 w_k1_DIS_s101
                                 + 0.905542 w_k1_FDX_s101
                                 + 0.980286 w_k1_IBM_s101
                                 + 0.74799 w_k1_INTC_s101
                                 + 0.91954 w_k1_KO_s101 + 1.23311 w_k1_MAT_s101
                                 + 0.854005 w_k1_MSFT_s101
                                 + 0.950981 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.16096 w_k1_ADM_s102
                                 + 1.06846 w_k1_AXP_s102
                                 + 1.17181 w_k1_DIS_s102
                                 + 1.11986 w_k1_FDX_s102
                                 + 1.14918 w_k1_IBM_s102
                                 + 1.08425 w_k1_INTC_s102
                                 + 1.25474 w_k1_KO_s102 + 1.32328 w_k1_MAT_s102
                                 + 1.0845 w_k1_MSFT_s102 + 1.2162 w_k1_PG_s102
                                 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.989455 w_k1_ADM_s102
                                 + 0.899293 w_k1_AXP_s102
                                 + 0.673744 w_k1_DIS_s102
                                 + 0.757258 w_k1_FDX_s102
                                 + 1.06339 w_k1_IBM_s102
                                 + 0.671087 w_k1_INTC_s102
                                 + 0.988756 w_k1_KO_s102
                                 + 1.41032 w_k1_MAT_s102
                                 + 0.760117 w_k1_MSFT_s102
                                 + 0.8835 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.03562 w_k1_ADM_s102
                                 + 0.816157 w_k1_AXP_s102
                                 + 1.03995 w_k1_DIS_s102 + 1.0696 w_k1_FDX_s102
                                 + 0.753536 w_k1_IBM_s102
                                 + 0.934994 w_k1_INTC_s102
                                 + 1.14345 w_k1_KO_s102
                                 + 0.851159 w_k1_MAT_s102
                                 + 1.15867 w_k1_MSFT_s102
                                 + 0.943548 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.813918 w_k1_ADM_s102
                                 + 1.08387 w_k1_AXP_s102
                                 + 0.889349 w_k1_DIS_s102
                                 + 0.819486 w_k1_FDX_s102
                                 + 1.31112 w_k1_IBM_s102 + 1.07 w_k1_INTC_s102
                                 + 1.32435 w_k1_KO_s102 + 1.96893 w_k1_MAT_s102
                                 + 1.13367 w_k1_MSFT_s102
                                 + 1.04778 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.992944 w_k1_ADM_s102
                                 + 1.2548 w_k1_AXP_s102 + 1.38984 w_k1_DIS_s102
                                 + 1.06398 w_k1_FDX_s102
                                 + 1.54975 w_k1_IBM_s102
                                 + 0.958165 w_k1_INTC_s102
                                 + 1.38123 w_k1_KO_s102 + 1.54153 w_k1_MAT_s102
                                 + 1.16682 w_k1_MSFT_s102
                                 + 1.11424 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.09247 w_k1_ADM_s102
                                 + 0.84182 w_k1_AXP_s102
                                 + 0.83654 w_k1_DIS_s102
                                 + 0.883847 w_k1_FDX_s102
                                 + 0.869715 w_k1_IBM_s102
                                 + 0.864545 w_k1_INTC_s102
                                 + 0.886864 w_k1_KO_s102
                                 + 1.00425 w_k1_MAT_s102
                                 + 0.923921 w_k1_MSFT_s102
                                 + 0.952818 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.944416 w_k1_ADM_s102
                                 + 0.776935 w_k1_AXP_s102
                                 + 1.03307 w_k1_DIS_s102
                                 + 0.964726 w_k1_FDX_s102
                                 + 1.1653 w_k1_IBM_s102
                                 + 1.02106 w_k1_INTC_s102
                                 + 1.10988 w_k1_KO_s102 + 1.25862 w_k1_MAT_s102
                                 + 1.09173 w_k1_MSFT_s102
                                 + 1.12241 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.5621 w_k1_ADM_s102
                                 + 1.30093 w_k1_AXP_s102
                                 + 1.63757 w_k1_DIS_s102
                                 + 1.31167 w_k1_FDX_s102
                                 + 1.10624 w_k1_IBM_s102
                                 + 1.39216 w_k1_INTC_s102
                                 + 1.50854 w_k1_KO_s102 + 1.36283 w_k1_MAT_s102
                                 + 1.55684 w_k1_MSFT_s102
                                 + 1.25116 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.856872 w_k1_ADM_s103
                                 + 0.640133 w_k1_AXP_s103
                                 + 0.664612 w_k1_DIS_s103
                                 + 0.78488 w_k1_FDX_s103
                                 + 0.642544 w_k1_IBM_s103
                                 + 0.98287 w_k1_INTC_s103
                                 + 0.718614 w_k1_KO_s103
                                 + 0.713116 w_k1_MAT_s103
                                 + 0.835676 w_k1_MSFT_s103
                                 + 0.837613 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.42565 w_k1_ADM_s103
                                 + 1.24866 w_k1_AXP_s103
                                 + 1.25998 w_k1_DIS_s103
                                 + 1.26241 w_k1_FDX_s103
                                 + 0.935469 w_k1_IBM_s103
                                 + 1.28985 w_k1_INTC_s103
                                 + 1.07069 w_k1_KO_s103 + 1.20063 w_k1_MAT_s103
                                 + 1.12699 w_k1_MSFT_s103
                                 + 1.03156 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.67945 w_k1_ADM_s103
                                 + 1.45483 w_k1_AXP_s103
                                 + 2.05958 w_k1_DIS_s103
                                 + 1.69904 w_k1_FDX_s103
                                 + 1.30565 w_k1_IBM_s103
                                 + 1.24487 w_k1_INTC_s103
                                 + 1.41947 w_k1_KO_s103 + 1.07574 w_k1_MAT_s103
                                 + 1.41408 w_k1_MSFT_s103
                                 + 1.17287 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.8792 w_k1_ADM_s103
                                 + 0.765594 w_k1_AXP_s103
                                 + 1.40176 w_k1_DIS_s103
                                 + 1.06869 w_k1_FDX_s103
                                 + 1.11333 w_k1_IBM_s103
                                 + 1.02948 w_k1_INTC_s103
                                 + 1.06495 w_k1_KO_s103 + 1.47524 w_k1_MAT_s103
                                 + 1.05242 w_k1_MSFT_s103
                                 + 1.07649 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.04079 w_k1_ADM_s103
                                 + 0.874441 w_k1_AXP_s103
                                 + 1.03605 w_k1_DIS_s103 + 1.1276 w_k1_FDX_s103
                                 + 0.679887 w_k1_IBM_s103
                                 + 1.0864 w_k1_INTC_s103
                                 + 0.824948 w_k1_KO_s103
                                 + 0.730973 w_k1_MAT_s103
                                 + 1.11623 w_k1_MSFT_s103
                                 + 1.05425 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.999877 w_k1_ADM_s103
                                 + 0.980748 w_k1_AXP_s103
                                 + 0.993858 w_k1_DIS_s103
                                 + 0.970161 w_k1_FDX_s103
                                 + 0.978536 w_k1_IBM_s103
                                 + 1.07179 w_k1_INTC_s103 + 1.0813 w_k1_KO_s103
                                 + 1.21658 w_k1_MAT_s103
                                 + 0.946794 w_k1_MSFT_s103
                                 + 1.00136 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.928228 w_k1_ADM_s103
                                 + 0.696333 w_k1_AXP_s103
                                 + 0.834366 w_k1_DIS_s103
                                 + 0.778486 w_k1_FDX_s103
                                 + 0.923469 w_k1_IBM_s103
                                 + 0.811951 w_k1_INTC_s103
                                 + 0.782726 w_k1_KO_s103
                                 + 0.80139 w_k1_MAT_s103
                                 + 0.840502 w_k1_MSFT_s103
                                 + 0.851875 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.231 w_k1_ADM_s103
                                 + 1.00499 w_k1_AXP_s103
                                 + 1.29644 w_k1_DIS_s103
                                 + 1.18512 w_k1_FDX_s103
                                 + 1.48768 w_k1_IBM_s103
                                 + 1.01458 w_k1_INTC_s103
                                 + 1.14839 w_k1_KO_s103 + 1.04949 w_k1_MAT_s103
                                 + 1.01635 w_k1_MSFT_s103
                                 + 1.09898 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.12022 w_k1_ADM_s104
                                 + 0.938095 w_k1_AXP_s104
                                 + 1.19159 w_k1_DIS_s104
                                 + 1.20598 w_k1_FDX_s104
                                 + 0.70109 w_k1_IBM_s104
                                 + 1.10323 w_k1_INTC_s104
                                 + 0.880314 w_k1_KO_s104
                                 + 0.714027 w_k1_MAT_s104
                                 + 1.17691 w_k1_MSFT_s104
                                 + 0.990634 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.975613 w_k1_ADM_s104
                                 + 0.61639 w_k1_AXP_s104
                                 + 0.817395 w_k1_DIS_s104
                                 + 0.905123 w_k1_FDX_s104
                                 + 0.651008 w_k1_IBM_s104
                                 + 0.799888 w_k1_INTC_s104
                                 + 0.763511 w_k1_KO_s104
                                 + 0.573519 w_k1_MAT_s104
                                 + 0.892883 w_k1_MSFT_s104
                                 + 0.913347 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.96597 w_k1_ADM_s104
                                 + 1.0831 w_k1_AXP_s104 + 1.04783 w_k1_DIS_s104
                                 + 1.04385 w_k1_FDX_s104
                                 + 1.22699 w_k1_IBM_s104
                                 + 1.10094 w_k1_INTC_s104 + 1.0643 w_k1_KO_s104
                                 + 1.2416 w_k1_MAT_s104
                                 + 1.06275 w_k1_MSFT_s104 + 1.2346 w_k1_PG_s104
                                 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.24174 w_k1_ADM_s104
                                 + 1.20344 w_k1_AXP_s104
                                 + 1.31831 w_k1_DIS_s104
                                 + 1.15259 w_k1_FDX_s104
                                 + 1.56058 w_k1_IBM_s104
                                 + 1.51592 w_k1_INTC_s104
                                 + 1.28752 w_k1_KO_s104 + 1.18195 w_k1_MAT_s104
                                 + 1.3352 w_k1_MSFT_s104 + 1.30774 w_k1_PG_s104
                                 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.774213 w_k1_ADM_s104
                                 + 0.728321 w_k1_AXP_s104
                                 + 0.693661 w_k1_DIS_s104
                                 + 0.805478 w_k1_FDX_s104
                                 + 0.802292 w_k1_IBM_s104
                                 + 1.00749 w_k1_INTC_s104
                                 + 0.697959 w_k1_KO_s104
                                 + 0.698342 w_k1_MAT_s104
                                 + 0.942529 w_k1_MSFT_s104
                                 + 0.905156 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.784674 w_k1_ADM_s104
                                 + 0.79389 w_k1_AXP_s104
                                 + 0.759929 w_k1_DIS_s104
                                 + 0.842017 w_k1_FDX_s104
                                 + 1.04242 w_k1_IBM_s104
                                 + 0.946169 w_k1_INTC_s104
                                 + 0.915346 w_k1_KO_s104
                                 + 1.00033 w_k1_MAT_s104
                                 + 0.997651 w_k1_MSFT_s104
                                 + 1.01821 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.10013 w_k1_ADM_s104
                                 + 1.0855 w_k1_AXP_s104 + 1.03831 w_k1_DIS_s104
                                 + 1.03214 w_k1_FDX_s104
                                 + 0.876131 w_k1_IBM_s104
                                 + 1.11966 w_k1_INTC_s104
                                 + 0.908744 w_k1_KO_s104
                                 + 0.709847 w_k1_MAT_s104
                                 + 0.971351 w_k1_MSFT_s104
                                 + 1.04397 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.5272 w_k1_ADM_s104
                                 + 1.59244 w_k1_AXP_s104
                                 + 1.29168 w_k1_DIS_s104
                                 + 1.29055 w_k1_FDX_s104
                                 + 0.98459 w_k1_IBM_s104
                                 + 1.20632 w_k1_INTC_s104
                                 + 1.03656 w_k1_KO_s104 + 1.08874 w_k1_MAT_s104
                                 + 1.2352 w_k1_MSFT_s104 + 1.24115 w_k1_PG_s104
                                 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.907171 w_k1_ADM_s105
                                 + 1.23953 w_k1_AXP_s105 + 1.2875 w_k1_DIS_s105
                                 + 1.14266 w_k1_FDX_s105
                                 + 1.34759 w_k1_IBM_s105
                                 + 1.01964 w_k1_INTC_s105
                                 + 1.22203 w_k1_KO_s105 + 1.28502 w_k1_MAT_s105
                                 + 1.09636 w_k1_MSFT_s105
                                 + 1.17611 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.7733 w_k1_ADM_s105
                                 + 0.87357 w_k1_AXP_s105
                                 + 1.32451 w_k1_DIS_s105
                                 + 0.902559 w_k1_FDX_s105
                                 + 1.50424 w_k1_IBM_s105
                                 + 1.00614 w_k1_INTC_s105
                                 + 1.09774 w_k1_KO_s105 + 1.15374 w_k1_MAT_s105
                                 + 1.0361 w_k1_MSFT_s105 + 1.0324 w_k1_PG_s105
                                 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.29664 w_k1_ADM_s105
                                 + 1.10357 w_k1_AXP_s105
                                 + 1.09534 w_k1_DIS_s105
                                 + 1.23962 w_k1_FDX_s105
                                 + 1.04241 w_k1_IBM_s105
                                 + 1.01132 w_k1_INTC_s105
                                 + 0.976689 w_k1_KO_s105
                                 + 0.888011 w_k1_MAT_s105
                                 + 0.973535 w_k1_MSFT_s105
                                 + 1.11491 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.887108 w_k1_ADM_s105
                                 + 0.866703 w_k1_AXP_s105
                                 + 0.798587 w_k1_DIS_s105
                                 + 0.846993 w_k1_FDX_s105
                                 + 1.05949 w_k1_IBM_s105
                                 + 0.774047 w_k1_INTC_s105
                                 + 0.845897 w_k1_KO_s105
                                 + 1.01216 w_k1_MAT_s105
                                 + 0.777967 w_k1_MSFT_s105
                                 + 0.881793 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.791805 w_k1_ADM_s105
                                 + 0.683935 w_k1_AXP_s105
                                 + 0.798667 w_k1_DIS_s105
                                 + 0.878091 w_k1_FDX_s105
                                 + 0.63422 w_k1_IBM_s105
                                 + 0.755951 w_k1_INTC_s105
                                 + 0.741988 w_k1_KO_s105 + 0.7637 w_k1_MAT_s105
                                 + 0.833131 w_k1_MSFT_s105
                                 + 0.7958 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.964432 w_k1_ADM_s105
                                 + 1.12632 w_k1_AXP_s105
                                 + 1.25655 w_k1_DIS_s105
                                 + 1.06634 w_k1_FDX_s105
                                 + 0.988671 w_k1_IBM_s105
                                 + 0.968126 w_k1_INTC_s105
                                 + 0.928503 w_k1_KO_s105
                                 + 1.11627 w_k1_MAT_s105
                                 + 1.08878 w_k1_MSFT_s105
                                 + 0.920412 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.20951 w_k1_ADM_s105
                                 + 1.31417 w_k1_AXP_s105
                                 + 1.74933 w_k1_DIS_s105
                                 + 1.41189 w_k1_FDX_s105
                                 + 1.20109 w_k1_IBM_s105
                                 + 1.07813 w_k1_INTC_s105
                                 + 1.17727 w_k1_KO_s105 + 1.38504 w_k1_MAT_s105
                                 + 1.22273 w_k1_MSFT_s105
                                 + 1.38215 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.935246 w_k1_ADM_s105
                                 + 0.829153 w_k1_AXP_s105
                                 + 1.01102 w_k1_DIS_s105
                                 + 1.04748 w_k1_FDX_s105
                                 + 0.923352 w_k1_IBM_s105
                                 + 0.897604 w_k1_INTC_s105
                                 + 0.922461 w_k1_KO_s105
                                 + 0.994019 w_k1_MAT_s105
                                 + 1.0389 w_k1_MSFT_s105 + 1.02872 w_k1_PG_s105
                                 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.976218 w_k1_ADM_s106
                                 + 1.02718 w_k1_AXP_s106
                                 + 0.705295 w_k1_DIS_s106
                                 + 0.863591 w_k1_FDX_s106
                                 + 1.05124 w_k1_IBM_s106
                                 + 0.884689 w_k1_INTC_s106
                                 + 0.859721 w_k1_KO_s106
                                 + 1.19044 w_k1_MAT_s106
                                 + 0.813481 w_k1_MSFT_s106
                                 + 0.971658 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.872515 w_k1_ADM_s106
                                 + 1.09029 w_k1_AXP_s106
                                 + 0.913738 w_k1_DIS_s106
                                 + 0.905392 w_k1_FDX_s106
                                 + 1.51251 w_k1_IBM_s106
                                 + 0.823381 w_k1_INTC_s106
                                 + 0.954379 w_k1_KO_s106
                                 + 1.52809 w_k1_MAT_s106
                                 + 0.831707 w_k1_MSFT_s106
                                 + 1.04957 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.11822 w_k1_ADM_s106
                                 + 1.24782 w_k1_AXP_s106
                                 + 1.29311 w_k1_DIS_s106
                                 + 1.36854 w_k1_FDX_s106
                                 + 0.962859 w_k1_IBM_s106
                                 + 1.0282 w_k1_INTC_s106 + 1.08338 w_k1_KO_s106
                                 + 0.962337 w_k1_MAT_s106
                                 + 1.17617 w_k1_MSFT_s106
                                 + 1.10571 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.03787 w_k1_ADM_s106
                                 + 0.835005 w_k1_AXP_s106
                                 + 1.33806 w_k1_DIS_s106
                                 + 1.16226 w_k1_FDX_s106
                                 + 1.00617 w_k1_IBM_s106
                                 + 1.07914 w_k1_INTC_s106
                                 + 1.11156 w_k1_KO_s106
                                 + 0.831193 w_k1_MAT_s106
                                 + 1.12366 w_k1_MSFT_s106
                                 + 1.24408 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.07667 w_k1_ADM_s106
                                 + 1.6697 w_k1_AXP_s106 + 1.24888 w_k1_DIS_s106
                                 + 1.30301 w_k1_FDX_s106
                                 + 1.30769 w_k1_IBM_s106
                                 + 1.23192 w_k1_INTC_s106
                                 + 1.19859 w_k1_KO_s106 + 1.95288 w_k1_MAT_s106
                                 + 1.1594 w_k1_MSFT_s106 + 1.17889 w_k1_PG_s106
                                 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.962948 w_k1_ADM_s106
                                 + 0.994743 w_k1_AXP_s106
                                 + 1.01401 w_k1_DIS_s106
                                 + 1.03298 w_k1_FDX_s106
                                 + 1.12711 w_k1_IBM_s106
                                 + 0.864712 w_k1_INTC_s106
                                 + 1.07612 w_k1_KO_s106
                                 + 0.968518 w_k1_MAT_s106
                                 + 1.02621 w_k1_MSFT_s106
                                 + 1.04975 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.18225 w_k1_ADM_s106
                                 + 1.24206 w_k1_AXP_s106
                                 + 1.53224 w_k1_DIS_s106
                                 + 1.22751 w_k1_FDX_s106 + 1.5815 w_k1_IBM_s106
                                 + 1.04668 w_k1_INTC_s106
                                 + 1.15435 w_k1_KO_s106 + 1.24647 w_k1_MAT_s106
                                 + 1.07658 w_k1_MSFT_s106
                                 + 1.20888 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.832438 w_k1_ADM_s106
                                 + 0.860753 w_k1_AXP_s106
                                 + 0.69738 w_k1_DIS_s106
                                 + 0.902954 w_k1_FDX_s106
                                 + 0.757941 w_k1_IBM_s106
                                 + 0.840906 w_k1_INTC_s106
                                 + 0.726493 w_k1_KO_s106
                                 + 0.648649 w_k1_MAT_s106
                                 + 0.840212 w_k1_MSFT_s106
                                 + 0.929235 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.765559 w_k1_ADM_s107
                                 + 1.07176 w_k1_AXP_s107
                                 + 1.45527 w_k1_DIS_s107
                                 + 1.06487 w_k1_FDX_s107
                                 + 1.83723 w_k1_IBM_s107
                                 + 1.24026 w_k1_INTC_s107
                                 + 1.49667 w_k1_KO_s107 + 1.72175 w_k1_MAT_s107
                                 + 1.06591 w_k1_MSFT_s107
                                 + 0.988363 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.957285 w_k1_ADM_s107
                                 + 1.0169 w_k1_AXP_s107 + 1.04549 w_k1_DIS_s107
                                 + 1.10837 w_k1_FDX_s107
                                 + 1.04581 w_k1_IBM_s107
                                 + 1.07249 w_k1_INTC_s107
                                 + 0.87402 w_k1_KO_s107 + 1.15064 w_k1_MAT_s107
                                 + 0.929064 w_k1_MSFT_s107
                                 + 1.08278 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.42348 w_k1_ADM_s107
                                 + 1.35152 w_k1_AXP_s107
                                 + 1.75782 w_k1_DIS_s107
                                 + 1.48724 w_k1_FDX_s107
                                 + 1.19266 w_k1_IBM_s107
                                 + 1.6447 w_k1_INTC_s107 + 1.36681 w_k1_KO_s107
                                 + 1.12 w_k1_MAT_s107 + 1.44041 w_k1_MSFT_s107
                                 + 1.22173 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.902901 w_k1_ADM_s107
                                 + 1.07387 w_k1_AXP_s107
                                 + 1.37716 w_k1_DIS_s107
                                 + 1.16325 w_k1_FDX_s107
                                 + 0.918016 w_k1_IBM_s107
                                 + 1.52386 w_k1_INTC_s107 + 1.0608 w_k1_KO_s107
                                 + 1.03447 w_k1_MAT_s107
                                 + 1.34694 w_k1_MSFT_s107
                                 + 0.96809 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.927352 w_k1_ADM_s107
                                 + 0.724948 w_k1_AXP_s107
                                 + 0.75072 w_k1_DIS_s107
                                 + 0.886109 w_k1_FDX_s107
                                 + 0.626418 w_k1_IBM_s107
                                 + 0.897975 w_k1_INTC_s107
                                 + 0.645592 w_k1_KO_s107
                                 + 0.478788 w_k1_MAT_s107
                                 + 0.794787 w_k1_MSFT_s107
                                 + 0.844648 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.879074 w_k1_ADM_s107
                                 + 0.906112 w_k1_AXP_s107
                                 + 0.993906 w_k1_DIS_s107
                                 + 1.02922 w_k1_FDX_s107
                                 + 0.840686 w_k1_IBM_s107
                                 + 1.01455 w_k1_INTC_s107
                                 + 0.866105 w_k1_KO_s107
                                 + 0.798089 w_k1_MAT_s107
                                 + 0.958809 w_k1_MSFT_s107
                                 + 0.963534 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.736873 w_k1_ADM_s107
                                 + 0.743742 w_k1_AXP_s107
                                 + 0.831431 w_k1_DIS_s107
                                 + 0.743141 w_k1_FDX_s107
                                 + 0.861583 w_k1_IBM_s107
                                 + 0.810454 w_k1_INTC_s107
                                 + 0.713893 w_k1_KO_s107
                                 + 0.687256 w_k1_MAT_s107
                                 + 0.807636 w_k1_MSFT_s107
                                 + 0.844518 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.948004 w_k1_ADM_s107
                                 + 0.93831 w_k1_AXP_s107
                                 + 0.972569 w_k1_DIS_s107
                                 + 0.906356 w_k1_FDX_s107
                                 + 1.27983 w_k1_IBM_s107
                                 + 0.931361 w_k1_INTC_s107
                                 + 1.11465 w_k1_KO_s107 + 1.48384 w_k1_MAT_s107
                                 + 1.00025 w_k1_MSFT_s107
                                 + 1.07467 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.954831 w_k1_ADM_s110
                                 + 0.717849 w_k1_AXP_s110
                                 + 0.733591 w_k1_DIS_s110
                                 + 0.803307 w_k1_FDX_s110
                                 + 0.79059 w_k1_IBM_s110
                                 + 0.832221 w_k1_INTC_s110
                                 + 0.785155 w_k1_KO_s110
                                 + 0.740057 w_k1_MAT_s110
                                 + 0.911167 w_k1_MSFT_s110
                                 + 0.831687 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.18533 w_k1_ADM_s110
                                 + 1.87401 w_k1_AXP_s110
                                 + 1.71032 w_k1_DIS_s110
                                 + 1.50949 w_k1_FDX_s110
                                 + 1.40367 w_k1_IBM_s110
                                 + 1.54539 w_k1_INTC_s110
                                 + 1.26071 w_k1_KO_s110 + 2.07699 w_k1_MAT_s110
                                 + 1.43246 w_k1_MSFT_s110
                                 + 1.12704 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.29041 w_k1_ADM_s110
                                 + 1.31464 w_k1_AXP_s110
                                 + 1.44433 w_k1_DIS_s110
                                 + 1.36225 w_k1_FDX_s110
                                 + 0.915721 w_k1_IBM_s110
                                 + 0.870838 w_k1_INTC_s110
                                 + 1.09678 w_k1_KO_s110 + 1.27528 w_k1_MAT_s110
                                 + 1.22345 w_k1_MSFT_s110
                                 + 1.14746 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.09715 w_k1_ADM_s110
                                 + 0.785728 w_k1_AXP_s110
                                 + 1.06108 w_k1_DIS_s110
                                 + 0.982484 w_k1_FDX_s110
                                 + 0.976155 w_k1_IBM_s110
                                 + 0.944555 w_k1_INTC_s110
                                 + 0.984574 w_k1_KO_s110
                                 + 1.01187 w_k1_MAT_s110
                                 + 1.04003 w_k1_MSFT_s110
                                 + 0.976662 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.959238 w_k1_ADM_s110
                                 + 0.779444 w_k1_AXP_s110
                                 + 0.914674 w_k1_DIS_s110
                                 + 0.837119 w_k1_FDX_s110
                                 + 1.30642 w_k1_IBM_s110
                                 + 1.0722 w_k1_INTC_s110 + 1.12486 w_k1_KO_s110
                                 + 1.4363 w_k1_MAT_s110 + 1.1131 w_k1_MSFT_s110
                                 + 0.883463 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.07824 w_k1_ADM_s110
                                 + 1.53203 w_k1_AXP_s110
                                 + 0.938978 w_k1_DIS_s110
                                 + 1.07035 w_k1_FDX_s110
                                 + 0.969293 w_k1_IBM_s110
                                 + 1.09667 w_k1_INTC_s110
                                 + 0.993057 w_k1_KO_s110
                                 + 1.67662 w_k1_MAT_s110
                                 + 1.02257 w_k1_MSFT_s110
                                 + 1.05562 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.08817 w_k1_ADM_s110
                                 + 0.915072 w_k1_AXP_s110
                                 + 0.747097 w_k1_DIS_s110
                                 + 0.889222 w_k1_FDX_s110
                                 + 0.644459 w_k1_IBM_s110
                                 + 0.769754 w_k1_INTC_s110
                                 + 0.770294 w_k1_KO_s110
                                 + 0.780092 w_k1_MAT_s110
                                 + 0.783029 w_k1_MSFT_s110
                                 + 0.902106 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.15819 w_k1_ADM_s110
                                 + 1.50243 w_k1_AXP_s110
                                 + 1.78511 w_k1_DIS_s110
                                 + 1.19986 w_k1_FDX_s110
                                 + 1.60563 w_k1_IBM_s110
                                 + 1.11243 w_k1_INTC_s110 + 1.4869 w_k1_KO_s110
                                 + 1.96969 w_k1_MAT_s110
                                 + 1.36184 w_k1_MSFT_s110
                                 + 1.21536 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.05431 w_k1_ADM_s111
                                 + 1.04272 w_k1_AXP_s111
                                 + 0.834108 w_k1_DIS_s111
                                 + 1.13313 w_k1_FDX_s111
                                 + 0.605762 w_k1_IBM_s111
                                 + 1.09859 w_k1_INTC_s111
                                 + 0.663799 w_k1_KO_s111
                                 + 0.650084 w_k1_MAT_s111
                                 + 1.01445 w_k1_MSFT_s111
                                 + 0.880272 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.807409 w_k1_ADM_s111
                                 + 0.768161 w_k1_AXP_s111
                                 + 0.743075 w_k1_DIS_s111
                                 + 0.827058 w_k1_FDX_s111
                                 + 0.741323 w_k1_IBM_s111
                                 + 0.843779 w_k1_INTC_s111
                                 + 0.67212 w_k1_KO_s111
                                 + 0.865368 w_k1_MAT_s111
                                 + 0.896923 w_k1_MSFT_s111
                                 + 0.858077 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.837899 w_k1_ADM_s111
                                 + 0.924104 w_k1_AXP_s111
                                 + 1.08144 w_k1_DIS_s111
                                 + 0.996943 w_k1_FDX_s111
                                 + 0.83359 w_k1_IBM_s111
                                 + 1.03157 w_k1_INTC_s111
                                 + 0.822029 w_k1_KO_s111
                                 + 0.890647 w_k1_MAT_s111
                                 + 1.07736 w_k1_MSFT_s111
                                 + 0.858879 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.880918 w_k1_ADM_s111
                                 + 1.13629 w_k1_AXP_s111
                                 + 1.02049 w_k1_DIS_s111
                                 + 0.931441 w_k1_FDX_s111
                                 + 1.20654 w_k1_IBM_s111
                                 + 1.05006 w_k1_INTC_s111
                                 + 0.934526 w_k1_KO_s111
                                 + 1.40831 w_k1_MAT_s111
                                 + 0.855575 w_k1_MSFT_s111
                                 + 0.927227 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.918532 w_k1_ADM_s111
                                 + 1.24872 w_k1_AXP_s111
                                 + 1.33213 w_k1_DIS_s111
                                 + 1.09727 w_k1_FDX_s111
                                 + 1.57027 w_k1_IBM_s111
                                 + 1.22641 w_k1_INTC_s111
                                 + 1.29494 w_k1_KO_s111 + 1.51969 w_k1_MAT_s111
                                 + 1.10513 w_k1_MSFT_s111
                                 + 1.02271 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.980213 w_k1_ADM_s111
                                 + 1.25334 w_k1_AXP_s111
                                 + 1.49325 w_k1_DIS_s111
                                 + 1.39267 w_k1_FDX_s111
                                 + 0.854353 w_k1_IBM_s111
                                 + 1.22547 w_k1_INTC_s111
                                 + 1.07499 w_k1_KO_s111
                                 + 0.756007 w_k1_MAT_s111
                                 + 1.17886 w_k1_MSFT_s111
                                 + 1.12158 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.757758 w_k1_ADM_s111
                                 + 0.694577 w_k1_AXP_s111
                                 + 0.737298 w_k1_DIS_s111
                                 + 0.737685 w_k1_FDX_s111
                                 + 1.07953 w_k1_IBM_s111
                                 + 0.655198 w_k1_INTC_s111
                                 + 0.816716 w_k1_KO_s111
                                 + 1.12411 w_k1_MAT_s111
                                 + 0.774658 w_k1_MSFT_s111
                                 + 0.812615 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.811831 w_k1_ADM_s111
                                 + 1.09409 w_k1_AXP_s111
                                 + 0.84622 w_k1_DIS_s111
                                 + 0.952372 w_k1_FDX_s111
                                 + 0.865507 w_k1_IBM_s111
                                 + 0.979596 w_k1_INTC_s111
                                 + 0.754839 w_k1_KO_s111
                                 + 0.849848 w_k1_MAT_s111
                                 + 0.831197 w_k1_MSFT_s111
                                 + 1.0604 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.96979 w_k1_ADM_s112
                                 + 0.794732 w_k1_AXP_s112
                                 + 1.04169 w_k1_DIS_s112
                                 + 0.935062 w_k1_FDX_s112
                                 + 0.806034 w_k1_IBM_s112
                                 + 0.929905 w_k1_INTC_s112
                                 + 0.991005 w_k1_KO_s112
                                 + 0.824517 w_k1_MAT_s112
                                 + 1.03377 w_k1_MSFT_s112
                                 + 0.974649 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.747249 w_k1_ADM_s112
                                 + 0.856422 w_k1_AXP_s112
                                 + 0.987164 w_k1_DIS_s112
                                 + 0.851185 w_k1_FDX_s112
                                 + 1.18325 w_k1_IBM_s112
                                 + 1.11799 w_k1_INTC_s112
                                 + 1.14402 w_k1_KO_s112 + 1.12144 w_k1_MAT_s112
                                 + 0.934118 w_k1_MSFT_s112
                                 + 0.978512 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.980742 w_k1_ADM_s112
                                 + 1.09035 w_k1_AXP_s112
                                 + 1.50862 w_k1_DIS_s112
                                 + 1.24859 w_k1_FDX_s112
                                 + 1.21767 w_k1_IBM_s112
                                 + 1.10809 w_k1_INTC_s112
                                 + 1.28058 w_k1_KO_s112 + 1.59599 w_k1_MAT_s112
                                 + 1.32427 w_k1_MSFT_s112
                                 + 1.13998 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.968918 w_k1_ADM_s112
                                 + 0.723022 w_k1_AXP_s112
                                 + 0.981003 w_k1_DIS_s112
                                 + 1.03339 w_k1_FDX_s112
                                 + 0.691043 w_k1_IBM_s112
                                 + 1.07426 w_k1_INTC_s112
                                 + 0.840041 w_k1_KO_s112
                                 + 0.654409 w_k1_MAT_s112
                                 + 1.10931 w_k1_MSFT_s112
                                 + 0.875062 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.786738 w_k1_ADM_s112
                                 + 0.875163 w_k1_AXP_s112
                                 + 0.743072 w_k1_DIS_s112
                                 + 0.815323 w_k1_FDX_s112
                                 + 1.03284 w_k1_IBM_s112
                                 + 0.880417 w_k1_INTC_s112
                                 + 0.857912 w_k1_KO_s112
                                 + 1.08653 w_k1_MAT_s112
                                 + 0.805612 w_k1_MSFT_s112
                                 + 0.733237 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.08418 w_k1_ADM_s112
                                 + 1.4558 w_k1_AXP_s112
                                 + 0.923189 w_k1_DIS_s112
                                 + 1.11731 w_k1_FDX_s112
                                 + 0.934586 w_k1_IBM_s112
                                 + 1.19454 w_k1_INTC_s112
                                 + 1.03386 w_k1_KO_s112 + 1.41178 w_k1_MAT_s112
                                 + 0.967829 w_k1_MSFT_s112
                                 + 0.967753 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.21901 w_k1_ADM_s112
                                 + 1.16614 w_k1_AXP_s112
                                 + 1.00096 w_k1_DIS_s112
                                 + 0.993616 w_k1_FDX_s112
                                 + 1.14157 w_k1_IBM_s112
                                 + 1.09687 w_k1_INTC_s112
                                 + 1.01252 w_k1_KO_s112 + 1.42844 w_k1_MAT_s112
                                 + 0.896161 w_k1_MSFT_s112
                                 + 0.904911 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.592988 w_k1_ADM_s112
                                 + 0.692606 w_k1_AXP_s112
                                 + 0.656598 w_k1_DIS_s112
                                 + 0.770233 w_k1_FDX_s112
                                 + 0.737882 w_k1_IBM_s112
                                 + 0.761059 w_k1_INTC_s112
                                 + 0.670981 w_k1_KO_s112
                                 + 0.916767 w_k1_MAT_s112
                                 + 0.82784 w_k1_MSFT_s112
                                 + 0.718906 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.35607 w_k1_ADM_s113
                                 + 1.29282 w_k1_AXP_s113
                                 + 1.43897 w_k1_DIS_s113 + 1.3374 w_k1_FDX_s113
                                 + 0.781479 w_k1_IBM_s113
                                 + 1.2039 w_k1_INTC_s113 + 1.14277 w_k1_KO_s113
                                 + 0.816188 w_k1_MAT_s113
                                 + 1.23102 w_k1_MSFT_s113
                                 + 1.13689 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.11953 w_k1_ADM_s113
                                 + 1.06208 w_k1_AXP_s113
                                 + 0.974415 w_k1_DIS_s113
                                 + 0.963108 w_k1_FDX_s113
                                 + 1.03424 w_k1_IBM_s113
                                 + 1.04866 w_k1_INTC_s113
                                 + 1.06617 w_k1_KO_s113
                                 + 0.870438 w_k1_MAT_s113
                                 + 0.906934 w_k1_MSFT_s113
                                 + 0.99702 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.906561 w_k1_ADM_s113
                                 + 1.42313 w_k1_AXP_s113
                                 + 1.60966 w_k1_DIS_s113
                                 + 1.15924 w_k1_FDX_s113
                                 + 1.65227 w_k1_IBM_s113
                                 + 1.38892 w_k1_INTC_s113
                                 + 1.58494 w_k1_KO_s113 + 1.4484 w_k1_MAT_s113
                                 + 1.19564 w_k1_MSFT_s113
                                 + 1.25279 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.74029 w_k1_ADM_s113
                                 + 1.06338 w_k1_AXP_s113
                                 + 0.901101 w_k1_DIS_s113
                                 + 0.823231 w_k1_FDX_s113
                                 + 1.39011 w_k1_IBM_s113
                                 + 0.960484 w_k1_INTC_s113
                                 + 1.12653 w_k1_KO_s113 + 1.39834 w_k1_MAT_s113
                                 + 0.862617 w_k1_MSFT_s113
                                 + 0.954111 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.798599 w_k1_ADM_s113
                                 + 0.731818 w_k1_AXP_s113
                                 + 0.709779 w_k1_DIS_s113
                                 + 0.820125 w_k1_FDX_s113
                                 + 0.767065 w_k1_IBM_s113
                                 + 0.914521 w_k1_INTC_s113
                                 + 0.858741 w_k1_KO_s113
                                 + 0.834081 w_k1_MAT_s113
                                 + 0.884901 w_k1_MSFT_s113
                                 + 0.854768 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.025 w_k1_ADM_s113
                                 + 0.838379 w_k1_AXP_s113
                                 + 1.3005 w_k1_DIS_s113 + 1.19157 w_k1_FDX_s113
                                 + 0.828634 w_k1_IBM_s113
                                 + 1.16403 w_k1_INTC_s113 + 1.13 w_k1_KO_s113
                                 + 0.502993 w_k1_MAT_s113
                                 + 1.1635 w_k1_MSFT_s113
                                 + 0.993053 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.915374 w_k1_ADM_s113
                                 + 1.07531 w_k1_AXP_s113
                                 + 1.27915 w_k1_DIS_s113
                                 + 1.05882 w_k1_FDX_s113 + 1.0235 w_k1_IBM_s113
                                 + 1.04797 w_k1_INTC_s113
                                 + 1.19163 w_k1_KO_s113 + 1.17657 w_k1_MAT_s113
                                 + 1.10642 w_k1_MSFT_s113
                                 + 1.02715 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.73001 w_k1_ADM_s113
                                 + 0.746421 w_k1_AXP_s113
                                 + 0.543789 w_k1_DIS_s113
                                 + 0.806942 w_k1_FDX_s113
                                 + 0.526393 w_k1_IBM_s113
                                 + 0.944094 w_k1_INTC_s113
                                 + 0.597174 w_k1_KO_s113
                                 + 0.572852 w_k1_MAT_s113
                                 + 0.750563 w_k1_MSFT_s113
                                 + 0.676462 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.782929 w_k1_ADM_s114
                                 + 0.971654 w_k1_AXP_s114
                                 + 0.901749 w_k1_DIS_s114
                                 + 0.840929 w_k1_FDX_s114
                                 + 1.40701 w_k1_IBM_s114
                                 + 0.93659 w_k1_INTC_s114
                                 + 1.16568 w_k1_KO_s114 + 1.3937 w_k1_MAT_s114
                                 + 0.839338 w_k1_MSFT_s114
                                 + 0.960043 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.860612 w_k1_ADM_s114
                                 + 0.811899 w_k1_AXP_s114
                                 + 0.75308 w_k1_DIS_s114
                                 + 0.853873 w_k1_FDX_s114
                                 + 0.800196 w_k1_IBM_s114
                                 + 0.768341 w_k1_INTC_s114
                                 + 0.88727 w_k1_KO_s114
                                 + 0.744983 w_k1_MAT_s114
                                 + 0.785938 w_k1_MSFT_s114
                                 + 0.914052 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.97463 w_k1_ADM_s114
                                 + 1.00008 w_k1_AXP_s114
                                 + 1.11952 w_k1_DIS_s114
                                 + 1.00949 w_k1_FDX_s114
                                 + 1.04967 w_k1_IBM_s114
                                 + 0.909917 w_k1_INTC_s114
                                 + 1.22511 w_k1_KO_s114 + 1.2993 w_k1_MAT_s114
                                 + 1.01169 w_k1_MSFT_s114
                                 + 1.15647 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.54339 w_k1_ADM_s114
                                 + 1.3821 w_k1_AXP_s114 + 1.53711 w_k1_DIS_s114
                                 + 1.30792 w_k1_FDX_s114
                                 + 1.11698 w_k1_IBM_s114
                                 + 1.02921 w_k1_INTC_s114
                                 + 1.17207 w_k1_KO_s114 + 1.2908 w_k1_MAT_s114
                                 + 1.14472 w_k1_MSFT_s114
                                 + 1.11014 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.779901 w_k1_ADM_s114
                                 + 1.07362 w_k1_AXP_s114 + 0.8622 w_k1_DIS_s114
                                 + 1.00327 w_k1_FDX_s114
                                 + 1.25505 w_k1_IBM_s114
                                 + 0.915887 w_k1_INTC_s114
                                 + 0.990055 w_k1_KO_s114
                                 + 1.41197 w_k1_MAT_s114
                                 + 0.860075 w_k1_MSFT_s114
                                 + 0.95771 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.12159 w_k1_ADM_s114
                                 + 0.91953 w_k1_AXP_s114
                                 + 0.985147 w_k1_DIS_s114
                                 + 1.08776 w_k1_FDX_s114
                                 + 0.677176 w_k1_IBM_s114
                                 + 1.08882 w_k1_INTC_s114
                                 + 0.899262 w_k1_KO_s114
                                 + 0.700809 w_k1_MAT_s114
                                 + 1.02756 w_k1_MSFT_s114
                                 + 0.962642 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.48883 w_k1_ADM_s114
                                 + 1.225 w_k1_AXP_s114 + 0.982456 w_k1_DIS_s114
                                 + 1.19948 w_k1_FDX_s114
                                 + 0.933765 w_k1_IBM_s114
                                 + 0.830368 w_k1_INTC_s114
                                 + 0.939393 w_k1_KO_s114
                                 + 1.30349 w_k1_MAT_s114
                                 + 0.901193 w_k1_MSFT_s114
                                 + 0.994751 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.784688 w_k1_ADM_s114
                                 + 0.726116 w_k1_AXP_s114
                                 + 0.582199 w_k1_DIS_s114
                                 + 0.721352 w_k1_FDX_s114
                                 + 1.00317 w_k1_IBM_s114
                                 + 0.75649 w_k1_INTC_s114
                                 + 0.832979 w_k1_KO_s114
                                 + 1.06425 w_k1_MAT_s114
                                 + 0.774375 w_k1_MSFT_s114
                                 + 0.849675 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.897237 w_k1_ADM_s115
                                 + 0.927297 w_k1_AXP_s115
                                 + 0.748754 w_k1_DIS_s115
                                 + 0.891318 w_k1_FDX_s115
                                 + 0.843916 w_k1_IBM_s115
                                 + 0.862022 w_k1_INTC_s115
                                 + 0.846056 w_k1_KO_s115
                                 + 0.936993 w_k1_MAT_s115
                                 + 0.864817 w_k1_MSFT_s115
                                 + 0.907666 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.30068 w_k1_ADM_s115
                                 + 1.22034 w_k1_AXP_s115
                                 + 0.912334 w_k1_DIS_s115
                                 + 1.15002 w_k1_FDX_s115
                                 + 0.899094 w_k1_IBM_s115
                                 + 1.13754 w_k1_INTC_s115
                                 + 0.947807 w_k1_KO_s115
                                 + 0.71069 w_k1_MAT_s115
                                 + 0.958434 w_k1_MSFT_s115
                                 + 0.919458 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.05334 w_k1_ADM_s115
                                 + 0.729702 w_k1_AXP_s115
                                 + 0.668725 w_k1_DIS_s115
                                 + 0.849832 w_k1_FDX_s115
                                 + 0.717105 w_k1_IBM_s115
                                 + 0.733396 w_k1_INTC_s115
                                 + 0.76291 w_k1_KO_s115
                                 + 0.809312 w_k1_MAT_s115
                                 + 0.911924 w_k1_MSFT_s115
                                 + 0.830152 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.58219 w_k1_ADM_s115
                                 + 1.33922 w_k1_AXP_s115
                                 + 1.38066 w_k1_DIS_s115
                                 + 1.23354 w_k1_FDX_s115
                                 + 1.19184 w_k1_IBM_s115
                                 + 0.979743 w_k1_INTC_s115
                                 + 1.37315 w_k1_KO_s115 + 1.11834 w_k1_MAT_s115
                                 + 1.2071 w_k1_MSFT_s115 + 1.14504 w_k1_PG_s115
                                 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.02275 w_k1_ADM_s115
                                 + 0.868928 w_k1_AXP_s115
                                 + 1.06856 w_k1_DIS_s115
                                 + 0.888581 w_k1_FDX_s115
                                 + 1.07079 w_k1_IBM_s115
                                 + 1.00399 w_k1_INTC_s115
                                 + 1.13599 w_k1_KO_s115
                                 + 0.835866 w_k1_MAT_s115
                                 + 0.990325 w_k1_MSFT_s115
                                 + 0.947716 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.883188 w_k1_ADM_s115
                                 + 0.774621 w_k1_AXP_s115
                                 + 0.708622 w_k1_DIS_s115
                                 + 0.734818 w_k1_FDX_s115
                                 + 1.04976 w_k1_IBM_s115
                                 + 0.786384 w_k1_INTC_s115
                                 + 0.959854 w_k1_KO_s115
                                 + 0.842891 w_k1_MAT_s115
                                 + 0.725876 w_k1_MSFT_s115
                                 + 0.766925 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.33517 w_k1_ADM_s115
                                 + 1.4492 w_k1_AXP_s115 + 1.31439 w_k1_DIS_s115
                                 + 1.05315 w_k1_FDX_s115
                                 + 1.49582 w_k1_IBM_s115
                                 + 1.01253 w_k1_INTC_s115
                                 + 1.24921 w_k1_KO_s115 + 1.80137 w_k1_MAT_s115
                                 + 1.07219 w_k1_MSFT_s115
                                 + 1.13573 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.10575 w_k1_ADM_s115
                                 + 1.02423 w_k1_AXP_s115
                                 + 0.949963 w_k1_DIS_s115
                                 + 0.989085 w_k1_FDX_s115
                                 + 1.13513 w_k1_IBM_s115
                                 + 0.928302 w_k1_INTC_s115
                                 + 0.952954 w_k1_KO_s115
                                 + 1.00614 w_k1_MAT_s115
                                 + 0.883543 w_k1_MSFT_s115
                                 + 0.949471 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.949702 w_k1_ADM_s116
                                 + 0.940882 w_k1_AXP_s116
                                 + 0.883407 w_k1_DIS_s116
                                 + 0.954698 w_k1_FDX_s116
                                 + 1.29771 w_k1_IBM_s116
                                 + 0.811334 w_k1_INTC_s116
                                 + 1.06862 w_k1_KO_s116 + 1.48045 w_k1_MAT_s116
                                 + 0.89404 w_k1_MSFT_s116 + 1.1323 w_k1_PG_s116
                                 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.960958 w_k1_ADM_s116
                                 + 1.25857 w_k1_AXP_s116 + 1.216 w_k1_DIS_s116
                                 + 1.10494 w_k1_FDX_s116
                                 + 1.46065 w_k1_IBM_s116
                                 + 0.950333 w_k1_INTC_s116
                                 + 1.26648 w_k1_KO_s116 + 2.74944 w_k1_MAT_s116
                                 + 1.02737 w_k1_MSFT_s116
                                 + 1.14031 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.19294 w_k1_ADM_s116
                                 + 0.983412 w_k1_AXP_s116
                                 + 1.33924 w_k1_DIS_s116
                                 + 1.09453 w_k1_FDX_s116
                                 + 1.16905 w_k1_IBM_s116
                                 + 0.933147 w_k1_INTC_s116
                                 + 1.15058 w_k1_KO_s116 + 1.54525 w_k1_MAT_s116
                                 + 1.12898 w_k1_MSFT_s116 + 1.1812 w_k1_PG_s116
                                 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.09681 w_k1_ADM_s116
                                 + 1.13814 w_k1_AXP_s116
                                 + 0.791912 w_k1_DIS_s116
                                 + 0.988889 w_k1_FDX_s116
                                 + 0.729753 w_k1_IBM_s116
                                 + 0.727299 w_k1_INTC_s116
                                 + 0.719309 w_k1_KO_s116
                                 + 0.986805 w_k1_MAT_s116
                                 + 0.812557 w_k1_MSFT_s116
                                 + 0.818182 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.755229 w_k1_ADM_s116
                                 + 0.756916 w_k1_AXP_s116
                                 + 0.708788 w_k1_DIS_s116
                                 + 0.791643 w_k1_FDX_s116
                                 + 1.31059 w_k1_IBM_s116
                                 + 0.757745 w_k1_INTC_s116
                                 + 0.894189 w_k1_KO_s116
                                 + 1.67069 w_k1_MAT_s116
                                 + 0.77351 w_k1_MSFT_s116
                                 + 0.806911 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.12412 w_k1_ADM_s116
                                 + 1.18478 w_k1_AXP_s116
                                 + 1.35323 w_k1_DIS_s116
                                 + 1.29185 w_k1_FDX_s116
                                 + 0.872777 w_k1_IBM_s116
                                 + 1.2747 w_k1_INTC_s116 + 1.04883 w_k1_KO_s116
                                 + 1.1797 w_k1_MAT_s116 + 1.3248 w_k1_MSFT_s116
                                 + 0.955283 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.04647 w_k1_ADM_s116
                                 + 0.86809 w_k1_AXP_s116
                                 + 0.906112 w_k1_DIS_s116
                                 + 0.99288 w_k1_FDX_s116
                                 + 0.977697 w_k1_IBM_s116
                                 + 0.826531 w_k1_INTC_s116
                                 + 0.920572 w_k1_KO_s116
                                 + 1.28421 w_k1_MAT_s116
                                 + 1.08678 w_k1_MSFT_s116
                                 + 0.980523 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.748362 w_k1_ADM_s116
                                 + 0.748213 w_k1_AXP_s116
                                 + 0.642927 w_k1_DIS_s116
                                 + 0.767846 w_k1_FDX_s116
                                 + 0.74814 w_k1_IBM_s116
                                 + 0.924142 w_k1_INTC_s116
                                 + 0.868189 w_k1_KO_s116
                                 + 0.894593 w_k1_MAT_s116
                                 + 0.927822 w_k1_MSFT_s116
                                 + 0.84895 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.96995 w_k1_ADM_s117
                                 + 0.982463 w_k1_AXP_s117
                                 + 1.00462 w_k1_DIS_s117
                                 + 0.972468 w_k1_FDX_s117
                                 + 1.27735 w_k1_IBM_s117
                                 + 1.19518 w_k1_INTC_s117
                                 + 1.40699 w_k1_KO_s117 + 1.62131 w_k1_MAT_s117
                                 + 1.08887 w_k1_MSFT_s117
                                 + 1.15881 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.06564 w_k1_ADM_s117
                                 + 1.41949 w_k1_AXP_s117
                                 + 1.30735 w_k1_DIS_s117
                                 + 1.11368 w_k1_FDX_s117
                                 + 1.30892 w_k1_IBM_s117
                                 + 1.32041 w_k1_INTC_s117
                                 + 1.56877 w_k1_KO_s117 + 1.74631 w_k1_MAT_s117
                                 + 1.12926 w_k1_MSFT_s117
                                 + 1.30264 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.46336 w_k1_ADM_s117
                                 + 1.53215 w_k1_AXP_s117
                                 + 2.25528 w_k1_DIS_s117
                                 + 1.49164 w_k1_FDX_s117
                                 + 1.46761 w_k1_IBM_s117
                                 + 1.62975 w_k1_INTC_s117
                                 + 1.87823 w_k1_KO_s117 + 1.69964 w_k1_MAT_s117
                                 + 1.69537 w_k1_MSFT_s117
                                 + 1.23474 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.03603 w_k1_ADM_s117
                                 + 1.0555 w_k1_AXP_s117 + 1.41415 w_k1_DIS_s117
                                 + 1.12625 w_k1_FDX_s117
                                 + 1.18307 w_k1_IBM_s117
                                 + 1.1709 w_k1_INTC_s117 + 1.25932 w_k1_KO_s117
                                 + 1.74606 w_k1_MAT_s117
                                 + 1.32455 w_k1_MSFT_s117
                                 + 1.03613 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.880979 w_k1_ADM_s117
                                 + 0.635806 w_k1_AXP_s117
                                 + 0.671335 w_k1_DIS_s117
                                 + 0.834171 w_k1_FDX_s117
                                 + 0.704321 w_k1_IBM_s117
                                 + 0.940704 w_k1_INTC_s117
                                 + 0.847139 w_k1_KO_s117
                                 + 0.642478 w_k1_MAT_s117
                                 + 0.857105 w_k1_MSFT_s117
                                 + 0.844014 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.05585 w_k1_ADM_s117
                                 + 1.22825 w_k1_AXP_s117
                                 + 0.921206 w_k1_DIS_s117
                                 + 1.06967 w_k1_FDX_s117
                                 + 0.854457 w_k1_IBM_s117
                                 + 1.03335 w_k1_INTC_s117
                                 + 1.05764 w_k1_KO_s117 + 1.44778 w_k1_MAT_s117
                                 + 1.12529 w_k1_MSFT_s117
                                 + 1.04705 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.07609 w_k1_ADM_s117
                                 + 0.997275 w_k1_AXP_s117
                                 + 0.632391 w_k1_DIS_s117
                                 + 0.87918 w_k1_FDX_s117
                                 + 0.847072 w_k1_IBM_s117
                                 + 0.919353 w_k1_INTC_s117
                                 + 0.856305 w_k1_KO_s117
                                 + 1.05513 w_k1_MAT_s117
                                 + 0.919086 w_k1_MSFT_s117
                                 + 0.845957 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.714774 w_k1_ADM_s117
                                 + 0.748699 w_k1_AXP_s117
                                 + 0.909887 w_k1_DIS_s117
                                 + 0.844233 w_k1_FDX_s117
                                 + 1.04954 w_k1_IBM_s117
                                 + 1.00751 w_k1_INTC_s117 + 1.1735 w_k1_KO_s117
                                 + 1.59281 w_k1_MAT_s117
                                 + 1.02979 w_k1_MSFT_s117
                                 + 0.893598 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0591799380982522 w_k1_ADM_s000
                                 + 0.0915827924959292 w_k1_ADM_s001
                                 + 0.049290560881264 w_k1_ADM_s002
                                 + 0.0178320066816746 w_k1_ADM_s003
                                 + 0.0352913484699061 w_k1_ADM_s004
                                 + 0.0213050034308211 w_k1_ADM_s005
                                 + 0.0298676476291997 w_k1_ADM_s006
                                 + 0.0518499688494854 w_k1_ADM_s007
                                 + 0.0321770021569023 w_k1_ADM_s010
                                 + 0.0713045978587673 w_k1_ADM_s011
                                 + 0.0208539239123383 w_k1_ADM_s012
                                 + 0.0331131794791885 w_k1_ADM_s013
                                 + 0.0869444689747398 w_k1_ADM_s014
                                 + 0.0292199684326537 w_k1_ADM_s015
                                 + 0.062246189497654 w_k1_ADM_s016
                                 + 0.0428057841801743 w_k1_ADM_s017
                                 + 0.0417550928917912 w_k1_ADM_s100
                                 + 0.0101875102105643 w_k1_ADM_s101
                                 + 0.0242862607256084 w_k1_ADM_s102
                                 + 0.0186514434057144 w_k1_ADM_s103
                                 + 0.013574562467038 w_k1_ADM_s104
                                 + 0.0248490366145413 w_k1_ADM_s105
                                 + 0.0145550297900269 w_k1_ADM_s106
                                 + 0.0213880693753465 w_k1_ADM_s107
                                 + 0.0203025654573303 w_k1_ADM_s110
                                 + 0.0116706889566362 w_k1_ADM_s111
                                 + 0.021828466789518 w_k1_ADM_s112
                                 + 0.0235901976660153 w_k1_ADM_s113
                                 + 0.0179300468630652 w_k1_ADM_s114
                                 + 0.00821322382626641 w_k1_ADM_s115
                                 + 0.00660009592845563 w_k1_ADM_s116
                                 + 0.0111751119837756 w_k1_ADM_s117
                                 + 0.0526437398690546 w_k1_AXP_s000
                                 + 0.0813982044141461 w_k1_AXP_s001
                                 + 0.0442625378940449 w_k1_AXP_s002
                                 + 0.0199010490939996 w_k1_AXP_s003
                                 + 0.030671020277927 w_k1_AXP_s004
                                 + 0.0228650997209338 w_k1_AXP_s005
                                 + 0.0307701098527122 w_k1_AXP_s006
                                 + 0.0512182775216733 w_k1_AXP_s007
                                 + 0.0299553619745587 w_k1_AXP_s010
                                 + 0.086730439983642 w_k1_AXP_s011
                                 + 0.0227421293814503 w_k1_AXP_s012
                                 + 0.0338375843672806 w_k1_AXP_s013
                                 + 0.101091705533148 w_k1_AXP_s014
                                 + 0.0270971821448326 w_k1_AXP_s015
                                 + 0.0529470007941552 w_k1_AXP_s016
                                 + 0.040398461022759 w_k1_AXP_s017
                                 + 0.0473562895086343 w_k1_AXP_s100
                                 + 0.00864622838818117 w_k1_AXP_s101
                                 + 0.0213561334548057 w_k1_AXP_s102
                                 + 0.0156528280089198 w_k1_AXP_s103
                                 + 0.0133404599650897 w_k1_AXP_s104
                                 + 0.0251356105063281 w_k1_AXP_s105
                                 + 0.0157623338626506 w_k1_AXP_s106
                                 + 0.0215687553354395 w_k1_AXP_s107
                                 + 0.0193733479905933 w_k1_AXP_s110
                                 + 0.0138629039562491 w_k1_AXP_s111
                                 + 0.0223583384661506 w_k1_AXP_s112
                                 + 0.0250775725830625 w_k1_AXP_s113
                                 + 0.0170519819608458 w_k1_AXP_s114
                                 + 0.00741378488982162 w_k1_AXP_s115
                                 + 0.00636237287637699 w_k1_AXP_s116
                                 + 0.0115564655678459 w_k1_AXP_s117
                                 + 0.0552827909343569 w_k1_DIS_s000
                                 + 0.0839081829306026 w_k1_DIS_s001
                                 + 0.0457066789520432 w_k1_DIS_s002
                                 + 0.0197160508283789 w_k1_DIS_s003
                                 + 0.032209749985031 w_k1_DIS_s004
                                 + 0.0222160311102947 w_k1_DIS_s005
                                 + 0.0317012689005907 w_k1_DIS_s006
                                 + 0.0475405001970015 w_k1_DIS_s007
                                 + 0.0312475431179864 w_k1_DIS_s010
                                 + 0.0814567760666837 w_k1_DIS_s011
                                 + 0.0224636020012941 w_k1_DIS_s012
                                 + 0.0342353867106289 w_k1_DIS_s013
                                 + 0.0944812684973622 w_k1_DIS_s014
                                 + 0.0249666844779185 w_k1_DIS_s015
                                 + 0.0544198744876504 w_k1_DIS_s016
                                 + 0.0440973234700318 w_k1_DIS_s017
                                 + 0.0415936052625131 w_k1_DIS_s100
                                 + 0.00918730027681743 w_k1_DIS_s101
                                 + 0.0230927744827539 w_k1_DIS_s102
                                 + 0.0183236706496292 w_k1_DIS_s103
                                 + 0.0131438813005205 w_k1_DIS_s104
                                 + 0.0277054896041082 w_k1_DIS_s105
                                 + 0.0147001128798662 w_k1_DIS_s106
                                 + 0.0247793744656777 w_k1_DIS_s107
                                 + 0.0204152331368538 w_k1_DIS_s110
                                 + 0.0141765233451515 w_k1_DIS_s111
                                 + 0.0250679340955834 w_k1_DIS_s112
                                 + 0.0264921292819564 w_k1_DIS_s113
                                 + 0.0167665080661965 w_k1_DIS_s114
                                 + 0.00709697504596469 w_k1_DIS_s115
                                 + 0.00661736432706815 w_k1_DIS_s116
                                 + 0.0122683957183345 w_k1_DIS_s117
                                 + 0.0551627897739471 w_k1_FDX_s000
                                 + 0.086404516186832 w_k1_FDX_s001
                                 + 0.0436084473546362 w_k1_FDX_s002
                                 + 0.0200455410575945 w_k1_FDX_s003
                                 + 0.0309317317713889 w_k1_FDX_s004
                                 + 0.022724979790688 w_k1_FDX_s005
                                 + 0.0319594619755945 w_k1_FDX_s006
                                 + 0.0480654968785814 w_k1_FDX_s007
                                 + 0.03202120118108 w_k1_FDX_s010
                                 + 0.084986689337754 w_k1_FDX_s011
                                 + 0.0224351416900762 w_k1_FDX_s012
                                 + 0.035466968676253 w_k1_FDX_s013
                                 + 0.0895339072935371 w_k1_FDX_s014
                                 + 0.0256808779816439 w_k1_FDX_s015
                                 + 0.0532643164791989 w_k1_FDX_s016
                                 + 0.0405925752527421 w_k1_FDX_s017
                                 + 0.0450417055209618 w_k1_FDX_s100
                                 + 0.00943169780277448 w_k1_FDX_s101
                                 + 0.0221526707388695 w_k1_FDX_s102
                                 + 0.0178359435610061 w_k1_FDX_s103
                                 + 0.0132059175971761 w_k1_FDX_s104
                                 + 0.0266976675451394 w_k1_FDX_s105
                                 + 0.0156068650735686 w_k1_FDX_s106
                                 + 0.0232177999322091 w_k1_FDX_s107
                                 + 0.0192485533171256 w_k1_FDX_s110
                                 + 0.0135848011319069 w_k1_FDX_s111
                                 + 0.023698350580357 w_k1_FDX_s112
                                 + 0.0250333091547369 w_k1_FDX_s113
                                 + 0.0167828962435382 w_k1_FDX_s114
                                 + 0.00698993692855482 w_k1_FDX_s115
                                 + 0.00667606330799922 w_k1_FDX_s116
                                 + 0.0114949437966518 w_k1_FDX_s117
                                 + 0.0517492246001723 w_k1_IBM_s000
                                 + 0.0857344474131554 w_k1_IBM_s001
                                 + 0.043772596665469 w_k1_IBM_s002
                                 + 0.0197777081344906 w_k1_IBM_s003
                                 + 0.0308980311090343 w_k1_IBM_s004
                                 + 0.023414445062133 w_k1_IBM_s005
                                 + 0.0340484769421997 w_k1_IBM_s006
                                 + 0.0467393686779469 w_k1_IBM_s007
                                 + 0.0311048700869693 w_k1_IBM_s010
                                 + 0.083095296316889 w_k1_IBM_s011
                                 + 0.0229257737462577 w_k1_IBM_s012
                                 + 0.0346754287523735 w_k1_IBM_s013
                                 + 0.0928059540149684 w_k1_IBM_s014
                                 + 0.0274499369960937 w_k1_IBM_s015
                                 + 0.0557526611629956 w_k1_IBM_s016
                                 + 0.044477526179864 w_k1_IBM_s017
                                 + 0.0459961541098587 w_k1_IBM_s100
                                 + 0.00987462844828205 w_k1_IBM_s101
                                 + 0.0229550797738007 w_k1_IBM_s102
                                 + 0.0165371211140275 w_k1_IBM_s103
                                 + 0.0133696487612381 w_k1_IBM_s104
                                 + 0.0274582519112771 w_k1_IBM_s105
                                 + 0.0157621910703121 w_k1_IBM_s106
                                 + 0.0231675509191939 w_k1_IBM_s107
                                 + 0.0186463210542792 w_k1_IBM_s110
                                 + 0.0134284211069424 w_k1_IBM_s111
                                 + 0.0236619575314894 w_k1_IBM_s112
                                 + 0.0245545685678409 w_k1_IBM_s113
                                 + 0.0168748612250843 w_k1_IBM_s114
                                 + 0.00762827003876615 w_k1_IBM_s115
                                 + 0.00656641354941573 w_k1_IBM_s116
                                 + 0.0117007597059472 w_k1_IBM_s117
                                 + 0.0472062617992714 w_k1_INTC_s000
                                 + 0.0778171904514665 w_k1_INTC_s001
                                 + 0.039749074595703 w_k1_INTC_s002
                                 + 0.0207577191751877 w_k1_INTC_s003
                                 + 0.0316970089666358 w_k1_INTC_s004
                                 + 0.0251702514216074 w_k1_INTC_s005
                                 + 0.0287992015478814 w_k1_INTC_s006
                                 + 0.0441088989085002 w_k1_INTC_s007
                                 + 0.026385365676223 w_k1_INTC_s010
                                 + 0.0904329048415351 w_k1_INTC_s011
                                 + 0.02209873331914 w_k1_INTC_s012
                                 + 0.0347347588657968 w_k1_INTC_s013
                                 + 0.0800302471304006 w_k1_INTC_s014
                                 + 0.0269798057649683 w_k1_INTC_s015
                                 + 0.0531704927342827 w_k1_INTC_s016
                                 + 0.0448210737592147 w_k1_INTC_s017
                                 + 0.0440733435125872 w_k1_INTC_s100
                                 + 0.00929618347876011 w_k1_INTC_s101
                                 + 0.0216054496422844 w_k1_INTC_s102
                                 + 0.0179619502644038 w_k1_INTC_s103
                                 + 0.0142436971677006 w_k1_INTC_s104
                                 + 0.0235637738763421 w_k1_INTC_s105
                                 + 0.0140363683326208 w_k1_INTC_s106
                                 + 0.0245570102870805 w_k1_INTC_s107
                                 + 0.0178557259219222 w_k1_INTC_s110
                                 + 0.0137878842482268 w_k1_INTC_s111
                                 + 0.0243774836313866 w_k1_INTC_s112
                                 + 0.0265765417143739 w_k1_INTC_s113
                                 + 0.0148112706779157 w_k1_INTC_s114
                                 + 0.00664893896902268 w_k1_INTC_s115
                                 + 0.00623889941162405 w_k1_INTC_s116
                                 + 0.01258942536633 w_k1_INTC_s117
                                 + 0.0512979600882709 w_k1_KO_s000
                                 + 0.0845071397957871 w_k1_KO_s001
                                 + 0.0424639200043536 w_k1_KO_s002
                                 + 0.0190530911723742 w_k1_KO_s003
                                 + 0.0333346954434115 w_k1_KO_s004
                                 + 0.0230007364142795 w_k1_KO_s005
                                 + 0.0339035080861207 w_k1_KO_s006
                                 + 0.0478177971447261 w_k1_KO_s007
                                 + 0.0342317197902306 w_k1_KO_s010
                                 + 0.0754985666859985 w_k1_KO_s011
                                 + 0.0251232467451682 w_k1_KO_s012
                                 + 0.0321275184896537 w_k1_KO_s013
                                 + 0.0876583512302187 w_k1_KO_s014
                                 + 0.0248179684666301 w_k1_KO_s015
                                 + 0.0516528248772179 w_k1_KO_s016
                                 + 0.0485167710428459 w_k1_KO_s017
                                 + 0.0443675707885571 w_k1_KO_s100
                                 + 0.0105557431446101 w_k1_KO_s101
                                 + 0.0251901230967292 w_k1_KO_s102
                                 + 0.0165294372722344 w_k1_KO_s103
                                 + 0.012509312296924 w_k1_KO_s104
                                 + 0.024897428766052 w_k1_KO_s105
                                 + 0.0142511695355425 w_k1_KO_s106
                                 + 0.0219087833094205 w_k1_KO_s107
                                 + 0.018794386986106 w_k1_KO_s110
                                 + 0.0121900404263069 w_k1_KO_s111
                                 + 0.0240423881141792 w_k1_KO_s112
                                 + 0.0267059813361892 w_k1_KO_s113
                                 + 0.0170117683321108 w_k1_KO_s114
                                 + 0.00748266050497453 w_k1_KO_s115
                                 + 0.00653126580933559 w_k1_KO_s116
                                 + 0.0135653665253433 w_k1_KO_s117
                                 + 0.061124952146502 w_k1_MAT_s000
                                 + 0.0954872725242859 w_k1_MAT_s001
                                 + 0.0408253292027614 w_k1_MAT_s002
                                 + 0.0163167105801428 w_k1_MAT_s003
                                 + 0.0282659056647608 w_k1_MAT_s004
                                 + 0.0244191010024091 w_k1_MAT_s005
                                 + 0.0442277434192148 w_k1_MAT_s006
                                 + 0.0510249559187286 w_k1_MAT_s007
                                 + 0.0355388542578551 w_k1_MAT_s010
                                 + 0.107694280400719 w_k1_MAT_s011
                                 + 0.0264987280561134 w_k1_MAT_s012
                                 + 0.0342736393274029 w_k1_MAT_s013
                                 + 0.106467993879237 w_k1_MAT_s014
                                 + 0.0259858476246627 w_k1_MAT_s015
                                 + 0.0480741768892724 w_k1_MAT_s016
                                 + 0.0610990879179572 w_k1_MAT_s017
                                 + 0.0493151074622385 w_k1_MAT_s100
                                 + 0.0103337361817875 w_k1_MAT_s101
                                 + 0.0269463618426536 w_k1_MAT_s102
                                 + 0.0167169934178177 w_k1_MAT_s103
                                 + 0.0123325208417805 w_k1_MAT_s104
                                 + 0.0274609046257146 w_k1_MAT_s105
                                 + 0.0155891028339866 w_k1_MAT_s106
                                 + 0.0227154668445362 w_k1_MAT_s107
                                 + 0.0223042279544803 w_k1_MAT_s110
                                 + 0.0139868755471277 w_k1_MAT_s111
                                 + 0.0277703523531151 w_k1_MAT_s112
                                 + 0.0235474864424441 w_k1_MAT_s113
                                 + 0.0185672353511859 w_k1_MAT_s114
                                 + 0.00715872313399649 w_k1_MAT_s115
                                 + 0.00880725584861246 w_k1_MAT_s116
                                 + 0.0159968761801018 w_k1_MAT_s117
                                 + 0.057638828413049 w_k1_MSFT_s000
                                 + 0.0800512217117397 w_k1_MSFT_s001
                                 + 0.0473506134135644 w_k1_MSFT_s002
                                 + 0.0197887458989746 w_k1_MSFT_s003
                                 + 0.0309745389794 w_k1_MSFT_s004
                                 + 0.0221584544177125 w_k1_MSFT_s005
                                 + 0.0316032008258025 w_k1_MSFT_s006
                                 + 0.0456104885305575 w_k1_MSFT_s007
                                 + 0.0293091639673119 w_k1_MSFT_s010
                                 + 0.0833311100028792 w_k1_MSFT_s011
                                 + 0.0231476528801127 w_k1_MSFT_s012
                                 + 0.0353941958600765 w_k1_MSFT_s013
                                 + 0.0843673899373673 w_k1_MSFT_s014
                                 + 0.0265823221432943 w_k1_MSFT_s015
                                 + 0.0502839017225873 w_k1_MSFT_s016
                                 + 0.0425600196126162 w_k1_MSFT_s017
                                 + 0.0452760731245306 w_k1_MSFT_s100
                                 + 0.00964711166109998 w_k1_MSFT_s101
                                 + 0.024137555966803 w_k1_MSFT_s102
                                 + 0.0170663672796952 w_k1_MSFT_s103
                                 + 0.0138132837351209 w_k1_MSFT_s104
                                 + 0.0251846128596193 w_k1_MSFT_s105
                                 + 0.0144130039655694 w_k1_MSFT_s106
                                 + 0.0226123196104343 w_k1_MSFT_s107
                                 + 0.0198694941363787 w_k1_MSFT_s110
                                 + 0.0130652775010923 w_k1_MSFT_s111
                                 + 0.0244791951214022 w_k1_MSFT_s112
                                 + 0.0249997911498488 w_k1_MSFT_s113
                                 + 0.0153828314860608 w_k1_MSFT_s114
                                 + 0.00684309936537538 w_k1_MSFT_s115
                                 + 0.00695737287064007 w_k1_MSFT_s116
                                 + 0.0126947900224189 w_k1_MSFT_s117
                                 + 0.0536900960285416 w_k1_PG_s000
                                 + 0.0847261157037728 w_k1_PG_s001
                                 + 0.0479486069425042 w_k1_PG_s002
                                 + 0.0183920619827676 w_k1_PG_s003
                                 + 0.0311205582370668 w_k1_PG_s004
                                 + 0.0233970734362932 w_k1_PG_s005
                                 + 0.0324181496072762 w_k1_PG_s006
                                 + 0.0476435654775453 w_k1_PG_s007
                                 + 0.0297962093418218 w_k1_PG_s010
                                 + 0.0857935341820194 w_k1_PG_s011
                                 + 0.022308654115123 w_k1_PG_s012
                                 + 0.0364589992705115 w_k1_PG_s013
                                 + 0.0836587356046049 w_k1_PG_s014
                                 + 0.0230340935961211 w_k1_PG_s015
                                 + 0.0513718459272705 w_k1_PG_s016
                                 + 0.0408437105386645 w_k1_PG_s017
                                 + 0.0474363473134136 w_k1_PG_s100
                                 + 0.010211046937953 w_k1_PG_s101
                                 + 0.0231120166346899 w_k1_PG_s102
                                 + 0.0169150243513039 w_k1_PG_s103
                                 + 0.0142112832140075 w_k1_PG_s104
                                 + 0.0261871246270649 w_k1_PG_s105
                                 + 0.015670935940306 w_k1_PG_s106
                                 + 0.0223871008243233 w_k1_PG_s107
                                 + 0.0183717801830029 w_k1_PG_s110
                                 + 0.0125672036166074 w_k1_PG_s111
                                 + 0.0222365335657981 w_k1_PG_s112
                                 + 0.0244333233139611 w_k1_PG_s113
                                 + 0.0165475541504759 w_k1_PG_s114
                                 + 0.00685727871067216 w_k1_PG_s115
                                 + 0.00649158567393882 w_k1_PG_s116
                                 + 0.0115192172118437 w_k1_PG_s117 >= 1.19
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
