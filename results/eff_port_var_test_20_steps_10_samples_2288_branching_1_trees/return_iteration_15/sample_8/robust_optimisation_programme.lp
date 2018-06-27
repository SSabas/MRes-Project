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
 w_k1_ADM_s0_return_constraint:  1.10763 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.976858 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.00391 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.1051 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.02405 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.903811 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.01124 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.03873 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.964253 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.913784 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.29041 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.14087 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.596355 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.43952 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.92384 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.40437 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.832978 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.16859 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.966449 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.19282 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.725485 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.608588 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.26806 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.33619 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.20257 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.758959 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.07043 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.597717 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.0744 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.868904 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.47567 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.741818 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.32616 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.25592 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.03125 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.829581 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.0445 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.99271 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.26609 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.919761 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.826265 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.15529 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.13051 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.851988 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.09456 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.982385 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.820976 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.987675 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.47735 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.38574 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.614574 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.61393 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.902704 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.43794 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.635471 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.22228 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.763663 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.05655 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.728425 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.55639 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.09472 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.40924 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.31992 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.912786 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.3276 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.548969 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.11421 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.77383 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.62298 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.697235 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.21648 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.4358 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.02878 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.981117 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.930451 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.03756 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.35256 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.873976 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.87487 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.10858 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.235 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.854914 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.17117 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.06781 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.622334 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.871476 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.32099 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.21072 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.492661 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.96378 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.845116 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.84158 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.756459 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.11257 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.938999 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.06219 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.618707 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.500882 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.42118 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.55131 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.46546 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.782139 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.02507 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.438808 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.06372 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.701819 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 2.05719 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.633657 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.22196 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.355 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.35722 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.984424 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.985935 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.814016 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.15285 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.924717 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.987188 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.08002 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.04182 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.983533 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.971947 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.06575 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.799121 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.907081 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.40064 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.20802 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.555767 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.55057 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.783875 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.48902 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.803485 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.07593 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.96656 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.17907 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.70587 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.589544 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.25714 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.2287 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.33853 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.863481 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.05014 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.645704 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.17205 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.821078 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.54623 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.68892 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.24505 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.23678 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.04108 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.828933 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.987401 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.815169 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.24391 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.918031 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.746092 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.20112 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.29543 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.791845 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.53774 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.728698 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.809048 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.989916 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.760508 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.06794 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.949184 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.14255 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.33713 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.21458 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.671849 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.12386 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.772347 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.692715 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.90278 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.666954 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.814178 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.71753 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.17211 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.0828 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.45085 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.752675 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.81522 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.736126 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.13722 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.764902 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.691314 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.19428 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.20167 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.46249 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.835397 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.04756 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.14137 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.967658 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.88478 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.99501 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.09785 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.954741 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.972317 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.08072 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.846038 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.07673 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.2505 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.25944 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.761285 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.62501 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.989291 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.15741 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.940757 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.01686 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.01908 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.970279 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.803367 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.67188 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.15509 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.07573 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.1822 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.922434 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.01273 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.698362 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.0296 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.839525 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.22267 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.80216 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.08 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.30797 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.03397 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.07441 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.02756 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.812545 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.23566 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.946999 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.859466 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.19097 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.14848 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.816065 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.30003 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.895078 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.73718 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.891646 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.967396 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.992056 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.787829 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.37928 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.03835 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.36413 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.823331 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.07031 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.825526 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.805809 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.800105 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.599175 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.07994 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.44667 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.23175 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.01306 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.17367 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.679048 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.923334 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.740867 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.33084 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.785528 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.79823 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.13736 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.28289 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.17195 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.02284 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.909663 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.35107 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.887714 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.545981 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.23347 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.2907 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.772781 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 2.06672 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.87998 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.05215 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.16952 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.962922 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.47965 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.21789 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.30458 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.25763 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.5028 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.496503 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.23932 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.742381 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.803152 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.831343 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.601225 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.808008 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.75734 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.26514 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.22766 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.77856 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.62246 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.99548 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.778004 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.09861 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.683147 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.733231 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.34868 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.955856 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.1731 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.06604 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.18387 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.15825 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.937227 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.967281 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.10558 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.02901 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.957262 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.994827 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.04167 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.776622 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.886739 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.17862 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.126 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.645282 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.52614 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.830803 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.27657 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.846559 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.99246 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.951214 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.03898 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.689874 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.580249 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.23622 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.28718 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.22407 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.903909 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.99365 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.701977 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.0777 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.943689 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.4293 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.760201 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.15496 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.25378 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.15863 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.923159 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.973817 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.750192 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.13394 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.91705 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.874956 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.09379 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.11604 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.957989 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.26053 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.04048 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.910845 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.01497 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.17617 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.16977 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.798331 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.29745 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.94971 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.26371 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.802645 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.04082 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.845476 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.883848 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.825081 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.72562 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.17748 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.25838 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.09351 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.99461 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.11614 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.754276 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.97052 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.87577 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.3277 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.860865 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.01009 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.11501 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.15112 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.98203 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.986026 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.911956 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.252999718098495 z_k1_s0000
                                 - 0.187999729334968 z_k1_s0001
                                 - 0.460998510624033 z_k1_s0002
                                 - 0.198999653408938 z_k1_s0003
                                 - 0.48200005828927 z_k1_s0004
                                 - 0.190999651573433 z_k1_s0005
                                 - 0.17099975158447 z_k1_s0006
                                 - 0.146999746077955 z_k1_s0007
                                 - 0.0770000243779113 z_k1_s0010
                                 - 0.19000022507539 z_k1_s0011
                                 - 0.0710000447913241 z_k1_s0012
                                 - 0.0719999791390794 z_k1_s0013
                                 - 0.0569999928226059 z_k1_s0014
                                 - 0.145000191425958 z_k1_s0015
                                 - 0.0520000222837851 z_k1_s0016
                                 - 0.0829999292644874 z_k1_s0017
                                 - 0.121999859822608 z_k1_s0020
                                 - 0.550000002249238 z_k1_s0021
                                 - 0.257999777913357 z_k1_s0022
                                 - 0.509000860692174 z_k1_s0023
                                 - 0.204999976583815 z_k1_s0024
                                 - 0.264999807047428 z_k1_s0025
                                 - 0.305000316556284 z_k1_s0026
                                 - 0.454000060067532 z_k1_s0027
                                 - 0.272000155737822 z_k1_s0030
                                 - 0.421000710481005 z_k1_s0031
                                 - 0.471000082649156 z_k1_s0032
                                 - 0.488000071326502 z_k1_s0033
                                 - 0.600999797887511 z_k1_s0034
                                 - 0.398000527623249 z_k1_s0035
                                 - 0.198000006848029 z_k1_s0036
                                 - 0.51999886095342 z_k1_s0037
                                 - 0.127999830112059 z_k1_s0040
                                 - 0.123000294794753 z_k1_s0041
                                 - 0.244999616527043 z_k1_s0042
                                 - 0.0899999505037262 z_k1_s0043
                                 - 0.0959999258839946 z_k1_s0044
                                 - 0.101999901264263 z_k1_s0045
                                 - 0.146000289252365 z_k1_s0046
                                 - 0.136000152618751 z_k1_s0047
                                 - 0.286999311180058 z_k1_s0050
                                 - 0.461999677193722 z_k1_s0051
                                 - 0.174000335590677 z_k1_s0052
                                 - 0.280001081422765 z_k1_s0053
                                 - 0.404999926287085 z_k1_s0054
                                 - 0.181000300651133 z_k1_s0055
                                 - 0.178000209382172 z_k1_s0056
                                 - 0.512001198343065 z_k1_s0057
                                 - 0.1549993806664 z_k1_s0060
                                 - 0.0949999734599212 z_k1_s0061
                                 - 0.191999662980156 z_k1_s0062
                                 - 0.0869999770190853 z_k1_s0063
                                 - 0.131000523543474 z_k1_s0064
                                 - 0.303000509921424 z_k1_s0065
                                 - 0.0749999194578546 z_k1_s0066
                                 - 0.220000216622873 z_k1_s0067
                                 - 0.204999624054732 z_k1_s0070
                                 - 0.0669999014300627 z_k1_s0071
                                 - 0.171999981572503 z_k1_s0072
                                 - 0.116000219269634 z_k1_s0073
                                 - 0.10699987397485 z_k1_s0074
                                 - 0.0639999339316783 z_k1_s0075
                                 - 0.0809998973557337 z_k1_s0076
                                 - 0.294999756005342 z_k1_s0077
                                 - 0.589998326712364 z_k1_s0100
                                 - 0.406000366211537 z_k1_s0101
                                 - 0.589000154058115 z_k1_s0102
                                 - 0.296000271812355 z_k1_s0103
                                 - 0.656000046972963 z_k1_s0104
                                 - 1.23199949150981 z_k1_s0105
                                 - 0.516999611865609 z_k1_s0106
                                 - 0.607001512857044 z_k1_s0107
                                 - 0.181999894836103 z_k1_s0110
                                 - 0.636999102126886 z_k1_s0111
                                 - 0.166999858868337 z_k1_s0112
                                 - 0.350999570126995 z_k1_s0113
                                 - 0.626000465002483 z_k1_s0114
                                 - 0.679997627609377 z_k1_s0115
                                 - 0.474000995301564 z_k1_s0116
                                 - 0.414998993643521 z_k1_s0117
                                 - 0.0730000308673554 z_k1_s0120
                                 - 0.208999934181781 z_k1_s0121
                                 - 0.230000427664483 z_k1_s0122
                                 - 0.115000213832241 z_k1_s0123
                                 - 0.104000481770212 z_k1_s0124
                                 - 0.155000244512993 z_k1_s0125
                                 - 0.0670000765152749 z_k1_s0126
                                 - 0.052000039884977 z_k1_s0127
                                 - 0.385001015351112 z_k1_s0130
                                 - 0.421001068274247 z_k1_s0131
                                 - 0.955003915104952 z_k1_s0132
                                 - 0.263000413066292 z_k1_s0133
                                 - 0.443000677681968 z_k1_s0134
                                 - 1.14800306219611 z_k1_s0135
                                 - 0.424000993384347 z_k1_s0136
                                 - 0.71900187642762 z_k1_s0137
                                 - 0.1780000742186 z_k1_s0140
                                 - 0.229000696630819 z_k1_s0141
                                 - 0.47399959321792 z_k1_s0142
                                 - 0.24100031759827 z_k1_s0143
                                 - 0.132000103043019 z_k1_s0144
                                 - 0.141000110068679 z_k1_s0145
                                 - 0.409000837144026 z_k1_s0146
                                 - 0.138000107726792 z_k1_s0147
                                 - 0.149000059864571 z_k1_s0150
                                 - 0.216999246799473 z_k1_s0151
                                 - 0.151000114309772 z_k1_s0152
                                 - 0.545000563806901 z_k1_s0153
                                 - 0.170000060638767 z_k1_s0154
                                 - 0.191000442013237 z_k1_s0155
                                 - 0.353001046071751 z_k1_s0156
                                 - 0.127000031867775 z_k1_s0157
                                 - 0.888997769465025 z_k1_s0160
                                 - 0.279999546224894 z_k1_s0161
                                 - 0.4289987200953 z_k1_s0162
                                 - 0.396998936069393 z_k1_s0163
                                 - 0.214999661822674 z_k1_s0164
                                 - 0.252999620753169 z_k1_s0165
                                 - 0.188999765501714 z_k1_s0166
                                 - 0.219999807576497 z_k1_s0167
                                 - 0.0660000184618547 z_k1_s0170
                                 - 0.142000492175653 z_k1_s0171
                                 - 0.187000328808678 z_k1_s0172
                                 - 0.0570000511352496 z_k1_s0173
                                 - 0.126999809297738 z_k1_s0174
                                 - 0.274000529099362 z_k1_s0175
                                 - 0.110000030769758 z_k1_s0176
                                 - 0.143000316500775 z_k1_s0177
                                 - 0.448001722280731 z_k1_s1000
                                 - 0.140000159312334 z_k1_s1001
                                 - 0.327001144343184 z_k1_s1002
                                 - 0.576001853503422 z_k1_s1003
                                 - 0.198000210878858 z_k1_s1004
                                 - 0.293001148955948 z_k1_s1005
                                 - 0.260001450609823 z_k1_s1006
                                 - 0.284001001588584 z_k1_s1007
                                 - 0.242000942425253 z_k1_s1010
                                 - 0.218000072506952 z_k1_s1011
                                 - 0.509000080890074 z_k1_s1012
                                 - 0.289000538140004 z_k1_s1013
                                 - 0.118000061347802 z_k1_s1014
                                 - 0.252000702641139 z_k1_s1015
                                 - 0.541001240781143 z_k1_s1016
                                 - 0.240000026781959 z_k1_s1017
                                 - 0.426999167648301 z_k1_s1020
                                 - 0.597996449873352 z_k1_s1021
                                 - 0.587000346037832 z_k1_s1022
                                 - 0.200999614693448 z_k1_s1023
                                 - 0.653000109000103 z_k1_s1024
                                 - 0.816997556607635 z_k1_s1025
                                 - 0.680999447470177 z_k1_s1026
                                 - 0.664000840825453 z_k1_s1027
                                 - 0.940997391901847 z_k1_s1030
                                 - 0.888000872236093 z_k1_s1031
                                 - 0.445999721449141 z_k1_s1032
                                 - 0.445999721449141 z_k1_s1033
                                 - 1.00799663808113 z_k1_s1034
                                 - 0.841999274830549 z_k1_s1035
                                 - 1.03300058671294 z_k1_s1036
                                 - 0.837001061502811 z_k1_s1037
                                 - 0.147000095654093 z_k1_s1040
                                 - 0.260000784959725 z_k1_s1041
                                 - 0.131000047348879 z_k1_s1042
                                 - 0.184000091309881 z_k1_s1043
                                 - 0.347001163669345 z_k1_s1044
                                 - 0.199000020545999 z_k1_s1045
                                 - 0.55700063717508 z_k1_s1046
                                 - 0.496000104861394 z_k1_s1047
                                 - 0.569000808739438 z_k1_s1050
                                 - 0.152000146960581 z_k1_s1051
                                 - 0.276999829772632 z_k1_s1052
                                 - 0.425000472481925 z_k1_s1053
                                 - 0.287000595902082 z_k1_s1054
                                 - 0.369001530160287 z_k1_s1055
                                 - 0.235000088231076 z_k1_s1056
                                 - 0.360000840643783 z_k1_s1057
                                 - 0.323999088796047 z_k1_s1060
                                 - 0.791997155547248 z_k1_s1061
                                 - 0.348999027057451 z_k1_s1062
                                 - 0.990996090237289 z_k1_s1063
                                 - 0.630998645349402 z_k1_s1064
                                 - 0.422999196038435 z_k1_s1065
                                 - 0.295999287526989 z_k1_s1066
                                 - 0.822000413613611 z_k1_s1067
                                 - 0.338000175321856 z_k1_s1070
                                 - 0.746001634262095 z_k1_s1071
                                 - 0.582998762085028 z_k1_s1072
                                 - 0.556999167952661 z_k1_s1073
                                 - 0.35099997238804 z_k1_s1074
                                 - 0.461001016002449 z_k1_s1075
                                 - 0.724999341116204 z_k1_s1076
                                 - 0.782000373368313 z_k1_s1077
                                 - 0.220999926336689 z_k1_s1100
                                 - 0.185999897901506 z_k1_s1101
                                 - 0.47199874196178 z_k1_s1102
                                 - 0.747998058125821 z_k1_s1103
                                 - 0.494998175642611 z_k1_s1104
                                 - 0.382998940329194 z_k1_s1105
                                 - 0.327000850671002 z_k1_s1106
                                 - 0.223999693025648 z_k1_s1107
                                 - 1.62199587887604 z_k1_s1110
                                 - 0.918999264569863 z_k1_s1111
                                 - 1.60099897320795 z_k1_s1112
                                 - 0.494999788527641 z_k1_s1113
                                 - 0.34499937266523 z_k1_s1114
                                 - 0.873996198434268 z_k1_s1115
                                 - 0.773996926791591 z_k1_s1116
                                 - 0.912000296013833 z_k1_s1117
                                 - 0.506998800461934 z_k1_s1120
                                 - 0.280000201045169 z_k1_s1121
                                 - 0.127999808906648 z_k1_s1122
                                 - 0.207000424554544 z_k1_s1123
                                 - 0.102999812181165 z_k1_s1124
                                 - 0.135999871250739 z_k1_s1125
                                 - 0.320000512765623 z_k1_s1126
                                 - 0.300000356905396 z_k1_s1127
                                 - 0.692003838975486 z_k1_s1130
                                 - 0.885999212321271 z_k1_s1131
                                 - 0.782999052539623 z_k1_s1132
                                 - 0.220000220605218 z_k1_s1133
                                 - 0.355000968041473 z_k1_s1134
                                 - 0.466001145261512 z_k1_s1135
                                 - 0.733000885678599 z_k1_s1136
                                 - 1.04400046305631 z_k1_s1137
                                 - 0.122000065929818 z_k1_s1140
                                 - 0.563999347416643 z_k1_s1141
                                 - 0.190999867528676 z_k1_s1142
                                 - 0.192000019714149 z_k1_s1143
                                 - 0.343999301608767 z_k1_s1144
                                 - 0.232000980433654 z_k1_s1145
                                 - 0.189999938243177 z_k1_s1146
                                 - 0.39400022388318 z_k1_s1147
                                 - 0.245000162245887 z_k1_s1150
                                 - 0.308000203966258 z_k1_s1151
                                 - 0.673000670879532 z_k1_s1152
                                 - 0.35699984231541 z_k1_s1153
                                 - 0.526001361732049 z_k1_s1154
                                 - 0.709001708119807 z_k1_s1155
                                 - 0.399999701891209 z_k1_s1156
                                 - 0.723000140990902 z_k1_s1157
                                 - 0.387000356261798 z_k1_s1160
                                 - 1.06399972181335 z_k1_s1161
                                 - 0.944998829240622 z_k1_s1162
                                 - 0.98500119397429 z_k1_s1163
                                 - 0.894004296718921 z_k1_s1164
                                 - 0.728998929699638 z_k1_s1165
                                 - 0.329000437403658 z_k1_s1166
                                 - 0.517001021634861 z_k1_s1167
                                 - 0.737999105920188 z_k1_s1170
                                 - 0.306999659828565 z_k1_s1171
                                 - 0.327000318354363 z_k1_s1172
                                 - 0.203999816368136 z_k1_s1173
                                 - 0.206000012420622 z_k1_s1174
                                 - 0.581000804537702 z_k1_s1175
                                 - 0.453001277169201 z_k1_s1176
                                 - 0.438998277302979 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.973536 w_k1_ADM_s000
                                 + 1.25502 w_k1_AXP_s000
                                 + 0.98087 w_k1_DIS_s000
                                 + 1.04191 w_k1_FDX_s000
                                 + 1.04956 w_k1_IBM_s000
                                 + 1.21794 w_k1_INTC_s000
                                 + 0.973169 w_k1_KO_s000
                                 + 1.42508 w_k1_MAT_s000
                                 + 0.985358 w_k1_MSFT_s000
                                 + 1.03261 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.17133 w_k1_ADM_s000
                                 + 1.03958 w_k1_AXP_s000
                                 + 1.30299 w_k1_DIS_s000
                                 + 1.13173 w_k1_FDX_s000
                                 + 0.956845 w_k1_IBM_s000
                                 + 1.38641 w_k1_INTC_s000
                                 + 1.09444 w_k1_KO_s000
                                 + 0.647942 w_k1_MAT_s000
                                 + 1.37579 w_k1_MSFT_s000
                                 + 0.919918 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.03048 w_k1_ADM_s000
                                 + 1.07668 w_k1_AXP_s000
                                 + 0.909361 w_k1_DIS_s000
                                 + 1.07556 w_k1_FDX_s000
                                 + 0.772687 w_k1_IBM_s000
                                 + 1.01255 w_k1_INTC_s000
                                 + 0.793626 w_k1_KO_s000
                                 + 0.786105 w_k1_MAT_s000
                                 + 0.977869 w_k1_MSFT_s000
                                 + 0.933948 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.27716 w_k1_ADM_s000
                                 + 1.61247 w_k1_AXP_s000
                                 + 1.47255 w_k1_DIS_s000
                                 + 1.53282 w_k1_FDX_s000
                                 + 1.07597 w_k1_IBM_s000
                                 + 1.58714 w_k1_INTC_s000
                                 + 1.19541 w_k1_KO_s000 + 1.16641 w_k1_MAT_s000
                                 + 1.41938 w_k1_MSFT_s000
                                 + 1.22156 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.743241 w_k1_ADM_s000
                                 + 0.725787 w_k1_AXP_s000
                                 + 0.669145 w_k1_DIS_s000
                                 + 0.793023 w_k1_FDX_s000
                                 + 0.811693 w_k1_IBM_s000
                                 + 1.02309 w_k1_INTC_s000
                                 + 0.740268 w_k1_KO_s000
                                 + 0.793341 w_k1_MAT_s000
                                 + 0.858389 w_k1_MSFT_s000
                                 + 0.793133 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.894137 w_k1_ADM_s000
                                 + 1.0194 w_k1_AXP_s000 + 1.09442 w_k1_DIS_s000
                                 + 1.03524 w_k1_FDX_s000 + 1.2298 w_k1_IBM_s000
                                 + 1.00901 w_k1_INTC_s000
                                 + 0.999404 w_k1_KO_s000
                                 + 0.885958 w_k1_MAT_s000
                                 + 0.906541 w_k1_MSFT_s000
                                 + 0.862059 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.834158 w_k1_ADM_s000
                                 + 0.952564 w_k1_AXP_s000
                                 + 0.858721 w_k1_DIS_s000
                                 + 0.82707 w_k1_FDX_s000
                                 + 1.49264 w_k1_IBM_s000
                                 + 1.08897 w_k1_INTC_s000
                                 + 1.01674 w_k1_KO_s000 + 1.05598 w_k1_MAT_s000
                                 + 0.823018 w_k1_MSFT_s000
                                 + 0.87442 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.4035 w_k1_ADM_s000
                                 + 1.43535 w_k1_AXP_s000
                                 + 1.36159 w_k1_DIS_s000
                                 + 1.28533 w_k1_FDX_s000
                                 + 1.47687 w_k1_IBM_s000
                                 + 1.34489 w_k1_INTC_s000
                                 + 1.35066 w_k1_KO_s000 + 1.63518 w_k1_MAT_s000
                                 + 1.18589 w_k1_MSFT_s000
                                 + 1.04297 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.836697 w_k1_ADM_s001
                                 + 0.635395 w_k1_AXP_s001
                                 + 0.858236 w_k1_DIS_s001
                                 + 0.865786 w_k1_FDX_s001
                                 + 0.804194 w_k1_IBM_s001
                                 + 0.834167 w_k1_INTC_s001
                                 + 0.878835 w_k1_KO_s001
                                 + 0.706363 w_k1_MAT_s001
                                 + 1.01347 w_k1_MSFT_s001
                                 + 0.891687 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.829477 w_k1_ADM_s001
                                 + 1.05029 w_k1_AXP_s001
                                 + 1.10417 w_k1_DIS_s001
                                 + 1.11851 w_k1_FDX_s001
                                 + 0.782786 w_k1_IBM_s001
                                 + 1.2069 w_k1_INTC_s001
                                 + 0.926278 w_k1_KO_s001
                                 + 0.903772 w_k1_MAT_s001
                                 + 1.04565 w_k1_MSFT_s001
                                 + 0.907573 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.21501 w_k1_ADM_s001
                                 + 1.24402 w_k1_AXP_s001
                                 + 1.24264 w_k1_DIS_s001
                                 + 1.10913 w_k1_FDX_s001
                                 + 0.957066 w_k1_IBM_s001
                                 + 0.872933 w_k1_INTC_s001
                                 + 0.952458 w_k1_KO_s001
                                 + 1.25819 w_k1_MAT_s001
                                 + 0.985794 w_k1_MSFT_s001
                                 + 1.03893 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.11986 w_k1_ADM_s001
                                 + 1.1224 w_k1_AXP_s001 + 1.34102 w_k1_DIS_s001
                                 + 1.07112 w_k1_FDX_s001
                                 + 1.56822 w_k1_IBM_s001
                                 + 1.32063 w_k1_INTC_s001
                                 + 1.38391 w_k1_KO_s001 + 1.43028 w_k1_MAT_s001
                                 + 1.22291 w_k1_MSFT_s001
                                 + 1.17816 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.796453 w_k1_ADM_s001
                                 + 0.728353 w_k1_AXP_s001
                                 + 0.597735 w_k1_DIS_s001
                                 + 0.886979 w_k1_FDX_s001
                                 + 0.551046 w_k1_IBM_s001
                                 + 0.762716 w_k1_INTC_s001
                                 + 0.598183 w_k1_KO_s001
                                 + 0.744465 w_k1_MAT_s001
                                 + 0.804839 w_k1_MSFT_s001
                                 + 0.828771 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.85807 w_k1_ADM_s001
                                 + 0.89076 w_k1_AXP_s001
                                 + 0.946233 w_k1_DIS_s001
                                 + 0.979923 w_k1_FDX_s001
                                 + 1.24492 w_k1_IBM_s001
                                 + 0.865785 w_k1_INTC_s001
                                 + 1.10046 w_k1_KO_s001 + 1.15995 w_k1_MAT_s001
                                 + 0.983937 w_k1_MSFT_s001
                                 + 1.08764 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.05613 w_k1_ADM_s001
                                 + 1.26094 w_k1_AXP_s001
                                 + 1.67304 w_k1_DIS_s001
                                 + 1.40512 w_k1_FDX_s001
                                 + 1.19725 w_k1_IBM_s001
                                 + 1.23721 w_k1_INTC_s001 + 1.3829 w_k1_KO_s001
                                 + 1.30218 w_k1_MAT_s001
                                 + 1.31818 w_k1_MSFT_s001
                                 + 1.28354 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.678841 w_k1_ADM_s001
                                 + 0.766461 w_k1_AXP_s001
                                 + 0.768637 w_k1_DIS_s001
                                 + 0.824121 w_k1_FDX_s001
                                 + 0.988785 w_k1_IBM_s001
                                 + 0.97492 w_k1_INTC_s001
                                 + 0.931099 w_k1_KO_s001
                                 + 1.12442 w_k1_MAT_s001
                                 + 0.909964 w_k1_MSFT_s001
                                 + 0.901485 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.92285 w_k1_ADM_s002
                                 + 0.933193 w_k1_AXP_s002
                                 + 1.11285 w_k1_DIS_s002
                                 + 1.10615 w_k1_FDX_s002
                                 + 0.625165 w_k1_IBM_s002
                                 + 0.956737 w_k1_INTC_s002
                                 + 0.816684 w_k1_KO_s002
                                 + 0.621611 w_k1_MAT_s002
                                 + 1.12431 w_k1_MSFT_s002
                                 + 0.979565 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.853209 w_k1_ADM_s002
                                 + 0.910691 w_k1_AXP_s002
                                 + 1.2242 w_k1_DIS_s002 + 1.05536 w_k1_FDX_s002
                                 + 1.15806 w_k1_IBM_s002
                                 + 1.09661 w_k1_INTC_s002
                                 + 1.19392 w_k1_KO_s002
                                 + 0.910587 w_k1_MAT_s002
                                 + 1.06676 w_k1_MSFT_s002
                                 + 1.03502 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.770934 w_k1_ADM_s002
                                 + 0.723752 w_k1_AXP_s002
                                 + 1.17229 w_k1_DIS_s002
                                 + 0.999912 w_k1_FDX_s002
                                 + 0.757723 w_k1_IBM_s002
                                 + 1.21958 w_k1_INTC_s002
                                 + 0.957732 w_k1_KO_s002
                                 + 0.625197 w_k1_MAT_s002
                                 + 1.09114 w_k1_MSFT_s002
                                 + 0.890618 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.877475 w_k1_ADM_s002
                                 + 0.804735 w_k1_AXP_s002
                                 + 0.825768 w_k1_DIS_s002
                                 + 0.811699 w_k1_FDX_s002
                                 + 0.959319 w_k1_IBM_s002
                                 + 0.909218 w_k1_INTC_s002
                                 + 0.914133 w_k1_KO_s002
                                 + 1.08036 w_k1_MAT_s002
                                 + 0.82549 w_k1_MSFT_s002
                                 + 0.941459 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.904601 w_k1_ADM_s002
                                 + 1.19776 w_k1_AXP_s002
                                 + 1.06913 w_k1_DIS_s002 + 1.2053 w_k1_FDX_s002
                                 + 0.695997 w_k1_IBM_s002
                                 + 1.27647 w_k1_INTC_s002
                                 + 0.903685 w_k1_KO_s002
                                 + 0.729639 w_k1_MAT_s002
                                 + 1.1146 w_k1_MSFT_s002
                                 + 0.940619 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.826706 w_k1_ADM_s002
                                 + 0.770129 w_k1_AXP_s002
                                 + 0.659606 w_k1_DIS_s002
                                 + 0.837083 w_k1_FDX_s002
                                 + 0.607627 w_k1_IBM_s002
                                 + 0.835144 w_k1_INTC_s002
                                 + 0.713612 w_k1_KO_s002
                                 + 0.621256 w_k1_MAT_s002
                                 + 0.868815 w_k1_MSFT_s002
                                 + 0.790708 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.10465 w_k1_ADM_s002
                                 + 1.45923 w_k1_AXP_s002
                                 + 1.54128 w_k1_DIS_s002
                                 + 1.32382 w_k1_FDX_s002 + 1.3188 w_k1_IBM_s002
                                 + 1.33554 w_k1_INTC_s002
                                 + 1.36998 w_k1_KO_s002 + 1.17682 w_k1_MAT_s002
                                 + 1.32064 w_k1_MSFT_s002
                                 + 1.29488 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.784418 w_k1_ADM_s002
                                 + 0.978196 w_k1_AXP_s002
                                 + 1.14848 w_k1_DIS_s002
                                 + 0.889738 w_k1_FDX_s002
                                 + 1.50921 w_k1_IBM_s002
                                 + 0.992361 w_k1_INTC_s002
                                 + 1.1623 w_k1_KO_s002 + 1.21173 w_k1_MAT_s002
                                 + 1.06169 w_k1_MSFT_s002
                                 + 1.06926 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.42227 w_k1_ADM_s003
                                 + 1.47892 w_k1_AXP_s003
                                 + 1.31383 w_k1_DIS_s003 + 1.4002 w_k1_FDX_s003
                                 + 0.939703 w_k1_IBM_s003
                                 + 1.08154 w_k1_INTC_s003
                                 + 1.09447 w_k1_KO_s003 + 1.25373 w_k1_MAT_s003
                                 + 1.1888 w_k1_MSFT_s003 + 1.14922 w_k1_PG_s003
                                 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.08781 w_k1_ADM_s003
                                 + 0.941327 w_k1_AXP_s003
                                 + 0.799619 w_k1_DIS_s003
                                 + 0.808711 w_k1_FDX_s003
                                 + 1.17778 w_k1_IBM_s003 + 1.01 w_k1_INTC_s003
                                 + 1.02122 w_k1_KO_s003 + 1.02397 w_k1_MAT_s003
                                 + 0.98236 w_k1_MSFT_s003
                                 + 0.951026 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.788343 w_k1_ADM_s003
                                 + 0.640945 w_k1_AXP_s003
                                 + 0.692196 w_k1_DIS_s003
                                 + 0.777874 w_k1_FDX_s003
                                 + 0.706678 w_k1_IBM_s003
                                 + 0.889134 w_k1_INTC_s003
                                 + 0.897996 w_k1_KO_s003
                                 + 0.588433 w_k1_MAT_s003
                                 + 0.885705 w_k1_MSFT_s003
                                 + 0.81012 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.983349 w_k1_ADM_s003
                                 + 1.10796 w_k1_AXP_s003
                                 + 1.25904 w_k1_DIS_s003
                                 + 1.15865 w_k1_FDX_s003
                                 + 0.868002 w_k1_IBM_s003
                                 + 0.992889 w_k1_INTC_s003
                                 + 1.07784 w_k1_KO_s003
                                 + 0.863065 w_k1_MAT_s003
                                 + 1.16115 w_k1_MSFT_s003
                                 + 1.00246 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.826811 w_k1_ADM_s003
                                 + 0.917705 w_k1_AXP_s003
                                 + 0.809426 w_k1_DIS_s003
                                 + 0.951967 w_k1_FDX_s003
                                 + 0.855258 w_k1_IBM_s003
                                 + 0.844247 w_k1_INTC_s003
                                 + 0.868279 w_k1_KO_s003
                                 + 0.928997 w_k1_MAT_s003
                                 + 0.966536 w_k1_MSFT_s003
                                 + 0.99115 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.30635 w_k1_ADM_s003
                                 + 1.17175 w_k1_AXP_s003
                                 + 0.933125 w_k1_DIS_s003
                                 + 1.06031 w_k1_FDX_s003
                                 + 1.22042 w_k1_IBM_s003
                                 + 0.913794 w_k1_INTC_s003
                                 + 1.08396 w_k1_KO_s003 + 1.49468 w_k1_MAT_s003
                                 + 1.03509 w_k1_MSFT_s003
                                 + 1.14054 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.839817 w_k1_ADM_s003
                                 + 0.895476 w_k1_AXP_s003
                                 + 1.19965 w_k1_DIS_s003
                                 + 0.88271 w_k1_FDX_s003 + 1.2964 w_k1_IBM_s003
                                 + 0.970488 w_k1_INTC_s003
                                 + 1.25411 w_k1_KO_s003 + 1.14083 w_k1_MAT_s003
                                 + 1.01401 w_k1_MSFT_s003
                                 + 1.00187 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.28138 w_k1_ADM_s003
                                 + 1.19243 w_k1_AXP_s003
                                 + 1.62028 w_k1_DIS_s003
                                 + 1.24456 w_k1_FDX_s003
                                 + 1.33067 w_k1_IBM_s003
                                 + 0.884361 w_k1_INTC_s003
                                 + 1.50851 w_k1_KO_s003 + 1.75588 w_k1_MAT_s003
                                 + 1.2379 w_k1_MSFT_s003 + 1.25749 w_k1_PG_s003
                                 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.928699 w_k1_ADM_s004
                                 + 0.831252 w_k1_AXP_s004
                                 + 0.764523 w_k1_DIS_s004
                                 + 0.898747 w_k1_FDX_s004
                                 + 0.88472 w_k1_IBM_s004
                                 + 0.918366 w_k1_INTC_s004
                                 + 0.773393 w_k1_KO_s004
                                 + 1.24807 w_k1_MAT_s004
                                 + 0.995162 w_k1_MSFT_s004
                                 + 0.874197 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.861826 w_k1_ADM_s004
                                 + 0.711603 w_k1_AXP_s004
                                 + 0.952149 w_k1_DIS_s004
                                 + 0.933902 w_k1_FDX_s004
                                 + 0.782483 w_k1_IBM_s004
                                 + 0.94839 w_k1_INTC_s004
                                 + 1.01518 w_k1_KO_s004 + 1.33671 w_k1_MAT_s004
                                 + 1.08588 w_k1_MSFT_s004
                                 + 0.872683 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.0981 w_k1_ADM_s004
                                 + 0.98917 w_k1_AXP_s004
                                 + 1.05974 w_k1_DIS_s004
                                 + 1.03279 w_k1_FDX_s004
                                 + 0.941004 w_k1_IBM_s004
                                 + 1.17856 w_k1_INTC_s004 + 1.0915 w_k1_KO_s004
                                 + 0.857674 w_k1_MAT_s004
                                 + 1.20616 w_k1_MSFT_s004 + 1.0609 w_k1_PG_s004
                                 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.52192 w_k1_ADM_s004
                                 + 1.49637 w_k1_AXP_s004
                                 + 1.62398 w_k1_DIS_s004
                                 + 1.39285 w_k1_FDX_s004
                                 + 1.02101 w_k1_IBM_s004
                                 + 1.30059 w_k1_INTC_s004
                                 + 1.21699 w_k1_KO_s004 + 1.52423 w_k1_MAT_s004
                                 + 1.33397 w_k1_MSFT_s004
                                 + 1.23781 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.776851 w_k1_ADM_s004
                                 + 0.795159 w_k1_AXP_s004
                                 + 1.17304 w_k1_DIS_s004
                                 + 0.884152 w_k1_FDX_s004
                                 + 1.15011 w_k1_IBM_s004
                                 + 1.02558 w_k1_INTC_s004
                                 + 1.09236 w_k1_KO_s004 + 1.50002 w_k1_MAT_s004
                                 + 1.19855 w_k1_MSFT_s004
                                 + 0.98137 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.42141 w_k1_ADM_s004
                                 + 1.37518 w_k1_AXP_s004
                                 + 1.30699 w_k1_DIS_s004
                                 + 1.18353 w_k1_FDX_s004
                                 + 1.37816 w_k1_IBM_s004
                                 + 1.26853 w_k1_INTC_s004
                                 + 1.52108 w_k1_KO_s004 + 1.42348 w_k1_MAT_s004
                                 + 1.27401 w_k1_MSFT_s004 + 1.2041 w_k1_PG_s004
                                 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.00479 w_k1_ADM_s004
                                 + 1.02385 w_k1_AXP_s004
                                 + 0.940503 w_k1_DIS_s004
                                 + 0.901392 w_k1_FDX_s004
                                 + 1.33305 w_k1_IBM_s004
                                 + 0.839647 w_k1_INTC_s004
                                 + 1.17792 w_k1_KO_s004 + 1.68739 w_k1_MAT_s004
                                 + 0.96031 w_k1_MSFT_s004
                                 + 1.15208 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.885704 w_k1_ADM_s004
                                 + 0.668455 w_k1_AXP_s004
                                 + 0.665066 w_k1_DIS_s004
                                 + 0.744192 w_k1_FDX_s004
                                 + 0.908369 w_k1_IBM_s004
                                 + 0.829696 w_k1_INTC_s004
                                 + 0.719732 w_k1_KO_s004
                                 + 0.780803 w_k1_MAT_s004
                                 + 0.807622 w_k1_MSFT_s004
                                 + 0.787031 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.16692 w_k1_ADM_s005
                                 + 1.19926 w_k1_AXP_s005
                                 + 1.16105 w_k1_DIS_s005
                                 + 1.14086 w_k1_FDX_s005
                                 + 1.02673 w_k1_IBM_s005
                                 + 1.00705 w_k1_INTC_s005
                                 + 1.00069 w_k1_KO_s005 + 1.22909 w_k1_MAT_s005
                                 + 1.07237 w_k1_MSFT_s005
                                 + 1.05264 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.05107 w_k1_ADM_s005
                                 + 0.790858 w_k1_AXP_s005
                                 + 0.910945 w_k1_DIS_s005
                                 + 0.934643 w_k1_FDX_s005
                                 + 0.915244 w_k1_IBM_s005
                                 + 0.97508 w_k1_INTC_s005
                                 + 1.07271 w_k1_KO_s005
                                 + 0.986374 w_k1_MAT_s005
                                 + 0.999666 w_k1_MSFT_s005
                                 + 0.947449 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.25916 w_k1_ADM_s005
                                 + 1.51657 w_k1_AXP_s005
                                 + 1.62655 w_k1_DIS_s005
                                 + 1.34275 w_k1_FDX_s005
                                 + 1.32219 w_k1_IBM_s005
                                 + 0.997273 w_k1_INTC_s005
                                 + 1.6394 w_k1_KO_s005 + 1.50104 w_k1_MAT_s005
                                 + 1.14117 w_k1_MSFT_s005
                                 + 1.24106 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.10296 w_k1_ADM_s005
                                 + 1.04275 w_k1_AXP_s005
                                 + 1.06147 w_k1_DIS_s005
                                 + 1.02072 w_k1_FDX_s005
                                 + 1.39588 w_k1_IBM_s005
                                 + 1.02858 w_k1_INTC_s005
                                 + 1.21872 w_k1_KO_s005 + 1.24371 w_k1_MAT_s005
                                 + 1.07569 w_k1_MSFT_s005
                                 + 0.980947 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.1116 w_k1_ADM_s005
                                 + 1.0685 w_k1_AXP_s005 + 1.4515 w_k1_DIS_s005
                                 + 1.11372 w_k1_FDX_s005
                                 + 1.22343 w_k1_IBM_s005
                                 + 1.14079 w_k1_INTC_s005
                                 + 1.38202 w_k1_KO_s005
                                 + 0.922825 w_k1_MAT_s005
                                 + 1.16047 w_k1_MSFT_s005
                                 + 1.05532 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.675293 w_k1_ADM_s005
                                 + 0.898994 w_k1_AXP_s005
                                 + 0.909826 w_k1_DIS_s005
                                 + 0.799701 w_k1_FDX_s005
                                 + 1.36673 w_k1_IBM_s005
                                 + 0.892377 w_k1_INTC_s005
                                 + 1.22828 w_k1_KO_s005 + 1.44494 w_k1_MAT_s005
                                 + 0.980743 w_k1_MSFT_s005
                                 + 1.05362 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.14088 w_k1_ADM_s005
                                 + 0.868245 w_k1_AXP_s005
                                 + 0.933668 w_k1_DIS_s005
                                 + 1.03113 w_k1_FDX_s005
                                 + 0.642883 w_k1_IBM_s005
                                 + 1.04434 w_k1_INTC_s005
                                 + 0.913022 w_k1_KO_s005
                                 + 0.662671 w_k1_MAT_s005
                                 + 1.09555 w_k1_MSFT_s005
                                 + 0.965251 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.750833 w_k1_ADM_s005
                                 + 0.751958 w_k1_AXP_s005
                                 + 0.653374 w_k1_DIS_s005
                                 + 0.77321 w_k1_FDX_s005
                                 + 0.90003 w_k1_IBM_s005
                                 + 0.840345 w_k1_INTC_s005
                                 + 0.961017 w_k1_KO_s005
                                 + 0.932008 w_k1_MAT_s005
                                 + 0.871282 w_k1_MSFT_s005
                                 + 0.766119 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.06818 w_k1_ADM_s006
                                 + 1.20029 w_k1_AXP_s006
                                 + 0.970287 w_k1_DIS_s006
                                 + 1.20859 w_k1_FDX_s006
                                 + 0.964642 w_k1_IBM_s006
                                 + 0.907893 w_k1_INTC_s006
                                 + 0.927102 w_k1_KO_s006
                                 + 1.05277 w_k1_MAT_s006
                                 + 1.04467 w_k1_MSFT_s006
                                 + 1.15204 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.3237 w_k1_ADM_s006
                                 + 0.931422 w_k1_AXP_s006
                                 + 0.990355 w_k1_DIS_s006
                                 + 1.19293 w_k1_FDX_s006
                                 + 0.600002 w_k1_IBM_s006
                                 + 0.926467 w_k1_INTC_s006
                                 + 0.699716 w_k1_KO_s006
                                 + 0.53044 w_k1_MAT_s006
                                 + 1.05701 w_k1_MSFT_s006
                                 + 0.900932 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.97927 w_k1_ADM_s006
                                 + 0.814177 w_k1_AXP_s006
                                 + 0.59262 w_k1_DIS_s006
                                 + 0.876452 w_k1_FDX_s006
                                 + 0.678595 w_k1_IBM_s006
                                 + 0.788966 w_k1_INTC_s006
                                 + 0.574178 w_k1_KO_s006
                                 + 0.586226 w_k1_MAT_s006
                                 + 0.747683 w_k1_MSFT_s006
                                 + 0.763558 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.992198 w_k1_ADM_s006
                                 + 0.904669 w_k1_AXP_s006
                                 + 0.822994 w_k1_DIS_s006
                                 + 1.00174 w_k1_FDX_s006
                                 + 0.621484 w_k1_IBM_s006
                                 + 1.01837 w_k1_INTC_s006
                                 + 0.698135 w_k1_KO_s006
                                 + 0.596549 w_k1_MAT_s006
                                 + 1.02062 w_k1_MSFT_s006
                                 + 0.916459 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.11776 w_k1_ADM_s006
                                 + 0.886137 w_k1_AXP_s006
                                 + 1.02764 w_k1_DIS_s006
                                 + 1.13384 w_k1_FDX_s006
                                 + 0.78905 w_k1_IBM_s006
                                 + 0.966695 w_k1_INTC_s006
                                 + 0.742822 w_k1_KO_s006 + 0.4934 w_k1_MAT_s006
                                 + 1.05888 w_k1_MSFT_s006
                                 + 0.942381 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.05639 w_k1_ADM_s006
                                 + 0.792703 w_k1_AXP_s006
                                 + 0.891744 w_k1_DIS_s006
                                 + 0.900591 w_k1_FDX_s006
                                 + 1.27535 w_k1_IBM_s006
                                 + 0.949972 w_k1_INTC_s006
                                 + 1.00647 w_k1_KO_s006 + 1.17075 w_k1_MAT_s006
                                 + 1.05467 w_k1_MSFT_s006
                                 + 0.983462 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.08076 w_k1_ADM_s006
                                 + 1.23852 w_k1_AXP_s006
                                 + 1.42609 w_k1_DIS_s006
                                 + 1.44582 w_k1_FDX_s006
                                 + 0.871396 w_k1_IBM_s006
                                 + 1.15883 w_k1_INTC_s006
                                 + 1.00199 w_k1_KO_s006 + 0.66815 w_k1_MAT_s006
                                 + 1.1865 w_k1_MSFT_s006 + 1.18116 w_k1_PG_s006
                                 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.45895 w_k1_ADM_s006
                                 + 1.34763 w_k1_AXP_s006
                                 + 1.42152 w_k1_DIS_s006 + 1.1969 w_k1_FDX_s006
                                 + 1.46269 w_k1_IBM_s006
                                 + 1.26848 w_k1_INTC_s006
                                 + 1.19164 w_k1_KO_s006 + 1.39859 w_k1_MAT_s006
                                 + 1.10514 w_k1_MSFT_s006
                                 + 1.21195 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.89054 w_k1_ADM_s007
                                 + 0.875774 w_k1_AXP_s007
                                 + 0.930549 w_k1_DIS_s007
                                 + 0.885693 w_k1_FDX_s007
                                 + 1.32643 w_k1_IBM_s007
                                 + 1.01226 w_k1_INTC_s007
                                 + 1.15027 w_k1_KO_s007 + 1.20824 w_k1_MAT_s007
                                 + 1.01454 w_k1_MSFT_s007
                                 + 1.04429 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.841934 w_k1_ADM_s007
                                 + 1.2399 w_k1_AXP_s007 + 1.29196 w_k1_DIS_s007
                                 + 1.00559 w_k1_FDX_s007
                                 + 1.28438 w_k1_IBM_s007
                                 + 1.11131 w_k1_INTC_s007
                                 + 1.18596 w_k1_KO_s007 + 1.51048 w_k1_MAT_s007
                                 + 1.10151 w_k1_MSFT_s007
                                 + 1.14741 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.728984 w_k1_ADM_s007
                                 + 1.07018 w_k1_AXP_s007
                                 + 0.655977 w_k1_DIS_s007
                                 + 0.836642 w_k1_FDX_s007
                                 + 0.993181 w_k1_IBM_s007
                                 + 0.679103 w_k1_INTC_s007
                                 + 0.690038 w_k1_KO_s007
                                 + 1.46322 w_k1_MAT_s007
                                 + 0.740182 w_k1_MSFT_s007
                                 + 0.829873 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.14914 w_k1_ADM_s007
                                 + 1.76784 w_k1_AXP_s007
                                 + 1.68299 w_k1_DIS_s007
                                 + 1.43832 w_k1_FDX_s007
                                 + 1.29619 w_k1_IBM_s007
                                 + 1.06148 w_k1_INTC_s007
                                 + 1.19954 w_k1_KO_s007 + 1.70832 w_k1_MAT_s007
                                 + 1.14487 w_k1_MSFT_s007
                                 + 1.32668 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.35706 w_k1_ADM_s007
                                 + 1.54542 w_k1_AXP_s007
                                 + 1.18224 w_k1_DIS_s007
                                 + 1.27513 w_k1_FDX_s007
                                 + 1.03073 w_k1_IBM_s007
                                 + 0.899661 w_k1_INTC_s007
                                 + 1.17101 w_k1_KO_s007 + 1.6619 w_k1_MAT_s007
                                 + 1.02631 w_k1_MSFT_s007
                                 + 1.23456 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.871915 w_k1_ADM_s007
                                 + 0.76277 w_k1_AXP_s007
                                 + 1.19866 w_k1_DIS_s007
                                 + 0.906517 w_k1_FDX_s007
                                 + 1.16107 w_k1_IBM_s007
                                 + 0.813004 w_k1_INTC_s007
                                 + 1.10037 w_k1_KO_s007
                                 + 0.829915 w_k1_MAT_s007
                                 + 0.897886 w_k1_MSFT_s007
                                 + 1.07318 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.38662 w_k1_ADM_s007
                                 + 1.32668 w_k1_AXP_s007
                                 + 1.23465 w_k1_DIS_s007
                                 + 1.17225 w_k1_FDX_s007
                                 + 1.27748 w_k1_IBM_s007
                                 + 0.973929 w_k1_INTC_s007
                                 + 1.06585 w_k1_KO_s007 + 1.38223 w_k1_MAT_s007
                                 + 1.05105 w_k1_MSFT_s007
                                 + 1.13727 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.885581 w_k1_ADM_s007
                                 + 0.879828 w_k1_AXP_s007
                                 + 0.861994 w_k1_DIS_s007
                                 + 0.984768 w_k1_FDX_s007
                                 + 0.697734 w_k1_IBM_s007
                                 + 0.879426 w_k1_INTC_s007
                                 + 0.803104 w_k1_KO_s007
                                 + 0.721259 w_k1_MAT_s007
                                 + 1.02813 w_k1_MSFT_s007
                                 + 0.970542 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.09847 w_k1_ADM_s010
                                 + 1.2461 w_k1_AXP_s010 + 1.17899 w_k1_DIS_s010
                                 + 1.20181 w_k1_FDX_s010
                                 + 1.00491 w_k1_IBM_s010
                                 + 1.20009 w_k1_INTC_s010 + 1.026 w_k1_KO_s010
                                 + 1.15846 w_k1_MAT_s010
                                 + 1.11313 w_k1_MSFT_s010
                                 + 1.05585 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.917122 w_k1_ADM_s010
                                 + 0.851093 w_k1_AXP_s010
                                 + 0.839699 w_k1_DIS_s010
                                 + 1.03942 w_k1_FDX_s010
                                 + 0.561872 w_k1_IBM_s010
                                 + 0.911124 w_k1_INTC_s010
                                 + 0.720873 w_k1_KO_s010
                                 + 0.523694 w_k1_MAT_s010
                                 + 1.02278 w_k1_MSFT_s010
                                 + 0.838105 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.893375 w_k1_ADM_s010
                                 + 0.821063 w_k1_AXP_s010
                                 + 1.17941 w_k1_DIS_s010
                                 + 1.12766 w_k1_FDX_s010 + 0.7144 w_k1_IBM_s010
                                 + 0.831501 w_k1_INTC_s010
                                 + 0.906406 w_k1_KO_s010
                                 + 0.614647 w_k1_MAT_s010
                                 + 1.09691 w_k1_MSFT_s010
                                 + 0.964069 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.38684 w_k1_ADM_s010
                                 + 1.52805 w_k1_AXP_s010
                                 + 1.66298 w_k1_DIS_s010
                                 + 1.43421 w_k1_FDX_s010
                                 + 1.00413 w_k1_IBM_s010
                                 + 1.28241 w_k1_INTC_s010
                                 + 1.26421 w_k1_KO_s010 + 1.11758 w_k1_MAT_s010
                                 + 1.29873 w_k1_MSFT_s010
                                 + 1.23778 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.73456 w_k1_ADM_s010
                                 + 0.671337 w_k1_AXP_s010
                                 + 0.695465 w_k1_DIS_s010
                                 + 0.784568 w_k1_FDX_s010
                                 + 0.713043 w_k1_IBM_s010
                                 + 0.777766 w_k1_INTC_s010
                                 + 0.747519 w_k1_KO_s010
                                 + 0.577186 w_k1_MAT_s010
                                 + 0.81411 w_k1_MSFT_s010
                                 + 0.809237 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.740068 w_k1_ADM_s010
                                 + 0.839566 w_k1_AXP_s010
                                 + 0.830308 w_k1_DIS_s010
                                 + 0.874764 w_k1_FDX_s010
                                 + 1.01645 w_k1_IBM_s010
                                 + 0.975723 w_k1_INTC_s010
                                 + 0.87789 w_k1_KO_s010
                                 + 0.965439 w_k1_MAT_s010
                                 + 0.945508 w_k1_MSFT_s010
                                 + 0.834782 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.00341 w_k1_ADM_s010
                                 + 1.19531 w_k1_AXP_s010
                                 + 1.23318 w_k1_DIS_s010
                                 + 1.10309 w_k1_FDX_s010
                                 + 1.73264 w_k1_IBM_s010
                                 + 0.974192 w_k1_INTC_s010
                                 + 1.23526 w_k1_KO_s010 + 1.59387 w_k1_MAT_s010
                                 + 1.00306 w_k1_MSFT_s010
                                 + 1.15184 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.04691 w_k1_ADM_s010
                                 + 1.11122 w_k1_AXP_s010
                                 + 1.40473 w_k1_DIS_s010
                                 + 1.21978 w_k1_FDX_s010
                                 + 1.20125 w_k1_IBM_s010
                                 + 1.35877 w_k1_INTC_s010 + 1.2088 w_k1_KO_s010
                                 + 1.15581 w_k1_MAT_s010
                                 + 1.3334 w_k1_MSFT_s010 + 1.14794 w_k1_PG_s010
                                 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.14464 w_k1_ADM_s011
                                 + 0.943402 w_k1_AXP_s011
                                 + 1.22697 w_k1_DIS_s011
                                 + 1.27314 w_k1_FDX_s011
                                 + 0.591765 w_k1_IBM_s011
                                 + 1.35448 w_k1_INTC_s011
                                 + 0.974933 w_k1_KO_s011
                                 + 0.361662 w_k1_MAT_s011
                                 + 1.28033 w_k1_MSFT_s011
                                 + 1.00439 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.1613 w_k1_ADM_s011
                                 + 1.2004 w_k1_AXP_s011 + 0.96408 w_k1_DIS_s011
                                 + 1.03061 w_k1_FDX_s011
                                 + 0.945903 w_k1_IBM_s011
                                 + 1.21882 w_k1_INTC_s011
                                 + 0.822848 w_k1_KO_s011
                                 + 0.702258 w_k1_MAT_s011
                                 + 0.947758 w_k1_MSFT_s011
                                 + 0.929978 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.812117 w_k1_ADM_s011
                                 + 0.96497 w_k1_AXP_s011
                                 + 0.76373 w_k1_DIS_s011
                                 + 1.03316 w_k1_FDX_s011
                                 + 0.558377 w_k1_IBM_s011
                                 + 1.33936 w_k1_INTC_s011
                                 + 0.625816 w_k1_KO_s011
                                 + 0.328166 w_k1_MAT_s011
                                 + 0.96203 w_k1_MSFT_s011
                                 + 0.854034 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.794505 w_k1_ADM_s011
                                 + 1.40602 w_k1_AXP_s011
                                 + 1.50978 w_k1_DIS_s011
                                 + 1.13672 w_k1_FDX_s011
                                 + 2.00877 w_k1_IBM_s011
                                 + 1.37424 w_k1_INTC_s011
                                 + 1.28911 w_k1_KO_s011 + 1.92853 w_k1_MAT_s011
                                 + 1.27429 w_k1_MSFT_s011
                                 + 1.13398 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.767744 w_k1_ADM_s011
                                 + 0.658224 w_k1_AXP_s011
                                 + 0.717895 w_k1_DIS_s011
                                 + 0.747328 w_k1_FDX_s011
                                 + 0.692153 w_k1_IBM_s011
                                 + 1.15628 w_k1_INTC_s011
                                 + 0.80718 w_k1_KO_s011
                                 + 0.544265 w_k1_MAT_s011
                                 + 0.865174 w_k1_MSFT_s011
                                 + 0.853991 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.835772 w_k1_ADM_s011
                                 + 0.873555 w_k1_AXP_s011
                                 + 1.13493 w_k1_DIS_s011
                                 + 0.912467 w_k1_FDX_s011
                                 + 1.35203 w_k1_IBM_s011 + 1.125 w_k1_INTC_s011
                                 + 1.1553 w_k1_KO_s011 + 1.1233 w_k1_MAT_s011
                                 + 1.04104 w_k1_MSFT_s011
                                 + 1.00016 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.686204 w_k1_ADM_s011
                                 + 0.770579 w_k1_AXP_s011
                                 + 0.633451 w_k1_DIS_s011
                                 + 0.695667 w_k1_FDX_s011
                                 + 1.32811 w_k1_IBM_s011
                                 + 1.19367 w_k1_INTC_s011
                                 + 0.907961 w_k1_KO_s011
                                 + 0.849816 w_k1_MAT_s011
                                 + 0.861442 w_k1_MSFT_s011
                                 + 0.846721 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.177 w_k1_ADM_s011
                                 + 1.51519 w_k1_AXP_s011
                                 + 1.46513 w_k1_DIS_s011
                                 + 1.39288 w_k1_FDX_s011
                                 + 1.03575 w_k1_IBM_s011
                                 + 1.50615 w_k1_INTC_s011
                                 + 1.11786 w_k1_KO_s011 + 1.01629 w_k1_MAT_s011
                                 + 1.27949 w_k1_MSFT_s011 + 1.065 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.11494 w_k1_ADM_s012
                                 + 1.42218 w_k1_AXP_s012
                                 + 1.62836 w_k1_DIS_s012
                                 + 1.24014 w_k1_FDX_s012
                                 + 1.67505 w_k1_IBM_s012
                                 + 1.23709 w_k1_INTC_s012 + 1.4197 w_k1_KO_s012
                                 + 2.02363 w_k1_MAT_s012
                                 + 1.27716 w_k1_MSFT_s012
                                 + 1.26244 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.919328 w_k1_ADM_s012
                                 + 1.23083 w_k1_AXP_s012
                                 + 1.06685 w_k1_DIS_s012
                                 + 1.01402 w_k1_FDX_s012
                                 + 1.18326 w_k1_IBM_s012
                                 + 1.12358 w_k1_INTC_s012
                                 + 0.977459 w_k1_KO_s012
                                 + 1.26553 w_k1_MAT_s012
                                 + 0.955981 w_k1_MSFT_s012
                                 + 1.13604 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.976315 w_k1_ADM_s012
                                 + 0.758307 w_k1_AXP_s012
                                 + 0.845333 w_k1_DIS_s012
                                 + 0.909698 w_k1_FDX_s012
                                 + 0.857109 w_k1_IBM_s012
                                 + 0.818252 w_k1_INTC_s012
                                 + 0.759683 w_k1_KO_s012 + 0.9386 w_k1_MAT_s012
                                 + 0.835017 w_k1_MSFT_s012
                                 + 0.960719 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.46072 w_k1_ADM_s012
                                 + 1.27503 w_k1_AXP_s012
                                 + 1.33314 w_k1_DIS_s012
                                 + 1.28281 w_k1_FDX_s012
                                 + 0.865871 w_k1_IBM_s012
                                 + 1.30532 w_k1_INTC_s012
                                 + 0.885433 w_k1_KO_s012
                                 + 0.680633 w_k1_MAT_s012
                                 + 1.19237 w_k1_MSFT_s012
                                 + 1.10617 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.12162 w_k1_ADM_s012
                                 + 1.42135 w_k1_AXP_s012 + 1.5673 w_k1_DIS_s012
                                 + 1.27572 w_k1_FDX_s012
                                 + 1.26203 w_k1_IBM_s012
                                 + 1.27935 w_k1_INTC_s012
                                 + 1.15332 w_k1_KO_s012 + 1.86698 w_k1_MAT_s012
                                 + 1.32796 w_k1_MSFT_s012
                                 + 1.23868 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.06932 w_k1_ADM_s012
                                 + 1.14243 w_k1_AXP_s012
                                 + 0.931661 w_k1_DIS_s012
                                 + 1.0337 w_k1_FDX_s012
                                 + 0.816997 w_k1_IBM_s012
                                 + 1.08253 w_k1_INTC_s012
                                 + 0.879472 w_k1_KO_s012
                                 + 1.21738 w_k1_MAT_s012
                                 + 1.02485 w_k1_MSFT_s012
                                 + 1.03917 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.804484 w_k1_ADM_s012
                                 + 0.622039 w_k1_AXP_s012
                                 + 0.692869 w_k1_DIS_s012
                                 + 0.78502 w_k1_FDX_s012
                                 + 0.66076 w_k1_IBM_s012
                                 + 0.889822 w_k1_INTC_s012
                                 + 0.709326 w_k1_KO_s012
                                 + 0.465096 w_k1_MAT_s012
                                 + 0.773783 w_k1_MSFT_s012
                                 + 0.768263 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.628698 w_k1_ADM_s012
                                 + 0.947038 w_k1_AXP_s012
                                 + 1.1301 w_k1_DIS_s012
                                 + 0.912954 w_k1_FDX_s012
                                 + 1.45633 w_k1_IBM_s012
                                 + 1.00151 w_k1_INTC_s012
                                 + 1.04196 w_k1_KO_s012 + 1.74331 w_k1_MAT_s012
                                 + 0.998955 w_k1_MSFT_s012 + 1.015 w_k1_PG_s012
                                 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.682673 w_k1_ADM_s013
                                 + 0.631281 w_k1_AXP_s013
                                 + 0.643715 w_k1_DIS_s013
                                 + 0.790678 w_k1_FDX_s013
                                 + 0.752097 w_k1_IBM_s013
                                 + 0.783274 w_k1_INTC_s013
                                 + 0.66313 w_k1_KO_s013
                                 + 0.742581 w_k1_MAT_s013
                                 + 0.751354 w_k1_MSFT_s013
                                 + 0.747161 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.840947 w_k1_ADM_s013
                                 + 0.764637 w_k1_AXP_s013
                                 + 0.766087 w_k1_DIS_s013
                                 + 0.940795 w_k1_FDX_s013
                                 + 0.631341 w_k1_IBM_s013
                                 + 0.994577 w_k1_INTC_s013
                                 + 0.715128 w_k1_KO_s013
                                 + 0.605782 w_k1_MAT_s013
                                 + 0.865759 w_k1_MSFT_s013
                                 + 0.827107 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.01215 w_k1_ADM_s013
                                 + 0.89223 w_k1_AXP_s013
                                 + 1.00384 w_k1_DIS_s013
                                 + 0.974995 w_k1_FDX_s013
                                 + 0.977062 w_k1_IBM_s013
                                 + 0.981151 w_k1_INTC_s013
                                 + 0.953142 w_k1_KO_s013
                                 + 0.952476 w_k1_MAT_s013
                                 + 0.913644 w_k1_MSFT_s013
                                 + 0.944849 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.10158 w_k1_ADM_s013
                                 + 1.24783 w_k1_AXP_s013
                                 + 0.941242 w_k1_DIS_s013
                                 + 1.17277 w_k1_FDX_s013
                                 + 0.840672 w_k1_IBM_s013
                                 + 1.26466 w_k1_INTC_s013
                                 + 0.957851 w_k1_KO_s013 + 1.1965 w_k1_MAT_s013
                                 + 1.18585 w_k1_MSFT_s013
                                 + 1.02826 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.740384 w_k1_ADM_s013
                                 + 0.940866 w_k1_AXP_s013
                                 + 0.9497 w_k1_DIS_s013
                                 + 0.898372 w_k1_FDX_s013
                                 + 1.29601 w_k1_IBM_s013
                                 + 1.31051 w_k1_INTC_s013
                                 + 1.05527 w_k1_KO_s013 + 1.12007 w_k1_MAT_s013
                                 + 1.01197 w_k1_MSFT_s013
                                 + 0.971422 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.15735 w_k1_ADM_s013
                                 + 1.39234 w_k1_AXP_s013
                                 + 1.37888 w_k1_DIS_s013
                                 + 1.29129 w_k1_FDX_s013
                                 + 1.13075 w_k1_IBM_s013
                                 + 1.45392 w_k1_INTC_s013 + 1.0888 w_k1_KO_s013
                                 + 1.2174 w_k1_MAT_s013
                                 + 1.23648 w_k1_MSFT_s013
                                 + 1.13559 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.94228 w_k1_ADM_s013
                                 + 1.1966 w_k1_AXP_s013 + 1.3325 w_k1_DIS_s013
                                 + 1.08369 w_k1_FDX_s013
                                 + 1.77814 w_k1_IBM_s013
                                 + 1.07673 w_k1_INTC_s013
                                 + 1.40088 w_k1_KO_s013 + 1.71701 w_k1_MAT_s013
                                 + 1.19931 w_k1_MSFT_s013 + 1.2624 w_k1_PG_s013
                                 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.95664 w_k1_ADM_s013
                                 + 0.87226 w_k1_AXP_s013
                                 + 1.12842 w_k1_DIS_s013
                                 + 1.10914 w_k1_FDX_s013
                                 + 0.788176 w_k1_IBM_s013
                                 + 1.25379 w_k1_INTC_s013
                                 + 0.975134 w_k1_KO_s013
                                 + 0.729108 w_k1_MAT_s013
                                 + 1.21305 w_k1_MSFT_s013
                                 + 1.02407 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.2269 w_k1_ADM_s014
                                 + 1.41802 w_k1_AXP_s014
                                 + 1.61052 w_k1_DIS_s014
                                 + 1.39793 w_k1_FDX_s014
                                 + 1.13932 w_k1_IBM_s014
                                 + 1.34399 w_k1_INTC_s014
                                 + 1.46071 w_k1_KO_s014 + 1.27393 w_k1_MAT_s014
                                 + 1.40124 w_k1_MSFT_s014
                                 + 1.30998 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.01024 w_k1_ADM_s014
                                 + 1.07036 w_k1_AXP_s014
                                 + 1.29408 w_k1_DIS_s014
                                 + 1.11482 w_k1_FDX_s014
                                 + 0.933922 w_k1_IBM_s014
                                 + 1.17059 w_k1_INTC_s014
                                 + 1.02134 w_k1_KO_s014
                                 + 0.884272 w_k1_MAT_s014
                                 + 1.03209 w_k1_MSFT_s014
                                 + 1.02974 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.949615 w_k1_ADM_s014
                                 + 0.777726 w_k1_AXP_s014
                                 + 0.820686 w_k1_DIS_s014
                                 + 0.89533 w_k1_FDX_s014
                                 + 0.719267 w_k1_IBM_s014
                                 + 0.833697 w_k1_INTC_s014
                                 + 0.851356 w_k1_KO_s014
                                 + 0.677308 w_k1_MAT_s014
                                 + 0.969227 w_k1_MSFT_s014
                                 + 0.951259 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.19607 w_k1_ADM_s014
                                 + 1.31088 w_k1_AXP_s014
                                 + 0.996653 w_k1_DIS_s014
                                 + 1.10455 w_k1_FDX_s014
                                 + 0.976334 w_k1_IBM_s014
                                 + 0.97168 w_k1_INTC_s014
                                 + 1.01039 w_k1_KO_s014
                                 + 0.884618 w_k1_MAT_s014
                                 + 0.989188 w_k1_MSFT_s014
                                 + 1.07145 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.776296 w_k1_ADM_s014
                                 + 0.831492 w_k1_AXP_s014
                                 + 0.584012 w_k1_DIS_s014
                                 + 0.733609 w_k1_FDX_s014
                                 + 0.93308 w_k1_IBM_s014
                                 + 0.837416 w_k1_INTC_s014
                                 + 0.845795 w_k1_KO_s014
                                 + 0.89562 w_k1_MAT_s014
                                 + 0.835856 w_k1_MSFT_s014
                                 + 0.817486 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.811232 w_k1_ADM_s014
                                 + 0.923422 w_k1_AXP_s014
                                 + 0.756922 w_k1_DIS_s014
                                 + 0.842794 w_k1_FDX_s014
                                 + 1.33242 w_k1_IBM_s014
                                 + 0.94453 w_k1_INTC_s014
                                 + 1.07104 w_k1_KO_s014 + 1.26913 w_k1_MAT_s014
                                 + 0.920169 w_k1_MSFT_s014
                                 + 1.12289 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.03834 w_k1_ADM_s014
                                 + 1.25954 w_k1_AXP_s014
                                 + 0.96089 w_k1_DIS_s014
                                 + 1.04749 w_k1_FDX_s014
                                 + 1.23806 w_k1_IBM_s014
                                 + 1.02749 w_k1_INTC_s014
                                 + 1.20164 w_k1_KO_s014 + 1.3221 w_k1_MAT_s014
                                 + 1.07843 w_k1_MSFT_s014
                                 + 1.09589 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.946695 w_k1_ADM_s014
                                 + 1.19286 w_k1_AXP_s014
                                 + 1.52121 w_k1_DIS_s014
                                 + 1.15004 w_k1_FDX_s014 + 1.4371 w_k1_IBM_s014
                                 + 1.1248 w_k1_INTC_s014 + 1.59451 w_k1_KO_s014
                                 + 1.47857 w_k1_MAT_s014
                                 + 1.22941 w_k1_MSFT_s014
                                 + 1.45012 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.696821 w_k1_ADM_s015
                                 + 1.07329 w_k1_AXP_s015 + 1.055 w_k1_DIS_s015
                                 + 0.953997 w_k1_FDX_s015
                                 + 1.34834 w_k1_IBM_s015
                                 + 1.48235 w_k1_INTC_s015
                                 + 1.31602 w_k1_KO_s015
                                 + 0.911379 w_k1_MAT_s015
                                 + 0.989532 w_k1_MSFT_s015
                                 + 1.14053 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.11576 w_k1_ADM_s015
                                 + 1.33835 w_k1_AXP_s015
                                 + 1.22723 w_k1_DIS_s015
                                 + 1.27669 w_k1_FDX_s015
                                 + 1.14994 w_k1_IBM_s015
                                 + 1.53505 w_k1_INTC_s015
                                 + 1.16423 w_k1_KO_s015 + 1.28582 w_k1_MAT_s015
                                 + 1.2249 w_k1_MSFT_s015 + 1.28947 w_k1_PG_s015
                                 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.12859 w_k1_ADM_s015
                                 + 1.34389 w_k1_AXP_s015
                                 + 1.08638 w_k1_DIS_s015
                                 + 1.18946 w_k1_FDX_s015
                                 + 0.735929 w_k1_IBM_s015
                                 + 1.00348 w_k1_INTC_s015
                                 + 0.970716 w_k1_KO_s015
                                 + 0.994363 w_k1_MAT_s015
                                 + 1.00398 w_k1_MSFT_s015
                                 + 1.10488 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.728194 w_k1_ADM_s015
                                 + 0.804014 w_k1_AXP_s015
                                 + 0.705215 w_k1_DIS_s015
                                 + 0.767858 w_k1_FDX_s015
                                 + 0.922466 w_k1_IBM_s015
                                 + 1.11521 w_k1_INTC_s015
                                 + 0.800323 w_k1_KO_s015
                                 + 0.691264 w_k1_MAT_s015
                                 + 0.80835 w_k1_MSFT_s015
                                 + 0.951112 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.764053 w_k1_ADM_s015
                                 + 1.02728 w_k1_AXP_s015
                                 + 0.840388 w_k1_DIS_s015
                                 + 0.99093 w_k1_FDX_s015
                                 + 1.13006 w_k1_IBM_s015
                                 + 1.18678 w_k1_INTC_s015
                                 + 0.935694 w_k1_KO_s015
                                 + 1.04065 w_k1_MAT_s015
                                 + 0.999967 w_k1_MSFT_s015
                                 + 1.0747 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.882554 w_k1_ADM_s015
                                 + 1.06216 w_k1_AXP_s015
                                 + 1.38864 w_k1_DIS_s015 + 1.0709 w_k1_FDX_s015
                                 + 1.20971 w_k1_IBM_s015
                                 + 1.47414 w_k1_INTC_s015
                                 + 1.35467 w_k1_KO_s015 + 1.18166 w_k1_MAT_s015
                                 + 1.22349 w_k1_MSFT_s015
                                 + 1.14618 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.722814 w_k1_ADM_s015
                                 + 0.770456 w_k1_AXP_s015
                                 + 0.975463 w_k1_DIS_s015
                                 + 0.950455 w_k1_FDX_s015
                                 + 0.769914 w_k1_IBM_s015
                                 + 1.06902 w_k1_INTC_s015
                                 + 0.932071 w_k1_KO_s015
                                 + 0.56349 w_k1_MAT_s015
                                 + 1.03367 w_k1_MSFT_s015
                                 + 0.921807 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.44774 w_k1_ADM_s015
                                 + 1.31354 w_k1_AXP_s015
                                 + 1.88436 w_k1_DIS_s015 + 1.4855 w_k1_FDX_s015
                                 + 1.18032 w_k1_IBM_s015
                                 + 1.54439 w_k1_INTC_s015
                                 + 1.31883 w_k1_KO_s015
                                 + 0.863238 w_k1_MAT_s015
                                 + 1.38416 w_k1_MSFT_s015
                                 + 1.24066 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.03833 w_k1_ADM_s016
                                 + 0.907408 w_k1_AXP_s016
                                 + 0.853726 w_k1_DIS_s016
                                 + 0.947975 w_k1_FDX_s016
                                 + 0.801008 w_k1_IBM_s016
                                 + 0.92186 w_k1_INTC_s016
                                 + 0.837683 w_k1_KO_s016
                                 + 0.632129 w_k1_MAT_s016
                                 + 0.961183 w_k1_MSFT_s016
                                 + 0.883722 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.986907 w_k1_ADM_s016
                                 + 1.43787 w_k1_AXP_s016
                                 + 1.26531 w_k1_DIS_s016
                                 + 1.22601 w_k1_FDX_s016
                                 + 1.63585 w_k1_IBM_s016
                                 + 1.46467 w_k1_INTC_s016 + 1.2159 w_k1_KO_s016
                                 + 1.49647 w_k1_MAT_s016
                                 + 1.1091 w_k1_MSFT_s016 + 1.09381 w_k1_PG_s016
                                 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.10854 w_k1_ADM_s016
                                 + 1.05492 w_k1_AXP_s016
                                 + 1.11755 w_k1_DIS_s016
                                 + 1.17491 w_k1_FDX_s016
                                 + 0.863233 w_k1_IBM_s016
                                 + 1.15777 w_k1_INTC_s016
                                 + 0.949805 w_k1_KO_s016
                                 + 0.896252 w_k1_MAT_s016
                                 + 1.27627 w_k1_MSFT_s016
                                 + 1.01707 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.15231 w_k1_ADM_s016
                                 + 1.07856 w_k1_AXP_s016
                                 + 1.23731 w_k1_DIS_s016
                                 + 1.17928 w_k1_FDX_s016
                                 + 1.20372 w_k1_IBM_s016
                                 + 1.22618 w_k1_INTC_s016
                                 + 1.02098 w_k1_KO_s016 + 1.21175 w_k1_MAT_s016
                                 + 1.28562 w_k1_MSFT_s016
                                 + 1.06126 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.05617 w_k1_ADM_s016
                                 + 0.92956 w_k1_AXP_s016
                                 + 0.893231 w_k1_DIS_s016
                                 + 0.885883 w_k1_FDX_s016
                                 + 1.09086 w_k1_IBM_s016
                                 + 0.936594 w_k1_INTC_s016
                                 + 0.85615 w_k1_KO_s016
                                 + 0.892692 w_k1_MAT_s016
                                 + 0.876471 w_k1_MSFT_s016
                                 + 0.896264 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.959514 w_k1_ADM_s016
                                 + 0.638089 w_k1_AXP_s016
                                 + 0.795275 w_k1_DIS_s016
                                 + 0.780058 w_k1_FDX_s016
                                 + 1.28014 w_k1_IBM_s016
                                 + 0.917821 w_k1_INTC_s016
                                 + 1.00703 w_k1_KO_s016 + 1.03704 w_k1_MAT_s016
                                 + 0.901303 w_k1_MSFT_s016
                                 + 0.920858 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.853741 w_k1_ADM_s016
                                 + 0.693306 w_k1_AXP_s016
                                 + 0.794757 w_k1_DIS_s016
                                 + 0.810078 w_k1_FDX_s016
                                 + 0.97291 w_k1_IBM_s016
                                 + 0.866957 w_k1_INTC_s016
                                 + 0.826263 w_k1_KO_s016
                                 + 0.898911 w_k1_MAT_s016
                                 + 0.869338 w_k1_MSFT_s016
                                 + 0.815764 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.48054 w_k1_ADM_s016
                                 + 1.40178 w_k1_AXP_s016
                                 + 1.69913 w_k1_DIS_s016
                                 + 1.44985 w_k1_FDX_s016
                                 + 1.01729 w_k1_IBM_s016
                                 + 1.06216 w_k1_INTC_s016
                                 + 1.14395 w_k1_KO_s016 + 1.09429 w_k1_MAT_s016
                                 + 1.26618 w_k1_MSFT_s016 + 1.1046 w_k1_PG_s016
                                 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.778882 w_k1_ADM_s017
                                 + 0.591306 w_k1_AXP_s017
                                 + 0.509919 w_k1_DIS_s017
                                 + 0.697862 w_k1_FDX_s017
                                 + 0.900767 w_k1_IBM_s017
                                 + 0.68363 w_k1_INTC_s017
                                 + 0.767605 w_k1_KO_s017
                                 + 0.593301 w_k1_MAT_s017
                                 + 0.712935 w_k1_MSFT_s017
                                 + 0.733823 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.40466 w_k1_ADM_s017
                                 + 1.25434 w_k1_AXP_s017
                                 + 1.00553 w_k1_DIS_s017 + 1.1796 w_k1_FDX_s017
                                 + 1.04391 w_k1_IBM_s017
                                 + 0.997502 w_k1_INTC_s017
                                 + 0.945778 w_k1_KO_s017
                                 + 1.40148 w_k1_MAT_s017
                                 + 1.12693 w_k1_MSFT_s017
                                 + 1.01452 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.01712 w_k1_ADM_s017
                                 + 0.831164 w_k1_AXP_s017
                                 + 0.866544 w_k1_DIS_s017
                                 + 0.924846 w_k1_FDX_s017
                                 + 1.13601 w_k1_IBM_s017
                                 + 1.09744 w_k1_INTC_s017
                                 + 0.983456 w_k1_KO_s017
                                 + 1.07991 w_k1_MAT_s017
                                 + 0.963503 w_k1_MSFT_s017
                                 + 0.888592 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.662183 w_k1_ADM_s017
                                 + 1.07696 w_k1_AXP_s017
                                 + 1.02954 w_k1_DIS_s017
                                 + 0.856312 w_k1_FDX_s017
                                 + 1.67656 w_k1_IBM_s017
                                 + 1.20319 w_k1_INTC_s017
                                 + 1.34852 w_k1_KO_s017 + 1.31385 w_k1_MAT_s017
                                 + 1.05511 w_k1_MSFT_s017
                                 + 0.980543 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.36377 w_k1_ADM_s017
                                 + 1.2273 w_k1_AXP_s017 + 1.39213 w_k1_DIS_s017
                                 + 1.44869 w_k1_FDX_s017
                                 + 0.994516 w_k1_IBM_s017
                                 + 1.26669 w_k1_INTC_s017
                                 + 0.988489 w_k1_KO_s017
                                 + 1.19523 w_k1_MAT_s017
                                 + 1.33842 w_k1_MSFT_s017
                                 + 1.06268 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.980458 w_k1_ADM_s017
                                 + 0.899132 w_k1_AXP_s017
                                 + 0.68438 w_k1_DIS_s017
                                 + 0.92562 w_k1_FDX_s017
                                 + 0.745325 w_k1_IBM_s017
                                 + 1.03749 w_k1_INTC_s017
                                 + 0.742972 w_k1_KO_s017
                                 + 0.681659 w_k1_MAT_s017
                                 + 0.834182 w_k1_MSFT_s017
                                 + 0.87035 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.33358 w_k1_ADM_s017
                                 + 0.981273 w_k1_AXP_s017
                                 + 0.917048 w_k1_DIS_s017
                                 + 1.23785 w_k1_FDX_s017
                                 + 0.645437 w_k1_IBM_s017
                                 + 1.00121 w_k1_INTC_s017
                                 + 0.83376 w_k1_KO_s017
                                 + 0.715873 w_k1_MAT_s017
                                 + 1.12168 w_k1_MSFT_s017
                                 + 1.05723 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.45017 w_k1_ADM_s017
                                 + 1.20578 w_k1_AXP_s017
                                 + 1.27861 w_k1_DIS_s017
                                 + 1.24354 w_k1_FDX_s017
                                 + 1.34603 w_k1_IBM_s017
                                 + 1.48773 w_k1_INTC_s017 + 1.2559 w_k1_KO_s017
                                 + 1.4146 w_k1_MAT_s017
                                 + 1.55798 w_k1_MSFT_s017
                                 + 1.17381 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.969615 w_k1_ADM_s100
                                 + 0.980219 w_k1_AXP_s100
                                 + 0.943587 w_k1_DIS_s100
                                 + 0.951382 w_k1_FDX_s100
                                 + 1.25797 w_k1_IBM_s100
                                 + 0.994502 w_k1_INTC_s100
                                 + 1.15024 w_k1_KO_s100 + 1.20727 w_k1_MAT_s100
                                 + 0.989544 w_k1_MSFT_s100
                                 + 0.924886 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.951484 w_k1_ADM_s100
                                 + 1.01625 w_k1_AXP_s100
                                 + 0.737915 w_k1_DIS_s100
                                 + 0.920409 w_k1_FDX_s100
                                 + 0.82873 w_k1_IBM_s100
                                 + 0.789342 w_k1_INTC_s100
                                 + 0.906086 w_k1_KO_s100
                                 + 1.11117 w_k1_MAT_s100
                                 + 0.854697 w_k1_MSFT_s100
                                 + 0.851607 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.680915 w_k1_ADM_s100
                                 + 0.730113 w_k1_AXP_s100
                                 + 0.724758 w_k1_DIS_s100
                                 + 0.75841 w_k1_FDX_s100
                                 + 1.04906 w_k1_IBM_s100
                                 + 0.737152 w_k1_INTC_s100
                                 + 0.873568 w_k1_KO_s100
                                 + 1.00111 w_k1_MAT_s100
                                 + 0.740282 w_k1_MSFT_s100
                                 + 0.824888 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.21531 w_k1_ADM_s100
                                 + 1.33998 w_k1_AXP_s100
                                 + 1.35218 w_k1_DIS_s100
                                 + 1.46462 w_k1_FDX_s100
                                 + 0.990992 w_k1_IBM_s100
                                 + 1.23462 w_k1_INTC_s100
                                 + 1.05674 w_k1_KO_s100
                                 + 0.918678 w_k1_MAT_s100
                                 + 1.36466 w_k1_MSFT_s100
                                 + 0.98367 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.739596 w_k1_ADM_s100
                                 + 0.667638 w_k1_AXP_s100
                                 + 0.558582 w_k1_DIS_s100
                                 + 0.741759 w_k1_FDX_s100
                                 + 0.679414 w_k1_IBM_s100
                                 + 0.919634 w_k1_INTC_s100
                                 + 0.706257 w_k1_KO_s100
                                 + 0.608059 w_k1_MAT_s100
                                 + 0.80938 w_k1_MSFT_s100
                                 + 0.797274 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.03825 w_k1_ADM_s100
                                 + 1.13338 w_k1_AXP_s100 + 1.2578 w_k1_DIS_s100
                                 + 1.05042 w_k1_FDX_s100
                                 + 1.39233 w_k1_IBM_s100
                                 + 1.01551 w_k1_INTC_s100 + 1.3327 w_k1_KO_s100
                                 + 1.29714 w_k1_MAT_s100
                                 + 1.15259 w_k1_MSFT_s100
                                 + 1.06821 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.920835 w_k1_ADM_s100
                                 + 0.756787 w_k1_AXP_s100
                                 + 0.921403 w_k1_DIS_s100
                                 + 0.985189 w_k1_FDX_s100
                                 + 0.726417 w_k1_IBM_s100
                                 + 1.12818 w_k1_INTC_s100
                                 + 0.967339 w_k1_KO_s100
                                 + 0.569107 w_k1_MAT_s100
                                 + 1.20325 w_k1_MSFT_s100
                                 + 0.876868 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.793093 w_k1_ADM_s100
                                 + 0.975513 w_k1_AXP_s100
                                 + 0.843123 w_k1_DIS_s100
                                 + 1.04281 w_k1_FDX_s100
                                 + 0.838377 w_k1_IBM_s100
                                 + 0.836934 w_k1_INTC_s100
                                 + 0.907826 w_k1_KO_s100
                                 + 0.920996 w_k1_MAT_s100
                                 + 0.991455 w_k1_MSFT_s100
                                 + 0.856252 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.98548 w_k1_ADM_s101
                                 + 1.14646 w_k1_AXP_s101
                                 + 1.21473 w_k1_DIS_s101
                                 + 1.06306 w_k1_FDX_s101 + 1.4023 w_k1_IBM_s101
                                 + 1.13767 w_k1_INTC_s101
                                 + 1.12378 w_k1_KO_s101 + 1.36419 w_k1_MAT_s101
                                 + 1.08446 w_k1_MSFT_s101 + 1.064 w_k1_PG_s101
                                 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.517963 w_k1_ADM_s101
                                 + 0.610366 w_k1_AXP_s101
                                 + 0.584764 w_k1_DIS_s101
                                 + 0.599123 w_k1_FDX_s101
                                 + 1.1857 w_k1_IBM_s101
                                 + 0.755932 w_k1_INTC_s101
                                 + 0.789456 w_k1_KO_s101
                                 + 0.963004 w_k1_MAT_s101
                                 + 0.635438 w_k1_MSFT_s101
                                 + 0.768903 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.07116 w_k1_ADM_s101
                                 + 1.5122 w_k1_AXP_s101 + 1.66211 w_k1_DIS_s101
                                 + 1.45476 w_k1_FDX_s101
                                 + 1.06508 w_k1_IBM_s101
                                 + 1.34519 w_k1_INTC_s101
                                 + 1.16725 w_k1_KO_s101 + 1.03855 w_k1_MAT_s101
                                 + 1.2334 w_k1_MSFT_s101 + 1.27176 w_k1_PG_s101
                                 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.983285 w_k1_ADM_s101
                                 + 0.867625 w_k1_AXP_s101
                                 + 0.924366 w_k1_DIS_s101
                                 + 0.9199 w_k1_FDX_s101 + 1.17272 w_k1_IBM_s101
                                 + 1.02162 w_k1_INTC_s101
                                 + 1.13722 w_k1_KO_s101 + 1.15496 w_k1_MAT_s101
                                 + 0.92163 w_k1_MSFT_s101
                                 + 0.939564 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.08774 w_k1_ADM_s101
                                 + 1.43041 w_k1_AXP_s101 + 1.272 w_k1_DIS_s101
                                 + 0.94237 w_k1_FDX_s101 + 1.9119 w_k1_IBM_s101
                                 + 1.06762 w_k1_INTC_s101
                                 + 1.46365 w_k1_KO_s101 + 2.35124 w_k1_MAT_s101
                                 + 0.853919 w_k1_MSFT_s101
                                 + 1.15987 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.703717 w_k1_ADM_s101
                                 + 0.740999 w_k1_AXP_s101
                                 + 0.753225 w_k1_DIS_s101
                                 + 0.823435 w_k1_FDX_s101
                                 + 1.05388 w_k1_IBM_s101
                                 + 0.906742 w_k1_INTC_s101
                                 + 1.0283 w_k1_KO_s101 + 1.17101 w_k1_MAT_s101
                                 + 0.858994 w_k1_MSFT_s101
                                 + 0.859571 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.868879 w_k1_ADM_s101
                                 + 0.691143 w_k1_AXP_s101
                                 + 0.768274 w_k1_DIS_s101
                                 + 0.804086 w_k1_FDX_s101
                                 + 0.759762 w_k1_IBM_s101
                                 + 0.779897 w_k1_INTC_s101
                                 + 0.872989 w_k1_KO_s101
                                 + 0.568596 w_k1_MAT_s101
                                 + 0.779177 w_k1_MSFT_s101
                                 + 0.824214 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.767491 w_k1_ADM_s101
                                 + 0.986466 w_k1_AXP_s101
                                 + 1.00386 w_k1_DIS_s101
                                 + 1.17323 w_k1_FDX_s101
                                 + 0.748939 w_k1_IBM_s101
                                 + 1.15391 w_k1_INTC_s101
                                 + 0.875122 w_k1_KO_s101
                                 + 0.692307 w_k1_MAT_s101
                                 + 0.954862 w_k1_MSFT_s101
                                 + 0.838554 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.06279 w_k1_ADM_s102
                                 + 0.828857 w_k1_AXP_s102
                                 + 0.743328 w_k1_DIS_s102
                                 + 0.925978 w_k1_FDX_s102
                                 + 0.625488 w_k1_IBM_s102
                                 + 1.02655 w_k1_INTC_s102
                                 + 0.800557 w_k1_KO_s102
                                 + 0.53631 w_k1_MAT_s102
                                 + 1.07423 w_k1_MSFT_s102
                                 + 0.964111 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.34533 w_k1_ADM_s102
                                 + 1.61304 w_k1_AXP_s102
                                 + 1.65082 w_k1_DIS_s102
                                 + 1.51299 w_k1_FDX_s102
                                 + 0.903375 w_k1_IBM_s102
                                 + 1.30267 w_k1_INTC_s102
                                 + 1.19159 w_k1_KO_s102 + 1.001 w_k1_MAT_s102
                                 + 1.49259 w_k1_MSFT_s102
                                 + 1.32955 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.780557 w_k1_ADM_s102
                                 + 0.766187 w_k1_AXP_s102
                                 + 0.782531 w_k1_DIS_s102
                                 + 0.833829 w_k1_FDX_s102
                                 + 0.917498 w_k1_IBM_s102
                                 + 0.899933 w_k1_INTC_s102
                                 + 0.830436 w_k1_KO_s102
                                 + 0.717802 w_k1_MAT_s102
                                 + 0.912885 w_k1_MSFT_s102
                                 + 0.872173 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.823185 w_k1_ADM_s102
                                 + 0.742966 w_k1_AXP_s102
                                 + 0.771471 w_k1_DIS_s102
                                 + 0.725175 w_k1_FDX_s102
                                 + 1.24559 w_k1_IBM_s102
                                 + 0.700499 w_k1_INTC_s102
                                 + 1.11567 w_k1_KO_s102 + 1.18674 w_k1_MAT_s102
                                 + 0.840974 w_k1_MSFT_s102
                                 + 1.0407 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.18062 w_k1_ADM_s102
                                 + 1.06539 w_k1_AXP_s102
                                 + 1.08013 w_k1_DIS_s102
                                 + 0.978903 w_k1_FDX_s102
                                 + 1.43994 w_k1_IBM_s102
                                 + 0.881503 w_k1_INTC_s102
                                 + 1.17055 w_k1_KO_s102 + 1.31074 w_k1_MAT_s102
                                 + 1.14352 w_k1_MSFT_s102
                                 + 1.12964 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.00474 w_k1_ADM_s102
                                 + 1.0646 w_k1_AXP_s102 + 1.35138 w_k1_DIS_s102
                                 + 1.12166 w_k1_FDX_s102
                                 + 1.00955 w_k1_IBM_s102
                                 + 1.24228 w_k1_INTC_s102
                                 + 1.31116 w_k1_KO_s102 + 1.15314 w_k1_MAT_s102
                                 + 1.40042 w_k1_MSFT_s102
                                 + 1.20772 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.82792 w_k1_ADM_s102
                                 + 0.687777 w_k1_AXP_s102
                                 + 0.697207 w_k1_DIS_s102
                                 + 0.740142 w_k1_FDX_s102
                                 + 0.957451 w_k1_IBM_s102
                                 + 0.927234 w_k1_INTC_s102
                                 + 0.978228 w_k1_KO_s102
                                 + 0.844938 w_k1_MAT_s102
                                 + 0.930446 w_k1_MSFT_s102
                                 + 0.933606 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.19421 w_k1_ADM_s102
                                 + 1.36547 w_k1_AXP_s102 + 1.1177 w_k1_DIS_s102
                                 + 1.10347 w_k1_FDX_s102
                                 + 0.957847 w_k1_IBM_s102
                                 + 0.791152 w_k1_INTC_s102
                                 + 1.08749 w_k1_KO_s102 + 1.11165 w_k1_MAT_s102
                                 + 1.041 w_k1_MSFT_s102 + 1.13829 w_k1_PG_s102
                                 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.27533 w_k1_ADM_s103
                                 + 0.923087 w_k1_AXP_s103
                                 + 0.820914 w_k1_DIS_s103
                                 + 0.866466 w_k1_FDX_s103
                                 + 1.0169 w_k1_IBM_s103
                                 + 0.732766 w_k1_INTC_s103
                                 + 0.901482 w_k1_KO_s103
                                 + 0.968622 w_k1_MAT_s103
                                 + 0.797375 w_k1_MSFT_s103
                                 + 0.928313 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.27229 w_k1_ADM_s103
                                 + 0.916123 w_k1_AXP_s103
                                 + 0.983493 w_k1_DIS_s103
                                 + 0.967774 w_k1_FDX_s103
                                 + 1.19226 w_k1_IBM_s103
                                 + 1.07652 w_k1_INTC_s103
                                 + 1.02899 w_k1_KO_s103 + 0.82111 w_k1_MAT_s103
                                 + 1.02336 w_k1_MSFT_s103
                                 + 0.990436 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.986017 w_k1_ADM_s103
                                 + 0.622304 w_k1_AXP_s103
                                 + 0.661385 w_k1_DIS_s103
                                 + 0.718245 w_k1_FDX_s103
                                 + 0.744327 w_k1_IBM_s103
                                 + 0.77634 w_k1_INTC_s103
                                 + 0.669751 w_k1_KO_s103
                                 + 0.438252 w_k1_MAT_s103
                                 + 0.824468 w_k1_MSFT_s103
                                 + 0.808395 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.09156 w_k1_ADM_s103
                                 + 0.909955 w_k1_AXP_s103
                                 + 0.98749 w_k1_DIS_s103
                                 + 0.881265 w_k1_FDX_s103
                                 + 1.60471 w_k1_IBM_s103
                                 + 0.820457 w_k1_INTC_s103
                                 + 1.26508 w_k1_KO_s103 + 1.42696 w_k1_MAT_s103
                                 + 0.997045 w_k1_MSFT_s103
                                 + 1.02162 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.44779 w_k1_ADM_s103
                                 + 1.1518 w_k1_AXP_s103 + 1.43466 w_k1_DIS_s103
                                 + 1.31292 w_k1_FDX_s103
                                 + 0.875641 w_k1_IBM_s103
                                 + 0.982322 w_k1_INTC_s103
                                 + 1.04549 w_k1_KO_s103
                                 + 0.847064 w_k1_MAT_s103
                                 + 1.28904 w_k1_MSFT_s103
                                 + 1.09308 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.18064 w_k1_ADM_s103
                                 + 1.27903 w_k1_AXP_s103
                                 + 1.55441 w_k1_DIS_s103
                                 + 1.28619 w_k1_FDX_s103
                                 + 1.33814 w_k1_IBM_s103
                                 + 0.961816 w_k1_INTC_s103
                                 + 1.17445 w_k1_KO_s103 + 1.46296 w_k1_MAT_s103
                                 + 1.26532 w_k1_MSFT_s103
                                 + 1.13537 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.89981 w_k1_ADM_s103
                                 + 0.851537 w_k1_AXP_s103
                                 + 0.903629 w_k1_DIS_s103
                                 + 0.926961 w_k1_FDX_s103
                                 + 0.910024 w_k1_IBM_s103
                                 + 0.884085 w_k1_INTC_s103
                                 + 0.788766 w_k1_KO_s103
                                 + 0.801059 w_k1_MAT_s103
                                 + 0.967368 w_k1_MSFT_s103
                                 + 0.911312 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.27592 w_k1_ADM_s103
                                 + 0.89303 w_k1_AXP_s103
                                 + 0.876767 w_k1_DIS_s103
                                 + 0.954963 w_k1_FDX_s103
                                 + 0.70553 w_k1_IBM_s103
                                 + 0.845857 w_k1_INTC_s103
                                 + 0.761516 w_k1_KO_s103
                                 + 0.688561 w_k1_MAT_s103
                                 + 0.917956 w_k1_MSFT_s103
                                 + 0.91537 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.997299 w_k1_ADM_s104
                                 + 1.229 w_k1_AXP_s104 + 1.32981 w_k1_DIS_s104
                                 + 1.1253 w_k1_FDX_s104 + 1.43581 w_k1_IBM_s104
                                 + 1.29746 w_k1_INTC_s104
                                 + 1.36269 w_k1_KO_s104 + 1.0603 w_k1_MAT_s104
                                 + 1.18414 w_k1_MSFT_s104
                                 + 1.25171 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.904232 w_k1_ADM_s104
                                 + 1.01715 w_k1_AXP_s104
                                 + 0.956458 w_k1_DIS_s104
                                 + 0.932561 w_k1_FDX_s104
                                 + 1.16806 w_k1_IBM_s104
                                 + 0.869554 w_k1_INTC_s104
                                 + 1.03173 w_k1_KO_s104
                                 + 0.990018 w_k1_MAT_s104
                                 + 0.894307 w_k1_MSFT_s104
                                 + 0.870116 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.712483 w_k1_ADM_s104
                                 + 0.987795 w_k1_AXP_s104
                                 + 0.865499 w_k1_DIS_s104
                                 + 0.828262 w_k1_FDX_s104
                                 + 1.59781 w_k1_IBM_s104
                                 + 0.848798 w_k1_INTC_s104
                                 + 1.09716 w_k1_KO_s104 + 1.47744 w_k1_MAT_s104
                                 + 0.999877 w_k1_MSFT_s104
                                 + 0.966695 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.54563 w_k1_ADM_s104
                                 + 1.62336 w_k1_AXP_s104
                                 + 1.66894 w_k1_DIS_s104 + 1.4181 w_k1_FDX_s104
                                 + 1.31857 w_k1_IBM_s104
                                 + 1.52884 w_k1_INTC_s104
                                 + 1.52536 w_k1_KO_s104 + 1.48472 w_k1_MAT_s104
                                 + 1.54095 w_k1_MSFT_s104
                                 + 1.30056 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.28347 w_k1_ADM_s104
                                 + 1.50762 w_k1_AXP_s104
                                 + 1.35646 w_k1_DIS_s104
                                 + 1.38311 w_k1_FDX_s104
                                 + 0.914403 w_k1_IBM_s104
                                 + 1.19816 w_k1_INTC_s104
                                 + 1.07874 w_k1_KO_s104
                                 + 0.947591 w_k1_MAT_s104
                                 + 1.2115 w_k1_MSFT_s104 + 1.09358 w_k1_PG_s104
                                 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.86755 w_k1_ADM_s104
                                 + 0.687608 w_k1_AXP_s104
                                 + 0.685693 w_k1_DIS_s104
                                 + 0.741582 w_k1_FDX_s104
                                 + 1.06092 w_k1_IBM_s104
                                 + 0.647609 w_k1_INTC_s104
                                 + 0.843965 w_k1_KO_s104
                                 + 0.785065 w_k1_MAT_s104
                                 + 0.774383 w_k1_MSFT_s104
                                 + 0.837497 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.930677 w_k1_ADM_s104
                                 + 0.908109 w_k1_AXP_s104
                                 + 0.706321 w_k1_DIS_s104
                                 + 0.855742 w_k1_FDX_s104
                                 + 0.76001 w_k1_IBM_s104
                                 + 0.728624 w_k1_INTC_s104
                                 + 0.671957 w_k1_KO_s104
                                 + 0.753364 w_k1_MAT_s104
                                 + 0.754433 w_k1_MSFT_s104
                                 + 0.801424 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.961906 w_k1_ADM_s104
                                 + 0.824281 w_k1_AXP_s104
                                 + 0.969237 w_k1_DIS_s104
                                 + 0.915772 w_k1_FDX_s104
                                 + 0.874264 w_k1_IBM_s104
                                 + 0.828806 w_k1_INTC_s104
                                 + 0.982016 w_k1_KO_s104
                                 + 0.811748 w_k1_MAT_s104
                                 + 1.10932 w_k1_MSFT_s104
                                 + 1.01515 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.03831 w_k1_ADM_s105
                                 + 0.81029 w_k1_AXP_s105
                                 + 0.766752 w_k1_DIS_s105
                                 + 0.777592 w_k1_FDX_s105
                                 + 1.07835 w_k1_IBM_s105
                                 + 0.872714 w_k1_INTC_s105
                                 + 0.833135 w_k1_KO_s105 + 1.0601 w_k1_MAT_s105
                                 + 0.767139 w_k1_MSFT_s105
                                 + 0.905598 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.14391 w_k1_ADM_s105
                                 + 1.17034 w_k1_AXP_s105
                                 + 1.67611 w_k1_DIS_s105
                                 + 1.19954 w_k1_FDX_s105
                                 + 0.783254 w_k1_IBM_s105
                                 + 1.26348 w_k1_INTC_s105
                                 + 1.15477 w_k1_KO_s105
                                 + 0.839328 w_k1_MAT_s105
                                 + 1.26822 w_k1_MSFT_s105
                                 + 0.910897 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.1666 w_k1_ADM_s105
                                 + 1.22714 w_k1_AXP_s105
                                 + 1.79237 w_k1_DIS_s105
                                 + 1.11739 w_k1_FDX_s105 + 1.6037 w_k1_IBM_s105
                                 + 1.07441 w_k1_INTC_s105 + 1.4806 w_k1_KO_s105
                                 + 1.91255 w_k1_MAT_s105
                                 + 1.25535 w_k1_MSFT_s105
                                 + 1.20294 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.16773 w_k1_ADM_s105
                                 + 1.02811 w_k1_AXP_s105
                                 + 0.927181 w_k1_DIS_s105
                                 + 1.08973 w_k1_FDX_s105
                                 + 0.740736 w_k1_IBM_s105
                                 + 0.975544 w_k1_INTC_s105
                                 + 0.907815 w_k1_KO_s105
                                 + 0.738515 w_k1_MAT_s105
                                 + 1.00862 w_k1_MSFT_s105
                                 + 0.993216 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.06212 w_k1_ADM_s105
                                 + 0.74052 w_k1_AXP_s105
                                 + 0.701953 w_k1_DIS_s105
                                 + 0.858949 w_k1_FDX_s105
                                 + 0.613045 w_k1_IBM_s105
                                 + 0.772841 w_k1_INTC_s105
                                 + 0.672962 w_k1_KO_s105
                                 + 0.585735 w_k1_MAT_s105
                                 + 0.852961 w_k1_MSFT_s105
                                 + 0.792442 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.16107 w_k1_ADM_s105
                                 + 1.34131 w_k1_AXP_s105
                                 + 1.18595 w_k1_DIS_s105
                                 + 1.12898 w_k1_FDX_s105
                                 + 1.00425 w_k1_IBM_s105
                                 + 1.02005 w_k1_INTC_s105
                                 + 1.02571 w_k1_KO_s105 + 1.12114 w_k1_MAT_s105
                                 + 0.947558 w_k1_MSFT_s105
                                 + 0.970557 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.0251 w_k1_ADM_s105
                                 + 0.904895 w_k1_AXP_s105
                                 + 0.838568 w_k1_DIS_s105
                                 + 0.839516 w_k1_FDX_s105
                                 + 1.59438 w_k1_IBM_s105
                                 + 1.03404 w_k1_INTC_s105
                                 + 1.23998 w_k1_KO_s105 + 1.60512 w_k1_MAT_s105
                                 + 0.953705 w_k1_MSFT_s105
                                 + 0.984718 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.27704 w_k1_ADM_s105
                                 + 1.19963 w_k1_AXP_s105
                                 + 1.43946 w_k1_DIS_s105
                                 + 1.15134 w_k1_FDX_s105
                                 + 1.12133 w_k1_IBM_s105
                                 + 1.00657 w_k1_INTC_s105
                                 + 1.10327 w_k1_KO_s105 + 1.21992 w_k1_MAT_s105
                                 + 1.1607 w_k1_MSFT_s105 + 1.01336 w_k1_PG_s105
                                 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.26396 w_k1_ADM_s106
                                 + 1.2786 w_k1_AXP_s106 + 1.03184 w_k1_DIS_s106
                                 + 1.25204 w_k1_FDX_s106
                                 + 0.69835 w_k1_IBM_s106
                                 + 1.15832 w_k1_INTC_s106
                                 + 0.957628 w_k1_KO_s106
                                 + 0.814449 w_k1_MAT_s106
                                 + 1.21139 w_k1_MSFT_s106
                                 + 1.11335 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.2358 w_k1_ADM_s106
                                 + 1.36623 w_k1_AXP_s106
                                 + 1.05909 w_k1_DIS_s106 + 1.1124 w_k1_FDX_s106
                                 + 1.25295 w_k1_IBM_s106
                                 + 1.0896 w_k1_INTC_s106 + 1.05792 w_k1_KO_s106
                                 + 1.25686 w_k1_MAT_s106
                                 + 1.19927 w_k1_MSFT_s106
                                 + 1.11734 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.33302 w_k1_ADM_s106
                                 + 1.23047 w_k1_AXP_s106 + 1.2262 w_k1_DIS_s106
                                 + 1.28611 w_k1_FDX_s106
                                 + 1.01731 w_k1_IBM_s106
                                 + 1.23707 w_k1_INTC_s106
                                 + 1.20175 w_k1_KO_s106 + 1.11006 w_k1_MAT_s106
                                 + 1.27607 w_k1_MSFT_s106
                                 + 1.34412 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.796344 w_k1_ADM_s106
                                 + 0.86486 w_k1_AXP_s106
                                 + 0.707434 w_k1_DIS_s106
                                 + 0.839895 w_k1_FDX_s106
                                 + 0.838073 w_k1_IBM_s106
                                 + 0.918304 w_k1_INTC_s106
                                 + 0.835952 w_k1_KO_s106
                                 + 0.975687 w_k1_MAT_s106
                                 + 0.941276 w_k1_MSFT_s106
                                 + 0.846707 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.27728 w_k1_ADM_s106
                                 + 0.917222 w_k1_AXP_s106
                                 + 0.9517 w_k1_DIS_s106 + 1.01495 w_k1_FDX_s106
                                 + 0.847403 w_k1_IBM_s106
                                 + 0.875886 w_k1_INTC_s106
                                 + 0.985105 w_k1_KO_s106
                                 + 0.902581 w_k1_MAT_s106
                                 + 0.982044 w_k1_MSFT_s106
                                 + 1.09712 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.22624 w_k1_ADM_s106
                                 + 1.70829 w_k1_AXP_s106
                                 + 1.61904 w_k1_DIS_s106
                                 + 1.46776 w_k1_FDX_s106
                                 + 1.34264 w_k1_IBM_s106
                                 + 1.5355 w_k1_INTC_s106 + 1.56302 w_k1_KO_s106
                                 + 1.57353 w_k1_MAT_s106
                                 + 1.49449 w_k1_MSFT_s106
                                 + 1.45951 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.866319 w_k1_ADM_s106
                                 + 0.609539 w_k1_AXP_s106
                                 + 0.576379 w_k1_DIS_s106
                                 + 0.773101 w_k1_FDX_s106
                                 + 0.599243 w_k1_IBM_s106
                                 + 0.806155 w_k1_INTC_s106
                                 + 0.602152 w_k1_KO_s106
                                 + 0.562962 w_k1_MAT_s106
                                 + 0.827814 w_k1_MSFT_s106
                                 + 0.726424 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.004 w_k1_ADM_s106
                                 + 0.955623 w_k1_AXP_s106
                                 + 0.834765 w_k1_DIS_s106
                                 + 0.888936 w_k1_FDX_s106
                                 + 1.32529 w_k1_IBM_s106
                                 + 0.948125 w_k1_INTC_s106
                                 + 1.1479 w_k1_KO_s106 + 1.34462 w_k1_MAT_s106
                                 + 1.03127 w_k1_MSFT_s106
                                 + 1.16664 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.09289 w_k1_ADM_s107
                                 + 1.7284 w_k1_AXP_s107 + 1.22251 w_k1_DIS_s107
                                 + 1.29022 w_k1_FDX_s107 + 1.0382 w_k1_IBM_s107
                                 + 1.09493 w_k1_INTC_s107 + 1.1411 w_k1_KO_s107
                                 + 1.52267 w_k1_MAT_s107
                                 + 1.03913 w_k1_MSFT_s107
                                 + 1.18324 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.769715 w_k1_ADM_s107
                                 + 0.796003 w_k1_AXP_s107
                                 + 0.687945 w_k1_DIS_s107
                                 + 0.866531 w_k1_FDX_s107
                                 + 0.939566 w_k1_IBM_s107
                                 + 0.784066 w_k1_INTC_s107
                                 + 0.790154 w_k1_KO_s107
                                 + 0.981117 w_k1_MAT_s107
                                 + 0.711796 w_k1_MSFT_s107
                                 + 0.83821 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.12619 w_k1_ADM_s107
                                 + 1.27021 w_k1_AXP_s107
                                 + 1.35049 w_k1_DIS_s107
                                 + 1.26768 w_k1_FDX_s107
                                 + 1.40024 w_k1_IBM_s107
                                 + 1.14039 w_k1_INTC_s107
                                 + 1.45232 w_k1_KO_s107 + 1.55819 w_k1_MAT_s107
                                 + 1.11212 w_k1_MSFT_s107
                                 + 1.21182 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.13746 w_k1_ADM_s107
                                 + 0.89597 w_k1_AXP_s107
                                 + 1.05287 w_k1_DIS_s107
                                 + 1.12295 w_k1_FDX_s107
                                 + 0.714385 w_k1_IBM_s107
                                 + 0.842948 w_k1_INTC_s107
                                 + 0.916003 w_k1_KO_s107
                                 + 0.876039 w_k1_MAT_s107
                                 + 0.962642 w_k1_MSFT_s107
                                 + 0.990761 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.774452 w_k1_ADM_s107
                                 + 0.746859 w_k1_AXP_s107
                                 + 0.770797 w_k1_DIS_s107
                                 + 0.906977 w_k1_FDX_s107
                                 + 0.701715 w_k1_IBM_s107
                                 + 0.792951 w_k1_INTC_s107
                                 + 0.712007 w_k1_KO_s107
                                 + 0.607277 w_k1_MAT_s107
                                 + 0.795944 w_k1_MSFT_s107
                                 + 0.830613 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.41697 w_k1_ADM_s107
                                 + 1.3615 w_k1_AXP_s107 + 1.65876 w_k1_DIS_s107
                                 + 1.46865 w_k1_FDX_s107
                                 + 0.903907 w_k1_IBM_s107
                                 + 1.31079 w_k1_INTC_s107
                                 + 1.19652 w_k1_KO_s107 + 1.02958 w_k1_MAT_s107
                                 + 1.1835 w_k1_MSFT_s107 + 1.07794 w_k1_PG_s107
                                 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.823804 w_k1_ADM_s107
                                 + 0.902729 w_k1_AXP_s107
                                 + 0.977367 w_k1_DIS_s107
                                 + 1.00193 w_k1_FDX_s107
                                 + 1.03236 w_k1_IBM_s107
                                 + 0.965809 w_k1_INTC_s107
                                 + 1.01746 w_k1_KO_s107 + 1.06567 w_k1_MAT_s107
                                 + 0.904613 w_k1_MSFT_s107
                                 + 0.994937 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.07183 w_k1_ADM_s107
                                 + 1.17419 w_k1_AXP_s107
                                 + 0.946272 w_k1_DIS_s107
                                 + 0.941886 w_k1_FDX_s107
                                 + 1.4244 w_k1_IBM_s107
                                 + 0.987974 w_k1_INTC_s107
                                 + 1.29393 w_k1_KO_s107 + 1.32256 w_k1_MAT_s107
                                 + 0.832572 w_k1_MSFT_s107
                                 + 0.987299 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.886044 w_k1_ADM_s110
                                 + 1.56572 w_k1_AXP_s110
                                 + 1.18963 w_k1_DIS_s110
                                 + 1.10236 w_k1_FDX_s110
                                 + 1.55417 w_k1_IBM_s110
                                 + 1.02361 w_k1_INTC_s110
                                 + 1.35246 w_k1_KO_s110 + 2.57111 w_k1_MAT_s110
                                 + 1.06319 w_k1_MSFT_s110
                                 + 1.03093 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.19242 w_k1_ADM_s110
                                 + 1.3043 w_k1_AXP_s110
                                 + 0.731997 w_k1_DIS_s110
                                 + 0.927892 w_k1_FDX_s110
                                 + 1.15394 w_k1_IBM_s110
                                 + 1.07366 w_k1_INTC_s110
                                 + 1.02042 w_k1_KO_s110 + 1.43895 w_k1_MAT_s110
                                 + 0.847515 w_k1_MSFT_s110
                                 + 1.02713 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.17435 w_k1_ADM_s110
                                 + 1.35289 w_k1_AXP_s110
                                 + 1.20061 w_k1_DIS_s110
                                 + 1.23978 w_k1_FDX_s110
                                 + 0.967867 w_k1_IBM_s110
                                 + 1.28869 w_k1_INTC_s110
                                 + 1.04727 w_k1_KO_s110
                                 + 0.832039 w_k1_MAT_s110
                                 + 1.07482 w_k1_MSFT_s110
                                 + 0.968239 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.910135 w_k1_ADM_s110
                                 + 0.86236 w_k1_AXP_s110
                                 + 0.739959 w_k1_DIS_s110
                                 + 0.911602 w_k1_FDX_s110
                                 + 0.679989 w_k1_IBM_s110
                                 + 1.15102 w_k1_INTC_s110
                                 + 0.781089 w_k1_KO_s110
                                 + 0.770399 w_k1_MAT_s110
                                 + 1.05106 w_k1_MSFT_s110
                                 + 0.878817 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.80295 w_k1_ADM_s110
                                 + 0.824282 w_k1_AXP_s110
                                 + 0.834793 w_k1_DIS_s110
                                 + 0.747682 w_k1_FDX_s110
                                 + 1.14103 w_k1_IBM_s110
                                 + 0.978898 w_k1_INTC_s110
                                 + 1.06667 w_k1_KO_s110 + 1.22348 w_k1_MAT_s110
                                 + 0.873413 w_k1_MSFT_s110
                                 + 0.95755 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.18568 w_k1_ADM_s110
                                 + 0.965862 w_k1_AXP_s110
                                 + 1.16164 w_k1_DIS_s110
                                 + 1.01829 w_k1_FDX_s110 + 1.3728 w_k1_IBM_s110
                                 + 1.10085 w_k1_INTC_s110
                                 + 1.39781 w_k1_KO_s110 + 1.70166 w_k1_MAT_s110
                                 + 1.13149 w_k1_MSFT_s110
                                 + 0.971579 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.2873 w_k1_ADM_s110
                                 + 1.67172 w_k1_AXP_s110
                                 + 1.77272 w_k1_DIS_s110
                                 + 1.40085 w_k1_FDX_s110
                                 + 1.42633 w_k1_IBM_s110
                                 + 1.2853 w_k1_INTC_s110 + 1.53072 w_k1_KO_s110
                                 + 1.57983 w_k1_MAT_s110
                                 + 1.30629 w_k1_MSFT_s110
                                 + 1.17336 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.976546 w_k1_ADM_s110
                                 + 1.12052 w_k1_AXP_s110
                                 + 0.843154 w_k1_DIS_s110
                                 + 1.06348 w_k1_FDX_s110
                                 + 0.928116 w_k1_IBM_s110
                                 + 1.1988 w_k1_INTC_s110
                                 + 0.982136 w_k1_KO_s110
                                 + 1.08398 w_k1_MAT_s110
                                 + 0.965382 w_k1_MSFT_s110
                                 + 0.912828 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.981571 w_k1_ADM_s111
                                 + 1.07311 w_k1_AXP_s111
                                 + 1.18472 w_k1_DIS_s111
                                 + 1.03691 w_k1_FDX_s111 + 1.5638 w_k1_IBM_s111
                                 + 1.03176 w_k1_INTC_s111
                                 + 1.23311 w_k1_KO_s111 + 1.67079 w_k1_MAT_s111
                                 + 1.07193 w_k1_MSFT_s111
                                 + 1.02501 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.91397 w_k1_ADM_s111
                                 + 0.827375 w_k1_AXP_s111
                                 + 1.02614 w_k1_DIS_s111
                                 + 1.00587 w_k1_FDX_s111
                                 + 0.941243 w_k1_IBM_s111
                                 + 1.00426 w_k1_INTC_s111
                                 + 0.925559 w_k1_KO_s111
                                 + 0.924055 w_k1_MAT_s111
                                 + 1.14532 w_k1_MSFT_s111
                                 + 0.919059 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.788699 w_k1_ADM_s111
                                 + 0.82793 w_k1_AXP_s111
                                 + 0.711329 w_k1_DIS_s111
                                 + 0.799348 w_k1_FDX_s111
                                 + 1.05694 w_k1_IBM_s111
                                 + 0.735269 w_k1_INTC_s111
                                 + 0.811781 w_k1_KO_s111
                                 + 1.32819 w_k1_MAT_s111
                                 + 0.760241 w_k1_MSFT_s111
                                 + 0.852453 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.28219 w_k1_ADM_s111
                                 + 1.18045 w_k1_AXP_s111
                                 + 1.30946 w_k1_DIS_s111
                                 + 1.15245 w_k1_FDX_s111
                                 + 1.18188 w_k1_IBM_s111
                                 + 1.02109 w_k1_INTC_s111
                                 + 1.12954 w_k1_KO_s111 + 1.54116 w_k1_MAT_s111
                                 + 1.15413 w_k1_MSFT_s111
                                 + 1.01792 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.25207 w_k1_ADM_s111
                                 + 1.04672 w_k1_AXP_s111
                                 + 1.17574 w_k1_DIS_s111
                                 + 1.33985 w_k1_FDX_s111
                                 + 0.851951 w_k1_IBM_s111
                                 + 1.20389 w_k1_INTC_s111
                                 + 1.03741 w_k1_KO_s111
                                 + 0.797366 w_k1_MAT_s111
                                 + 1.17117 w_k1_MSFT_s111
                                 + 0.999521 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.913028 w_k1_ADM_s111
                                 + 0.733433 w_k1_AXP_s111
                                 + 0.647675 w_k1_DIS_s111
                                 + 0.833096 w_k1_FDX_s111
                                 + 0.599713 w_k1_IBM_s111
                                 + 0.780867 w_k1_INTC_s111
                                 + 0.624418 w_k1_KO_s111
                                 + 0.475576 w_k1_MAT_s111
                                 + 0.769794 w_k1_MSFT_s111
                                 + 0.756595 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.22868 w_k1_ADM_s111
                                 + 1.04357 w_k1_AXP_s111 + 1.0801 w_k1_DIS_s111
                                 + 1.12427 w_k1_FDX_s111 + 0.6781 w_k1_IBM_s111
                                 + 0.86368 w_k1_INTC_s111
                                 + 0.852736 w_k1_KO_s111
                                 + 0.457866 w_k1_MAT_s111
                                 + 1.00997 w_k1_MSFT_s111
                                 + 0.912438 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.35596 w_k1_ADM_s111
                                 + 1.33416 w_k1_AXP_s111
                                 + 1.70198 w_k1_DIS_s111
                                 + 1.44496 w_k1_FDX_s111
                                 + 1.09131 w_k1_IBM_s111
                                 + 1.33415 w_k1_INTC_s111
                                 + 1.23186 w_k1_KO_s111
                                 + 0.959239 w_k1_MAT_s111
                                 + 1.40418 w_k1_MSFT_s111
                                 + 1.17465 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.25501 w_k1_ADM_s112
                                 + 1.32444 w_k1_AXP_s112
                                 + 1.45311 w_k1_DIS_s112
                                 + 1.23427 w_k1_FDX_s112
                                 + 1.15202 w_k1_IBM_s112
                                 + 1.23611 w_k1_INTC_s112
                                 + 1.49087 w_k1_KO_s112 + 1.04218 w_k1_MAT_s112
                                 + 1.10694 w_k1_MSFT_s112
                                 + 1.22382 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.825129 w_k1_ADM_s112
                                 + 0.684333 w_k1_AXP_s112
                                 + 0.649671 w_k1_DIS_s112
                                 + 0.805632 w_k1_FDX_s112
                                 + 0.76502 w_k1_IBM_s112
                                 + 1.0229 w_k1_INTC_s112
                                 + 0.956901 w_k1_KO_s112
                                 + 0.759876 w_k1_MAT_s112
                                 + 0.846305 w_k1_MSFT_s112
                                 + 0.952518 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.930155 w_k1_ADM_s112
                                 + 1.01803 w_k1_AXP_s112
                                 + 1.43853 w_k1_DIS_s112
                                 + 1.06478 w_k1_FDX_s112
                                 + 1.51018 w_k1_IBM_s112
                                 + 1.05312 w_k1_INTC_s112 + 1.8774 w_k1_KO_s112
                                 + 1.41188 w_k1_MAT_s112
                                 + 1.32831 w_k1_MSFT_s112
                                 + 1.11601 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.02564 w_k1_ADM_s112
                                 + 0.800952 w_k1_AXP_s112
                                 + 1.24083 w_k1_DIS_s112
                                 + 0.993934 w_k1_FDX_s112
                                 + 1.13217 w_k1_IBM_s112
                                 + 1.10273 w_k1_INTC_s112
                                 + 1.34678 w_k1_KO_s112 + 1.02696 w_k1_MAT_s112
                                 + 1.17096 w_k1_MSFT_s112 + 1.1452 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.79034 w_k1_ADM_s112
                                 + 0.637085 w_k1_AXP_s112
                                 + 0.813455 w_k1_DIS_s112
                                 + 0.831738 w_k1_FDX_s112
                                 + 0.686329 w_k1_IBM_s112
                                 + 0.971269 w_k1_INTC_s112
                                 + 0.883769 w_k1_KO_s112
                                 + 0.499201 w_k1_MAT_s112
                                 + 0.995437 w_k1_MSFT_s112
                                 + 0.947985 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.926395 w_k1_ADM_s112
                                 + 0.943688 w_k1_AXP_s112
                                 + 1.1593 w_k1_DIS_s112 + 1.11724 w_k1_FDX_s112
                                 + 1.11394 w_k1_IBM_s112
                                 + 1.08634 w_k1_INTC_s112 + 1.2551 w_k1_KO_s112
                                 + 1.09651 w_k1_MAT_s112
                                 + 1.13099 w_k1_MSFT_s112
                                 + 1.02851 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.753059 w_k1_ADM_s112
                                 + 0.894355 w_k1_AXP_s112
                                 + 0.749575 w_k1_DIS_s112
                                 + 0.835146 w_k1_FDX_s112
                                 + 0.969832 w_k1_IBM_s112
                                 + 0.892358 w_k1_INTC_s112
                                 + 0.96452 w_k1_KO_s112
                                 + 0.795361 w_k1_MAT_s112
                                 + 0.808514 w_k1_MSFT_s112
                                 + 0.864855 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.990312 w_k1_ADM_s112
                                 + 1.10135 w_k1_AXP_s112
                                 + 0.970808 w_k1_DIS_s112
                                 + 1.07925 w_k1_FDX_s112
                                 + 0.858239 w_k1_IBM_s112
                                 + 0.953042 w_k1_INTC_s112
                                 + 1.18149 w_k1_KO_s112
                                 + 0.713785 w_k1_MAT_s112
                                 + 1.0852 w_k1_MSFT_s112 + 1.09353 w_k1_PG_s112
                                 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.19864 w_k1_ADM_s113
                                 + 1.43321 w_k1_AXP_s113
                                 + 1.40573 w_k1_DIS_s113
                                 + 1.31187 w_k1_FDX_s113
                                 + 0.974544 w_k1_IBM_s113
                                 + 0.984252 w_k1_INTC_s113
                                 + 1.12982 w_k1_KO_s113 + 1.46466 w_k1_MAT_s113
                                 + 1.04872 w_k1_MSFT_s113
                                 + 1.10754 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.10647 w_k1_ADM_s113
                                 + 1.06262 w_k1_AXP_s113
                                 + 1.11779 w_k1_DIS_s113
                                 + 0.978036 w_k1_FDX_s113
                                 + 1.38826 w_k1_IBM_s113
                                 + 0.852599 w_k1_INTC_s113
                                 + 1.27016 w_k1_KO_s113 + 1.33641 w_k1_MAT_s113
                                 + 0.948518 w_k1_MSFT_s113
                                 + 1.15169 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.863192 w_k1_ADM_s113
                                 + 0.844204 w_k1_AXP_s113
                                 + 0.787041 w_k1_DIS_s113
                                 + 0.868608 w_k1_FDX_s113
                                 + 0.736613 w_k1_IBM_s113
                                 + 0.823267 w_k1_INTC_s113
                                 + 0.905007 w_k1_KO_s113
                                 + 0.798855 w_k1_MAT_s113
                                 + 0.762903 w_k1_MSFT_s113
                                 + 0.90487 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.12665 w_k1_ADM_s113
                                 + 1.04799 w_k1_AXP_s113
                                 + 1.45608 w_k1_DIS_s113
                                 + 1.18829 w_k1_FDX_s113
                                 + 0.763692 w_k1_IBM_s113
                                 + 1.06025 w_k1_INTC_s113
                                 + 1.17813 w_k1_KO_s113
                                 + 0.644544 w_k1_MAT_s113
                                 + 1.18619 w_k1_MSFT_s113
                                 + 1.16373 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.708511 w_k1_ADM_s113
                                 + 0.703421 w_k1_AXP_s113
                                 + 0.533889 w_k1_DIS_s113
                                 + 0.760831 w_k1_FDX_s113
                                 + 0.77448 w_k1_IBM_s113
                                 + 0.72626 w_k1_INTC_s113
                                 + 0.829071 w_k1_KO_s113
                                 + 0.685477 w_k1_MAT_s113
                                 + 0.686093 w_k1_MSFT_s113
                                 + 0.793435 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.16627 w_k1_ADM_s113
                                 + 1.23357 w_k1_AXP_s113
                                 + 1.68591 w_k1_DIS_s113
                                 + 1.19186 w_k1_FDX_s113
                                 + 1.51496 w_k1_IBM_s113
                                 + 1.12469 w_k1_INTC_s113
                                 + 1.68182 w_k1_KO_s113 + 1.79232 w_k1_MAT_s113
                                 + 1.3846 w_k1_MSFT_s113 + 1.31526 w_k1_PG_s113
                                 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.990089 w_k1_ADM_s113
                                 + 1.06281 w_k1_AXP_s113
                                 + 1.05471 w_k1_DIS_s113
                                 + 1.05692 w_k1_FDX_s113
                                 + 0.795555 w_k1_IBM_s113
                                 + 0.940353 w_k1_INTC_s113
                                 + 0.963071 w_k1_KO_s113
                                 + 0.95936 w_k1_MAT_s113
                                 + 0.88994 w_k1_MSFT_s113
                                 + 0.994957 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.99423 w_k1_ADM_s113
                                 + 0.972157 w_k1_AXP_s113
                                 + 0.896726 w_k1_DIS_s113
                                 + 0.878618 w_k1_FDX_s113
                                 + 1.03761 w_k1_IBM_s113
                                 + 0.823693 w_k1_INTC_s113
                                 + 1.17073 w_k1_KO_s113 + 1.29035 w_k1_MAT_s113
                                 + 0.827484 w_k1_MSFT_s113
                                 + 0.937874 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.936492 w_k1_ADM_s114
                                 + 0.856811 w_k1_AXP_s114
                                 + 1.29048 w_k1_DIS_s114
                                 + 0.942249 w_k1_FDX_s114
                                 + 1.30715 w_k1_IBM_s114
                                 + 1.00832 w_k1_INTC_s114
                                 + 1.27294 w_k1_KO_s114 + 1.30011 w_k1_MAT_s114
                                 + 1.07883 w_k1_MSFT_s114
                                 + 0.944592 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.888303 w_k1_ADM_s114
                                 + 0.812596 w_k1_AXP_s114
                                 + 0.808087 w_k1_DIS_s114
                                 + 0.893592 w_k1_FDX_s114
                                 + 0.725308 w_k1_IBM_s114
                                 + 0.820583 w_k1_INTC_s114
                                 + 0.824895 w_k1_KO_s114
                                 + 1.10657 w_k1_MAT_s114
                                 + 0.793575 w_k1_MSFT_s114
                                 + 0.822416 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.14408 w_k1_ADM_s114
                                 + 1.32459 w_k1_AXP_s114
                                 + 1.31784 w_k1_DIS_s114 + 1.1211 w_k1_FDX_s114
                                 + 1.62074 w_k1_IBM_s114
                                 + 1.02701 w_k1_INTC_s114
                                 + 1.42418 w_k1_KO_s114 + 2.18722 w_k1_MAT_s114
                                 + 1.14162 w_k1_MSFT_s114
                                 + 1.20221 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.10302 w_k1_ADM_s114
                                 + 1.23904 w_k1_AXP_s114 + 1.0178 w_k1_DIS_s114
                                 + 1.05507 w_k1_FDX_s114
                                 + 1.09775 w_k1_IBM_s114
                                 + 1.16728 w_k1_INTC_s114
                                 + 1.03859 w_k1_KO_s114 + 1.11236 w_k1_MAT_s114
                                 + 0.944204 w_k1_MSFT_s114
                                 + 0.983346 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.999504 w_k1_ADM_s114
                                 + 1.32262 w_k1_AXP_s114
                                 + 1.47737 w_k1_DIS_s114
                                 + 1.26559 w_k1_FDX_s114
                                 + 1.13061 w_k1_IBM_s114
                                 + 1.54895 w_k1_INTC_s114
                                 + 1.23204 w_k1_KO_s114 + 1.22556 w_k1_MAT_s114
                                 + 1.24926 w_k1_MSFT_s114
                                 + 1.10471 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.811501 w_k1_ADM_s114
                                 + 1.01661 w_k1_AXP_s114
                                 + 0.920515 w_k1_DIS_s114
                                 + 0.944313 w_k1_FDX_s114
                                 + 1.35907 w_k1_IBM_s114
                                 + 1.16526 w_k1_INTC_s114
                                 + 1.15746 w_k1_KO_s114 + 1.70563 w_k1_MAT_s114
                                 + 0.945901 w_k1_MSFT_s114
                                 + 0.989452 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.31491 w_k1_ADM_s114
                                 + 1.09611 w_k1_AXP_s114
                                 + 1.32474 w_k1_DIS_s114
                                 + 1.17171 w_k1_FDX_s114
                                 + 0.875823 w_k1_IBM_s114
                                 + 1.33148 w_k1_INTC_s114
                                 + 1.08401 w_k1_KO_s114
                                 + 0.883087 w_k1_MAT_s114
                                 + 1.18222 w_k1_MSFT_s114
                                 + 0.99451 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.03911 w_k1_ADM_s114
                                 + 0.968369 w_k1_AXP_s114
                                 + 0.807116 w_k1_DIS_s114
                                 + 0.838734 w_k1_FDX_s114
                                 + 1.05376 w_k1_IBM_s114
                                 + 0.8434 w_k1_INTC_s114
                                 + 0.920184 w_k1_KO_s114
                                 + 1.01252 w_k1_MAT_s114
                                 + 0.80178 w_k1_MSFT_s114
                                 + 0.832229 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.25212 w_k1_ADM_s115
                                 + 1.15781 w_k1_AXP_s115
                                 + 1.22122 w_k1_DIS_s115 + 1.2875 w_k1_FDX_s115
                                 + 0.866163 w_k1_IBM_s115
                                 + 1.01765 w_k1_INTC_s115
                                 + 0.953991 w_k1_KO_s115
                                 + 1.37663 w_k1_MAT_s115
                                 + 1.13119 w_k1_MSFT_s115
                                 + 1.10199 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.75919 w_k1_ADM_s115
                                 + 0.728875 w_k1_AXP_s115
                                 + 0.604549 w_k1_DIS_s115
                                 + 0.679008 w_k1_FDX_s115
                                 + 0.845376 w_k1_IBM_s115
                                 + 0.664928 w_k1_INTC_s115
                                 + 0.794478 w_k1_KO_s115
                                 + 0.941734 w_k1_MAT_s115
                                 + 0.728895 w_k1_MSFT_s115
                                 + 0.822358 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.02139 w_k1_ADM_s115
                                 + 1.03057 w_k1_AXP_s115
                                 + 1.02709 w_k1_DIS_s115
                                 + 0.99428 w_k1_FDX_s115 + 0.9797 w_k1_IBM_s115
                                 + 0.859129 w_k1_INTC_s115
                                 + 0.903195 w_k1_KO_s115
                                 + 1.21776 w_k1_MAT_s115
                                 + 0.857853 w_k1_MSFT_s115
                                 + 1.00023 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.01839 w_k1_ADM_s115
                                 + 0.665574 w_k1_AXP_s115
                                 + 0.661667 w_k1_DIS_s115
                                 + 0.730679 w_k1_FDX_s115
                                 + 0.669483 w_k1_IBM_s115
                                 + 0.559228 w_k1_INTC_s115
                                 + 0.783906 w_k1_KO_s115
                                 + 0.810703 w_k1_MAT_s115
                                 + 0.725014 w_k1_MSFT_s115
                                 + 0.854871 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.18403 w_k1_ADM_s115
                                 + 1.45016 w_k1_AXP_s115
                                 + 1.52692 w_k1_DIS_s115
                                 + 1.36431 w_k1_FDX_s115
                                 + 1.16013 w_k1_IBM_s115
                                 + 1.01799 w_k1_INTC_s115
                                 + 1.16235 w_k1_KO_s115 + 1.43427 w_k1_MAT_s115
                                 + 1.10125 w_k1_MSFT_s115
                                 + 1.22042 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.0461 w_k1_ADM_s115
                                 + 1.00369 w_k1_AXP_s115
                                 + 0.688737 w_k1_DIS_s115
                                 + 0.842319 w_k1_FDX_s115
                                 + 1.17649 w_k1_IBM_s115
                                 + 0.724623 w_k1_INTC_s115
                                 + 0.869094 w_k1_KO_s115
                                 + 1.15311 w_k1_MAT_s115
                                 + 0.939341 w_k1_MSFT_s115 + 1.008 w_k1_PG_s115
                                 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.19881 w_k1_ADM_s115
                                 + 1.33387 w_k1_AXP_s115
                                 + 1.21148 w_k1_DIS_s115
                                 + 1.08839 w_k1_FDX_s115
                                 + 1.64164 w_k1_IBM_s115
                                 + 1.12201 w_k1_INTC_s115
                                 + 1.34167 w_k1_KO_s115 + 1.74952 w_k1_MAT_s115
                                 + 1.16432 w_k1_MSFT_s115
                                 + 1.15908 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.12998 w_k1_ADM_s115
                                 + 0.961923 w_k1_AXP_s115
                                 + 0.859706 w_k1_DIS_s115
                                 + 0.994292 w_k1_FDX_s115
                                 + 0.745491 w_k1_IBM_s115
                                 + 0.796298 w_k1_INTC_s115
                                 + 0.863205 w_k1_KO_s115
                                 + 0.767946 w_k1_MAT_s115
                                 + 0.95753 w_k1_MSFT_s115
                                 + 0.996298 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.20146 w_k1_ADM_s116
                                 + 0.969368 w_k1_AXP_s116
                                 + 1.10445 w_k1_DIS_s116
                                 + 1.08198 w_k1_FDX_s116
                                 + 0.880423 w_k1_IBM_s116
                                 + 0.988059 w_k1_INTC_s116
                                 + 0.970734 w_k1_KO_s116
                                 + 0.552041 w_k1_MAT_s116
                                 + 1.12849 w_k1_MSFT_s116
                                 + 1.00548 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.846992 w_k1_ADM_s116
                                 + 0.753436 w_k1_AXP_s116
                                 + 0.679296 w_k1_DIS_s116
                                 + 0.780431 w_k1_FDX_s116
                                 + 0.665057 w_k1_IBM_s116
                                 + 0.802996 w_k1_INTC_s116
                                 + 0.641615 w_k1_KO_s116
                                 + 0.520574 w_k1_MAT_s116
                                 + 0.859615 w_k1_MSFT_s116
                                 + 0.813226 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.906339 w_k1_ADM_s116
                                 + 0.988002 w_k1_AXP_s116
                                 + 1.05007 w_k1_DIS_s116
                                 + 0.990557 w_k1_FDX_s116
                                 + 1.01126 w_k1_IBM_s116
                                 + 0.966205 w_k1_INTC_s116
                                 + 1.01513 w_k1_KO_s116 + 1.10704 w_k1_MAT_s116
                                 + 1.06603 w_k1_MSFT_s116
                                 + 0.947073 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.06769 w_k1_ADM_s116
                                 + 0.876776 w_k1_AXP_s116
                                 + 1.1926 w_k1_DIS_s116 + 1.02107 w_k1_FDX_s116
                                 + 1.24022 w_k1_IBM_s116
                                 + 0.880185 w_k1_INTC_s116
                                 + 1.15289 w_k1_KO_s116 + 1.08963 w_k1_MAT_s116
                                 + 1.09579 w_k1_MSFT_s116 + 1.0268 w_k1_PG_s116
                                 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.21493 w_k1_ADM_s116
                                 + 1.46863 w_k1_AXP_s116 + 1.3699 w_k1_DIS_s116
                                 + 1.21066 w_k1_FDX_s116
                                 + 1.46281 w_k1_IBM_s116
                                 + 1.06866 w_k1_INTC_s116 + 1.0659 w_k1_KO_s116
                                 + 1.4371 w_k1_MAT_s116
                                 + 1.10767 w_k1_MSFT_s116
                                 + 1.06657 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.29135 w_k1_ADM_s116
                                 + 1.30768 w_k1_AXP_s116
                                 + 1.32169 w_k1_DIS_s116
                                 + 1.28467 w_k1_FDX_s116
                                 + 0.895942 w_k1_IBM_s116
                                 + 1.17303 w_k1_INTC_s116
                                 + 0.954128 w_k1_KO_s116
                                 + 0.57357 w_k1_MAT_s116
                                 + 1.13435 w_k1_MSFT_s116
                                 + 1.03789 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.86413 w_k1_ADM_s116
                                 + 0.866469 w_k1_AXP_s116
                                 + 0.95402 w_k1_DIS_s116
                                 + 0.871949 w_k1_FDX_s116
                                 + 0.895573 w_k1_IBM_s116
                                 + 0.88873 w_k1_INTC_s116
                                 + 0.836956 w_k1_KO_s116
                                 + 0.599212 w_k1_MAT_s116
                                 + 1.0082 w_k1_MSFT_s116
                                 + 0.914669 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.837727 w_k1_ADM_s116
                                 + 0.880987 w_k1_AXP_s116
                                 + 0.794009 w_k1_DIS_s116
                                 + 0.762297 w_k1_FDX_s116
                                 + 1.11825 w_k1_IBM_s116
                                 + 0.757362 w_k1_INTC_s116
                                 + 0.88614 w_k1_KO_s116 + 1.07395 w_k1_MAT_s116
                                 + 0.881373 w_k1_MSFT_s116
                                 + 0.937896 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.34966 w_k1_ADM_s117
                                 + 1.54808 w_k1_AXP_s117
                                 + 1.35261 w_k1_DIS_s117
                                 + 1.35049 w_k1_FDX_s117
                                 + 1.13997 w_k1_IBM_s117
                                 + 1.16554 w_k1_INTC_s117
                                 + 1.07086 w_k1_KO_s117 + 1.32358 w_k1_MAT_s117
                                 + 1.29711 w_k1_MSFT_s117
                                 + 1.10611 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.893598 w_k1_ADM_s117
                                 + 0.841136 w_k1_AXP_s117
                                 + 0.900938 w_k1_DIS_s117
                                 + 0.983948 w_k1_FDX_s117
                                 + 0.93475 w_k1_IBM_s117
                                 + 0.8946 w_k1_INTC_s117
                                 + 0.889954 w_k1_KO_s117
                                 + 0.804852 w_k1_MAT_s117
                                 + 1.03004 w_k1_MSFT_s117
                                 + 0.997931 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.740064 w_k1_ADM_s117
                                 + 0.902717 w_k1_AXP_s117
                                 + 0.586592 w_k1_DIS_s117
                                 + 0.843904 w_k1_FDX_s117
                                 + 0.765302 w_k1_IBM_s117
                                 + 0.946246 w_k1_INTC_s117
                                 + 0.810346 w_k1_KO_s117
                                 + 1.05648 w_k1_MAT_s117
                                 + 0.958925 w_k1_MSFT_s117
                                 + 0.870866 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.859476 w_k1_ADM_s117
                                 + 0.908059 w_k1_AXP_s117
                                 + 0.8437 w_k1_DIS_s117
                                 + 0.937485 w_k1_FDX_s117
                                 + 0.839562 w_k1_IBM_s117
                                 + 0.901269 w_k1_INTC_s117
                                 + 0.725711 w_k1_KO_s117
                                 + 0.693134 w_k1_MAT_s117
                                 + 1.00754 w_k1_MSFT_s117
                                 + 0.946423 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.677981 w_k1_ADM_s117
                                 + 0.693454 w_k1_AXP_s117
                                 + 0.726967 w_k1_DIS_s117
                                 + 0.784099 w_k1_FDX_s117
                                 + 0.769236 w_k1_IBM_s117
                                 + 0.841944 w_k1_INTC_s117
                                 + 0.805763 w_k1_KO_s117
                                 + 0.754497 w_k1_MAT_s117
                                 + 0.985691 w_k1_MSFT_s117
                                 + 0.848015 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.06614 w_k1_ADM_s117
                                 + 1.04092 w_k1_AXP_s117
                                 + 0.832408 w_k1_DIS_s117 + 0.889 w_k1_FDX_s117
                                 + 1.30564 w_k1_IBM_s117
                                 + 1.08879 w_k1_INTC_s117
                                 + 1.03873 w_k1_KO_s117 + 1.23439 w_k1_MAT_s117
                                 + 1.09731 w_k1_MSFT_s117
                                 + 1.08157 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.831072 w_k1_ADM_s117
                                 + 1.05643 w_k1_AXP_s117
                                 + 1.10801 w_k1_DIS_s117 + 1.0638 w_k1_FDX_s117
                                 + 1.19727 w_k1_IBM_s117
                                 + 1.21533 w_k1_INTC_s117
                                 + 1.21308 w_k1_KO_s117 + 1.11517 w_k1_MAT_s117
                                 + 1.16885 w_k1_MSFT_s117
                                 + 1.08538 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.861201 w_k1_ADM_s117
                                 + 0.998212 w_k1_AXP_s117
                                 + 1.09968 w_k1_DIS_s117
                                 + 1.15162 w_k1_FDX_s117
                                 + 0.714207 w_k1_IBM_s117
                                 + 1.01132 w_k1_INTC_s117
                                 + 0.878267 w_k1_KO_s117
                                 + 0.738541 w_k1_MAT_s117
                                 + 1.25584 w_k1_MSFT_s117
                                 + 1.05417 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0207369521428078 w_k1_ADM_s000
                                 + 0.00670002835209204 w_k1_ADM_s001
                                 + 0.0232495338668269 w_k1_ADM_s002
                                 + 0.0354545159001569 w_k1_ADM_s003
                                 + 0.0111760179775787 w_k1_ADM_s004
                                 + 0.0250835392096243 w_k1_ADM_s005
                                 + 0.0143419949934115 w_k1_ADM_s006
                                 + 0.010722255947192 w_k1_ADM_s007
                                 + 0.0450502368795068 w_k1_ADM_s010
                                 + 0.0322521202749703 w_k1_ADM_s011
                                 + 0.0103505281824766 w_k1_ADM_s012
                                 + 0.0461717816010334 w_k1_ADM_s013
                                 + 0.0196028522291418 w_k1_ADM_s014
                                 + 0.0165070319457688 w_k1_ADM_s015
                                 + 0.0308934288617448 w_k1_ADM_s016
                                 + 0.0127472802771738 w_k1_ADM_s017
                                 + 0.0240558414730792 w_k1_ADM_s100
                                 + 0.0214074652392293 w_k1_ADM_s101
                                 + 0.0483054472660207 w_k1_ADM_s102
                                 + 0.0770739172564685 w_k1_ADM_s103
                                 + 0.0237293665627837 w_k1_ADM_s104
                                 + 0.0301800726450013 w_k1_ADM_s105
                                 + 0.050490471525532 w_k1_ADM_s106
                                 + 0.0459422336663263 w_k1_ADM_s107
                                 + 0.031439437602379 w_k1_ADM_s110
                                 + 0.0774700877367575 w_k1_ADM_s111
                                 + 0.0194416012002667 w_k1_ADM_s112
                                 + 0.0529225473754864 w_k1_ADM_s113
                                 + 0.0223689348525095 w_k1_ADM_s114
                                 + 0.0425254892378253 w_k1_ADM_s115
                                 + 0.0601928019000704 w_k1_ADM_s116
                                 + 0.0320135099009709 w_k1_ADM_s117
                                 + 0.0224860802640187 w_k1_AXP_s000
                                 + 0.00717480622360455 w_k1_AXP_s001
                                 + 0.0254985959317017 w_k1_AXP_s002
                                 + 0.0345639964678104 w_k1_AXP_s003
                                 + 0.0102794279762884 w_k1_AXP_s004
                                 + 0.0240042954565221 w_k1_AXP_s005
                                 + 0.0125519878003804 w_k1_AXP_s006
                                 + 0.0123293425044679 w_k1_AXP_s007
                                 + 0.0478387944923034 w_k1_AXP_s010
                                 + 0.0359113645218274 w_k1_AXP_s011
                                 + 0.0109792291830253 w_k1_AXP_s012
                                 + 0.0489495079306701 w_k1_AXP_s013
                                 + 0.021018125110961 w_k1_AXP_s014
                                 + 0.0190775893072815 w_k1_AXP_s015
                                 + 0.0289075050015122 w_k1_AXP_s016
                                 + 0.0111655431214189 w_k1_AXP_s017
                                 + 0.0253008020613058 w_k1_AXP_s100
                                 + 0.0239714007031991 w_k1_AXP_s101
                                 + 0.0485812572645994 w_k1_AXP_s102
                                 + 0.0623058570855199 w_k1_AXP_s103
                                 + 0.0244786315053891 w_k1_AXP_s104
                                 + 0.0276780566117597 w_k1_AXP_s105
                                 + 0.0505013796724771 w_k1_AXP_s106
                                 + 0.0488009976427932 w_k1_AXP_s107
                                 + 0.0344782186174698 w_k1_AXP_s110
                                 + 0.0743738427934377 w_k1_AXP_s111
                                 + 0.0196976845189126 w_k1_AXP_s112
                                 + 0.0544336946656956 w_k1_AXP_s113
                                 + 0.023343601392809 w_k1_AXP_s114
                                 + 0.0414276352708951 w_k1_AXP_s115
                                 + 0.0598088630561202 w_k1_AXP_s116
                                 + 0.0354554711509335 w_k1_AXP_s117
                                 + 0.0208392883668598 w_k1_DIS_s000
                                 + 0.00782727439883185 w_k1_DIS_s001
                                 + 0.0291731269371982 w_k1_DIS_s002
                                 + 0.0357236168212366 w_k1_DIS_s003
                                 + 0.010944548265993 w_k1_DIS_s004
                                 + 0.0258756738294524 w_k1_DIS_s005
                                 + 0.0125437248995369 w_k1_DIS_s006
                                 + 0.0114288686351958 w_k1_DIS_s007
                                 + 0.0519282635541495 w_k1_DIS_s010
                                 + 0.0362433762990277 w_k1_DIS_s011
                                 + 0.0110217540560296 w_k1_DIS_s012
                                 + 0.0515656616600069 w_k1_DIS_s013
                                 + 0.0199896410849162 w_k1_DIS_s014
                                 + 0.0196363922980778 w_k1_DIS_s015
                                 + 0.0300115080480095 w_k1_DIS_s016
                                 + 0.0104520593340549 w_k1_DIS_s017
                                 + 0.0250004179233135 w_k1_DIS_s100
                                 + 0.0253107371650471 w_k1_DIS_s101
                                 + 0.0494534676554776 w_k1_DIS_s102
                                 + 0.0680346711195856 w_k1_DIS_s103
                                 + 0.0234593657059875 w_k1_DIS_s104
                                 + 0.0293593868044022 w_k1_DIS_s105
                                 + 0.044442774000794 w_k1_DIS_s106
                                 + 0.047840136601434 w_k1_DIS_s107
                                 + 0.0314590890056966 w_k1_DIS_s110
                                 + 0.0801154767934174 w_k1_DIS_s111
                                 + 0.0213217443326067 w_k1_DIS_s112
                                 + 0.0558417472063553 w_k1_DIS_s113
                                 + 0.0235180310483563 w_k1_DIS_s114
                                 + 0.0381048449392359 w_k1_DIS_s115
                                 + 0.0622980418600578 w_k1_DIS_s116
                                 + 0.0325681554543304 w_k1_DIS_s117
                                 + 0.021875696364783 w_k1_FDX_s000
                                 + 0.00769166667423253 w_k1_FDX_s001
                                 + 0.0266315064063947 w_k1_FDX_s002
                                 + 0.0346920480704378 w_k1_FDX_s003
                                 + 0.0104671145755611 w_k1_FDX_s004
                                 + 0.0245390089469503 w_k1_FDX_s005
                                 + 0.0134925667408818 w_k1_FDX_s006
                                 + 0.0113960323367627 w_k1_FDX_s007
                                 + 0.0512288010848513 w_k1_FDX_s010
                                 + 0.0345582851574366 w_k1_FDX_s011
                                 + 0.0105218401672666 w_k1_FDX_s012
                                 + 0.050773942262177 w_k1_FDX_s013
                                 + 0.0199750959811797 w_k1_FDX_s014
                                 + 0.0191444870742204 w_k1_FDX_s015
                                 + 0.0301812789236369 w_k1_FDX_s016
                                 + 0.0118691289448556 w_k1_FDX_s017
                                 + 0.0265365531263792 w_k1_FDX_s100
                                 + 0.0242948760351359 w_k1_FDX_s101
                                 + 0.0472773083699641 w_k1_FDX_s102
                                 + 0.0655135371167387 w_k1_FDX_s103
                                 + 0.0233570615037206 w_k1_FDX_s104
                                 + 0.0267231726212986 w_k1_FDX_s105
                                 + 0.0478869773480285 w_k1_FDX_s106
                                 + 0.0490541805233846 w_k1_FDX_s107
                                 + 0.0313966397426315 w_k1_FDX_s110
                                 + 0.0783483098955118 w_k1_FDX_s111
                                 + 0.0202202281528581 w_k1_FDX_s112
                                 + 0.0523340626647832 w_k1_FDX_s113
                                 + 0.0224317298664188 w_k1_FDX_s114
                                 + 0.039236391655455 w_k1_FDX_s115
                                 + 0.0589077361385592 w_k1_FDX_s116
                                 + 0.0343143183208482 w_k1_FDX_s117
                                 + 0.0211421822741314 w_k1_IBM_s000
                                 + 0.00747765378036153 w_k1_IBM_s001
                                 + 0.0278810586195711 w_k1_IBM_s002
                                 + 0.0345624779570182 w_k1_IBM_s003
                                 + 0.0110107278488965 w_k1_IBM_s004
                                 + 0.0265653999645172 w_k1_IBM_s005
                                 + 0.0126782260175672 w_k1_IBM_s006
                                 + 0.0117305991466971 w_k1_IBM_s007
                                 + 0.0488397681871583 w_k1_IBM_s010
                                 + 0.0392059277799195 w_k1_IBM_s011
                                 + 0.0104417740194928 w_k1_IBM_s012
                                 + 0.0490241938283392 w_k1_IBM_s013
                                 + 0.0200862245670844 w_k1_IBM_s014
                                 + 0.0190915524603169 w_k1_IBM_s015
                                 + 0.0298442582619572 w_k1_IBM_s016
                                 + 0.0110968821425219 w_k1_IBM_s017
                                 + 0.0256289989435248 w_k1_IBM_s100
                                 + 0.025608435073231 w_k1_IBM_s101
                                 + 0.0464934832742213 w_k1_IBM_s102
                                 + 0.0660323879406361 w_k1_IBM_s103
                                 + 0.0235207400707052 w_k1_IBM_s104
                                 + 0.028165477724936 w_k1_IBM_s105
                                 + 0.0477357508110048 w_k1_IBM_s106
                                 + 0.047914252333526 w_k1_IBM_s107
                                 + 0.0328846417169957 w_k1_IBM_s110
                                 + 0.080168607867687 w_k1_IBM_s111
                                 + 0.0201594713227647 w_k1_IBM_s112
                                 + 0.052964872301425 w_k1_IBM_s113
                                 + 0.0237469738649681 w_k1_IBM_s114
                                 + 0.040109391267453 w_k1_IBM_s115
                                 + 0.0605928166002622 w_k1_IBM_s116
                                 + 0.0332273026201258 w_k1_IBM_s117
                                 + 0.02421169619307 w_k1_INTC_s000
                                 + 0.00764872801396671 w_k1_INTC_s001
                                 + 0.0283816279140705 w_k1_INTC_s002
                                 + 0.0314580797015163 w_k1_INTC_s003
                                 + 0.0110327575147675 w_k1_INTC_s004
                                 + 0.0244072940745826 w_k1_INTC_s005
                                 + 0.0124927414878764 w_k1_INTC_s006
                                 + 0.0100852261650285 w_k1_INTC_s007
                                 + 0.0498805279913847 w_k1_INTC_s010
                                 + 0.0440861369450855 w_k1_INTC_s011
                                 + 0.0107598827636603 w_k1_INTC_s012
                                 + 0.0559756179602676 w_k1_INTC_s013
                                 + 0.0195582809627027 w_k1_INTC_s014
                                 + 0.0237010742705261 w_k1_INTC_s015
                                 + 0.0304421784622251 w_k1_INTC_s016
                                 + 0.0122859008898983 w_k1_INTC_s017
                                 + 0.0251889142401096 w_k1_INTC_s100
                                 + 0.0247340351376221 w_k1_INTC_s101
                                 + 0.0463372175701265 w_k1_INTC_s102
                                 + 0.0577892178968202 w_k1_INTC_s103
                                 + 0.0217087975819871 w_k1_INTC_s104
                                 + 0.0260441354129174 w_k1_INTC_s105
                                 + 0.0480020521806099 w_k1_INTC_s106
                                 + 0.0444477656864922 w_k1_INTC_s107
                                 + 0.0349014130158932 w_k1_INTC_s110
                                 + 0.0726207586242433 w_k1_INTC_s111
                                 + 0.0209543381224116 w_k1_INTC_s112
                                 + 0.0464552600620977 w_k1_INTC_s113
                                 + 0.0239456002722422 w_k1_INTC_s114
                                 + 0.0330571137868506 w_k1_INTC_s115
                                 + 0.0551128630806799 w_k1_INTC_s116
                                 + 0.0342863294775086 w_k1_INTC_s117
                                 + 0.0197581648550407 w_k1_KO_s000
                                 + 0.00753784716016459 w_k1_KO_s001
                                 + 0.0278857151293056 w_k1_KO_s002
                                 + 0.0366256196886245 w_k1_KO_s003
                                 + 0.0113068454449682 w_k1_KO_s004
                                 + 0.028458850927558 w_k1_KO_s005
                                 + 0.0112073417024518 w_k1_KO_s006
                                 + 0.0109206830133567 w_k1_KO_s007
                                 + 0.0475039766447543 w_k1_KO_s010
                                 + 0.0344376088743028 w_k1_KO_s011
                                 + 0.00942450478221711 w_k1_KO_s012
                                 + 0.0473107367220675 w_k1_KO_s013
                                 + 0.0211511687124895 w_k1_KO_s014
                                 + 0.01945802900637 w_k1_KO_s015
                                 + 0.0274462533153829 w_k1_KO_s016
                                 + 0.0104615554873293 w_k1_KO_s017
                                 + 0.0257615832236753 w_k1_KO_s100
                                 + 0.0248100536866558 w_k1_KO_s101
                                 + 0.0498997001764779 w_k1_KO_s102
                                 + 0.0611989221002262 w_k1_KO_s103
                                 + 0.0229659416520971 w_k1_KO_s104
                                 + 0.0270573004176434 w_k1_KO_s105
                                 + 0.048005419362834 w_k1_KO_s106
                                 + 0.0488308479210387 w_k1_KO_s107
                                 + 0.0335115911135054 w_k1_KO_s110
                                 + 0.073965864725628 w_k1_KO_s111
                                 + 0.0246770844271911 w_k1_KO_s112
                                 + 0.0588123463210074 w_k1_KO_s113
                                 + 0.0235283378685937 w_k1_KO_s114
                                 + 0.0375448424521932 w_k1_KO_s115
                                 + 0.0553521891946305 w_k1_KO_s116
                                 + 0.0318111225156828 w_k1_KO_s117
                                 + 0.0204941761884127 w_k1_MAT_s000
                                 + 0.00790086169337825 w_k1_MAT_s001
                                 + 0.0251112882727891 w_k1_MAT_s002
                                 + 0.0376258501160666 w_k1_MAT_s003
                                 + 0.0131322428120745 w_k1_MAT_s004
                                 + 0.0264829748269746 w_k1_MAT_s005
                                 + 0.0115520023208642 w_k1_MAT_s006
                                 + 0.0135439937736558 w_k1_MAT_s007
                                 + 0.0468260356894482 w_k1_MAT_s010
                                 + 0.031740014733414 w_k1_MAT_s011
                                 + 0.0121104675535575 w_k1_MAT_s012
                                 + 0.0491124227413711 w_k1_MAT_s013
                                 + 0.0200544770204539 w_k1_MAT_s014
                                 + 0.0165285765185737 w_k1_MAT_s015
                                 + 0.0271146337607574 w_k1_MAT_s016
                                 + 0.0113460497747711 w_k1_MAT_s017
                                 + 0.0246293911526196 w_k1_MAT_s100
                                 + 0.024487829500979 w_k1_MAT_s101
                                 + 0.0459904744620013 w_k1_MAT_s102
                                 + 0.059619708326309 w_k1_MAT_s103
                                 + 0.0218729743602937 w_k1_MAT_s104
                                 + 0.0297260522176369 w_k1_MAT_s105
                                 + 0.0512066600138108 w_k1_MAT_s106
                                 + 0.05137597545484 w_k1_MAT_s107
                                 + 0.0382160927683142 w_k1_MAT_s110
                                 + 0.0836848057488325 w_k1_MAT_s111
                                 + 0.0180364547948776 w_k1_MAT_s112
                                 + 0.0609381697887303 w_k1_MAT_s113
                                 + 0.027980693342584 w_k1_MAT_s114
                                 + 0.0456332045503697 w_k1_MAT_s115
                                 + 0.0534224875797199 w_k1_MAT_s116
                                 + 0.0341275828734273 w_k1_MAT_s117
                                 + 0.0214314969008668 w_k1_MSFT_s000
                                 + 0.0076737107028206 w_k1_MSFT_s001
                                 + 0.0276910438605194 w_k1_MSFT_s002
                                 + 0.0355807133781954 w_k1_MSFT_s003
                                 + 0.0117156195689756 w_k1_MSFT_s004
                                 + 0.0255798374667662 w_k1_MSFT_s005
                                 + 0.0128508575151274 w_k1_MSFT_s006
                                 + 0.0109761056800306 w_k1_MSFT_s007
                                 + 0.0512937750534704 w_k1_MSFT_s010
                                 + 0.0363349052115936 w_k1_MSFT_s011
                                 + 0.0102295945798608 w_k1_MSFT_s012
                                 + 0.0508665023331735 w_k1_MSFT_s013
                                 + 0.0203439247253654 w_k1_MSFT_s014
                                 + 0.0194975154199925 w_k1_MSFT_s015
                                 + 0.0308227932873131 w_k1_MSFT_s016
                                 + 0.0119211690131221 w_k1_MSFT_s017
                                 + 0.0268348344640753 w_k1_MSFT_s100
                                 + 0.0226305035314373 w_k1_MSFT_s101
                                 + 0.0527187458790765 w_k1_MSFT_s102
                                 + 0.0660383502535615 w_k1_MSFT_s103
                                 + 0.0236604473103647 w_k1_MSFT_s104
                                 + 0.0264209245273222 w_k1_MSFT_s105
                                 + 0.0506502893211436 w_k1_MSFT_s106
                                 + 0.0419866894605219 w_k1_MSFT_s107
                                 + 0.0319520766741212 w_k1_MSFT_s110
                                 + 0.0771882964313499 w_k1_MSFT_s111
                                 + 0.0205122506595002 w_k1_MSFT_s112
                                 + 0.0482942820701033 w_k1_MSFT_s113
                                 + 0.0216824732205134 w_k1_MSFT_s114
                                 + 0.0374108242141677 w_k1_MSFT_s115
                                 + 0.0604267837086811 w_k1_MSFT_s116
                                 + 0.0371398719560902 w_k1_MSFT_s117
                                 + 0.0195761807634887 w_k1_PG_s000
                                 + 0.00746205816056601 w_k1_PG_s001
                                 + 0.0268049773706253 w_k1_PG_s002
                                 + 0.0348561788509341 w_k1_PG_s003
                                 + 0.0108283014338803 w_k1_PG_s004
                                 + 0.0241261794954575 w_k1_PG_s005
                                 + 0.0126714685246555 w_k1_PG_s006
                                 + 0.0116679834588202 w_k1_PG_s007
                                 + 0.0474905333528372 w_k1_PG_s010
                                 + 0.0337386979068438 w_k1_PG_s011
                                 + 0.0107191465271306 w_k1_PG_s012
                                 + 0.048142064804865 w_k1_PG_s013
                                 + 0.0209267637737484 w_k1_PG_s014
                                 + 0.020195292861985 w_k1_PG_s015
                                 + 0.0277239818344885 w_k1_PG_s016
                                 + 0.0107213972007548 w_k1_PG_s017
                                 + 0.0231192125483554 w_k1_PG_s100
                                 + 0.0234460136323375 w_k1_PG_s101
                                 + 0.0504385694572111 w_k1_PG_s102
                                 + 0.0633458749817482 w_k1_PG_s103
                                 + 0.0227221622013329 w_k1_PG_s104
                                 + 0.025908615298452 w_k1_PG_s105
                                 + 0.0503748672549173 w_k1_PG_s106
                                 + 0.0456545740114561 w_k1_PG_s107
                                 + 0.0296750466901737 w_k1_PG_s110
                                 + 0.0715942373066975 w_k1_PG_s111
                                 + 0.0210941939470357 w_k1_PG_s112
                                 + 0.0535437968594888 w_k1_PG_s113
                                 + 0.0212393532918582 w_k1_PG_s114
                                 + 0.0404218998636729 w_k1_PG_s115
                                 + 0.0565673401132797 w_k1_PG_s116
                                 + 0.0335805772692744 w_k1_PG_s117 >= 1.18
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
