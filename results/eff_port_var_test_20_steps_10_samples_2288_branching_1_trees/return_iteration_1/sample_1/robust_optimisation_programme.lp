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
 w_k1_ADM_s0_return_constraint:  1.17285 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.878314 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 0.908928 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.28034 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.894613 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.17213 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.995335 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.947089 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.997201 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.02307 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.87643 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.3842 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.00849 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.723688 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.00553 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.828031 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.978036 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.70433 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.39284 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.906278 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.867937 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.928745 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.870712 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.990631 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.05184 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.847225 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.05781 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.02375 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.30955 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.23715 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.971034 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.17106 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.777837 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.840789 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.932954 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.968581 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.894464 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.3612 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.2244 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.891429 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.961789 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.34913 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.787086 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.20107 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.866821 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.918493 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.2048 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.27265 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.705384 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.35224 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.04987 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.629025 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.951326 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.894702 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.05459 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.661264 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.63831 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.847536 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.802139 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.30911 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.696628 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.06827 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.924787 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.812459 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.901773 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.13148 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.47898 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.33089 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.70188 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.21238 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.702489 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.596147 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.785468 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.857252 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.906507 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.30898 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.3092 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.850623 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.914107 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.49023 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.737983 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.21077 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.759124 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.882375 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.3881 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.16528 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.872347 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.50588 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.985258 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.587649 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.07177 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.785648 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.20115 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.609371 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.68354 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.76801 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.96927 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.39149 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.646432 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.10329 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.936701 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.776917 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.18506 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.18392 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.41426 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.296 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.807966 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.29146 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.655253 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.702504 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.703598 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.03536 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.795709 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.47073 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.18503 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.904035 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 0.914927 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 1.28577 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.845698 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.12338 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.962959 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.818576 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.06728 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.11369 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.902466 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.42136 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.0865 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.717431 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.1234 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.869485 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.07682 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.735748 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.45412 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.965966 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.893503 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.10385 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.773155 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.949201 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.04673 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.737912 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.15095 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.961399 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.22264 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.25122 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.929971 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.2296 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.761866 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.713964 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.921167 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.00996 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.848272 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.15654 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.15682 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.94409 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.02144 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.08952 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.838101 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.13511 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.68822 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.50202 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.84654 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.15439 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.801376 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.17779 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.92299 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.682663 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.867466 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.27234 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.22861 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.736177 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.32811 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.586769 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.03345 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.82671 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.737596 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.13753 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.648611 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.22864 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.892711 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.79694 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.38974 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.920681 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.673585 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.944053 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.685006 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.99517 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.481155 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.841804 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.22507 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.63108 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.16043 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.944458 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.954481 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 1.12456 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.871587 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.06287 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.892311 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.944016 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.04921 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.964046 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.987954 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.13699 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.990576 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.734994 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.15879 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.81724 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.21441 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.700337 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.20399 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.9913 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.866014 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.33627 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.815786 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.938179 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.892734 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.832311 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.16267 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.97017 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.01554 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.13331 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.85426 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.01078 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.816506 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.775118 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.859358 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.890426 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.826623 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.05359 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.2317 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.937995 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.880419 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.17616 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.836039 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.17302 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.794115 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.13883 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.47527 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.20051 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.971939 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.35337 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.06327 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.659915 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.968768 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.896031 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.2309 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.693683 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.40204 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.691316 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.04567 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.6009 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.703954 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.0413 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.808113 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.959795 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.01288 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.27126 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.16588 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.998508 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.748616 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.999407 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.595105 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.813373 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.547375 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.925526 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.922766 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.37657 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.22697 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.897428 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.17883 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.11847 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.75382 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.18078 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.519083 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.17627 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.55038 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.0686 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.516266 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.28569 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.913789 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.47228 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.977364 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.22623 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.31867 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.629331 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.90897 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.529365 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.04711 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.9621 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.745119 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.32388 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.702832 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.35082 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.05706 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.63901 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.66298 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.1394 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.547506 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.165 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.523496 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.772785 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.630793 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.787656 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.24975 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.71867 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.1643 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.94051 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.918175 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.21255 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.847718 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.03244 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.88804 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.928102 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.13513 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.09507 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.990673 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.36 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.987674 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.739342 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.02971 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.827341 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.14633 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.686416 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.4863 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.930914 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.963245 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.25724 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.737107 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.936422 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.992127 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.752452 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.16983 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.983272 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.079 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.20624 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.87696 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.1819 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.777822 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.843502 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.95068 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.09405 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.808219 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.09625 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.12547 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.962109 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.975841 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.09918 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.815308 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.08838 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.897335 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.888287 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.14388 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.08703 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.898078 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.29194 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.05983 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.741151 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.97882 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.03656 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.20188 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.810258 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.25165 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.945704 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.02445 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.28455 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.878355 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.09567 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.900369 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.978017 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.08149 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.17642 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.32061 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.09035 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.852116 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.06326 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.805741 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.811057 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.782551 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.00277 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.944476 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.14344 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.304001289258732 z_k1_s0000
                                 - 0.767003578505941 z_k1_s0001
                                 - 0.218000592717231 z_k1_s0002
                                 - 0.711000735633052 z_k1_s0003
                                 - 0.305001139880737 z_k1_s0004
                                 - 0.567000763986494 z_k1_s0005
                                 - 0.917004521891297 z_k1_s0006
                                 - 0.881003361475428 z_k1_s0007
                                 - 0.496001435767198 z_k1_s0010
                                 - 0.219999781476152 z_k1_s0011
                                 - 0.261001152644199 z_k1_s0012
                                 - 0.195000839400807 z_k1_s0013
                                 - 0.117000124839938 z_k1_s0014
                                 - 0.245000617554229 z_k1_s0015
                                 - 0.227000015578013 z_k1_s0016
                                 - 0.133000281129361 z_k1_s0017
                                 - 0.938004700918313 z_k1_s0020
                                 - 0.726001517089638 z_k1_s0021
                                 - 0.224001042771192 z_k1_s0022
                                 - 0.225000953500121 z_k1_s0023
                                 - 0.331001475609827 z_k1_s0024
                                 - 0.379003252824666 z_k1_s0025
                                 - 0.359001472230647 z_k1_s0026
                                 - 0.384002806469311 z_k1_s0027
                                 - 0.389999635910579 z_k1_s0030
                                 - 0.17700005145959 z_k1_s0031
                                 - 0.496999680486161 z_k1_s0032
                                 - 0.181000119958731 z_k1_s0033
                                 - 0.198000248554948 z_k1_s0034
                                 - 0.186000043057526 z_k1_s0035
                                 - 0.243001073345324 z_k1_s0036
                                 - 0.295000446932939 z_k1_s0037
                                 - 0.222000511762151 z_k1_s0040
                                 - 1.04400218949726 z_k1_s0041
                                 - 0.226000263598803 z_k1_s0042
                                 - 0.58999910797608 z_k1_s0043
                                 - 0.454003081521179 z_k1_s0044
                                 - 0.684002204154885 z_k1_s0045
                                 - 0.963002750796318 z_k1_s0046
                                 - 0.281000422559759 z_k1_s0047
                                 - 0.318001222770093 z_k1_s0050
                                 - 0.404001639454138 z_k1_s0051
                                 - 0.782004544148338 z_k1_s0052
                                 - 0.50800170611234 z_k1_s0053
                                 - 0.315001138673887 z_k1_s0054
                                 - 0.695001678056844 z_k1_s0055
                                 - 0.595000299188457 z_k1_s0056
                                 - 0.656001866710787 z_k1_s0057
                                 - 1.78800285082944 z_k1_s0060
                                 - 0.935001963448953 z_k1_s0061
                                 - 0.503000935972903 z_k1_s0062
                                 - 0.560001016653693 z_k1_s0063
                                 - 0.534000467851204 z_k1_s0064
                                 - 0.377000661625757 z_k1_s0065
                                 - 0.74100104885027 z_k1_s0066
                                 - 0.642000268719571 z_k1_s0067
                                 - 0.473001252146276 z_k1_s0070
                                 - 0.907999136655882 z_k1_s0071
                                 - 0.82699949818173 z_k1_s0072
                                 - 0.50400140372235 z_k1_s0073
                                 - 0.340999318840487 z_k1_s0074
                                 - 0.743998940833123 z_k1_s0075
                                 - 0.315999575391448 z_k1_s0076
                                 - 0.584000582759274 z_k1_s0077
                                 - 0.217999563863389 z_k1_s0100
                                 - 0.172999627077874 z_k1_s0101
                                 - 0.663000060584875 z_k1_s0102
                                 - 0.251999539719071 z_k1_s0103
                                 - 0.353000437018093 z_k1_s0104
                                 - 0.265999779847448 z_k1_s0105
                                 - 0.376998419413807 z_k1_s0106
                                 - 0.35499849755414 z_k1_s0107
                                 - 0.211999049048949 z_k1_s0110
                                 - 0.146999548179538 z_k1_s0111
                                 - 0.109999707592713 z_k1_s0112
                                 - 0.16499944558942 z_k1_s0113
                                 - 0.212999558014603 z_k1_s0114
                                 - 0.0919996943831813 z_k1_s0115
                                 - 0.121999561999535 z_k1_s0116
                                 - 0.0969996916191818 z_k1_s0117
                                 - 0.0979999373771872 z_k1_s0120
                                 - 0.0579998787539852 z_k1_s0121
                                 - 0.0719998580221743 z_k1_s0122
                                 - 0.0609999037756911 z_k1_s0123
                                 - 0.20099936645813 z_k1_s0124
                                 - 0.133999577638753 z_k1_s0125
                                 - 0.043999899485796 z_k1_s0126
                                 - 0.156999714471924 z_k1_s0127
                                 - 0.11800002357304 z_k1_s0130
                                 - 0.104999879357702 z_k1_s0131
                                 - 0.0439998605072475 z_k1_s0132
                                 - 0.11800002357304 z_k1_s0133
                                 - 0.183999470284726 z_k1_s0134
                                 - 0.102999479094699 z_k1_s0135
                                 - 0.212999867611072 z_k1_s0136
                                 - 0.0979997563341653 z_k1_s0137
                                 - 0.166999688442282 z_k1_s0140
                                 - 0.0719998723936661 z_k1_s0141
                                 - 0.0529999264439115 z_k1_s0142
                                 - 0.101999509983258 z_k1_s0143
                                 - 0.173999468676766 z_k1_s0144
                                 - 0.101000157805778 z_k1_s0145
                                 - 0.0409999333283267 z_k1_s0146
                                 - 0.153000127973313 z_k1_s0147
                                 - 0.229998935657251 z_k1_s0150
                                 - 0.150999854713292 z_k1_s0151
                                 - 0.168000577130255 z_k1_s0152
                                 - 0.276999366312342 z_k1_s0153
                                 - 0.176000365327287 z_k1_s0154
                                 - 0.283999180984745 z_k1_s0155
                                 - 0.28899904860789 z_k1_s0156
                                 - 0.099999864959191 z_k1_s0157
                                 - 0.0829995983119403 z_k1_s0160
                                 - 0.0969995459066833 z_k1_s0161
                                 - 0.10599969421709 z_k1_s0162
                                 - 0.0719999124867505 z_k1_s0163
                                 - 0.152999336285655 z_k1_s0164
                                 - 0.0419999064840543 z_k1_s0165
                                 - 0.03599989254355 z_k1_s0166
                                 - 0.0479998567247333 z_k1_s0167
                                 - 0.174000393420156 z_k1_s0170
                                 - 0.264000596913339 z_k1_s0171
                                 - 0.1290000170734 z_k1_s0172
                                 - 0.071000069000124 z_k1_s0173
                                 - 0.189999971929783 z_k1_s0174
                                 - 0.121000090518505 z_k1_s0175
                                 - 0.256000121158118 z_k1_s0176
                                 - 0.168000379853943 z_k1_s0177
                                 - 0.591998575777055 z_k1_s1000
                                 - 0.509998417814771 z_k1_s1001
                                 - 0.310999983588799 z_k1_s1002
                                 - 1.04900140524936 z_k1_s1003
                                 - 0.402999904351418 z_k1_s1004
                                 - 0.355999967690075 z_k1_s1005
                                 - 0.479000204633501 z_k1_s1006
                                 - 0.505998512694637 z_k1_s1007
                                 - 0.396001014666814 z_k1_s1010
                                 - 0.198000062333447 z_k1_s1011
                                 - 0.237999328764512 z_k1_s1012
                                 - 0.269000250998456 z_k1_s1013
                                 - 0.328000570734153 z_k1_s1014
                                 - 0.358999267968294 z_k1_s1015
                                 - 0.201000029565775 z_k1_s1016
                                 - 0.235999053942987 z_k1_s1017
                                 - 0.689999387070385 z_k1_s1020
                                 - 0.52299884867087 z_k1_s1021
                                 - 0.428997880591233 z_k1_s1022
                                 - 0.260999285284255 z_k1_s1023
                                 - 0.703996616918728 z_k1_s1024
                                 - 0.191999716776366 z_k1_s1025
                                 - 0.249999557950645 z_k1_s1026
                                 - 0.652997982062668 z_k1_s1027
                                 - 0.256000545900988 z_k1_s1030
                                 - 0.540000337853558 z_k1_s1031
                                 - 0.166999916000987 z_k1_s1032
                                 - 0.23999962415708 z_k1_s1033
                                 - 0.458000347916543 z_k1_s1034
                                 - 0.126999978641474 z_k1_s1035
                                 - 0.355999205799642 z_k1_s1036
                                 - 0.223000065004332 z_k1_s1037
                                 - 0.368997543549408 z_k1_s1040
                                 - 0.367998798438827 z_k1_s1041
                                 - 0.184999084325393 z_k1_s1042
                                 - 0.224998943503576 z_k1_s1043
                                 - 0.313998412801098 z_k1_s1044
                                 - 0.317998093220412 z_k1_s1045
                                 - 0.215998957563571 z_k1_s1046
                                 - 0.354998662081809 z_k1_s1047
                                 - 0.116999845662003 z_k1_s1050
                                 - 0.183000784922009 z_k1_s1051
                                 - 0.105999929518526 z_k1_s1052
                                 - 0.122999865485679 z_k1_s1053
                                 - 0.147999887984365 z_k1_s1054
                                 - 0.451999209584949 z_k1_s1055
                                 - 0.492000543742072 z_k1_s1056
                                 - 0.182000120518457 z_k1_s1057
                                 - 0.310999238937559 z_k1_s1060
                                 - 0.472001389218431 z_k1_s1061
                                 - 0.204000178655208 z_k1_s1062
                                 - 0.841999054307401 z_k1_s1063
                                 - 0.276000011451046 z_k1_s1064
                                 - 0.241000224764248 z_k1_s1065
                                 - 0.276000011451046 z_k1_s1066
                                 - 0.174000201723585 z_k1_s1067
                                 - 0.213000235985484 z_k1_s1070
                                 - 0.415999873228999 z_k1_s1071
                                 - 0.238000415238463 z_k1_s1072
                                 - 0.546001068864882 z_k1_s1073
                                 - 0.181000296414097 z_k1_s1074
                                 - 0.728001425153176 z_k1_s1075
                                 - 0.364000712576588 z_k1_s1076
                                 - 0.608000828259261 z_k1_s1077
                                 - 0.18900006670819 z_k1_s1100
                                 - 0.112000124516019 z_k1_s1101
                                 - 0.178000164366509 z_k1_s1102
                                 - 0.201000206699274 z_k1_s1103
                                 - 0.346000872640877 z_k1_s1104
                                 - 0.478000952341856 z_k1_s1105
                                 - 0.168000082473961 z_k1_s1106
                                 - 0.414000344789497 z_k1_s1107
                                 - 0.626000685451053 z_k1_s1110
                                 - 0.485000298111927 z_k1_s1111
                                 - 0.450000680485831 z_k1_s1112
                                 - 1.03100216848946 z_k1_s1113
                                 - 0.443000321660032 z_k1_s1114
                                 - 0.521000381570833 z_k1_s1115
                                 - 1.92100322413883 z_k1_s1116
                                 - 1.77799972863119 z_k1_s1117
                                 - 0.332000593538686 z_k1_s1120
                                 - 0.1459998731828 z_k1_s1121
                                 - 0.18099996195946 z_k1_s1122
                                 - 0.231000585925958 z_k1_s1123
                                 - 0.235999604322943 z_k1_s1124
                                 - 0.232999845284864 z_k1_s1125
                                 - 0.255000398230034 z_k1_s1126
                                 - 0.125999971596089 z_k1_s1127
                                 - 0.461998998396777 z_k1_s1130
                                 - 0.403996909043589 z_k1_s1131
                                 - 0.225999118861796 z_k1_s1132
                                 - 0.563998918652678 z_k1_s1133
                                 - 0.281999096427701 z_k1_s1134
                                 - 0.332998693657013 z_k1_s1135
                                 - 0.879996536904459 z_k1_s1136
                                 - 0.477999199357686 z_k1_s1137
                                 - 0.184000037178317 z_k1_s1140
                                 - 0.403999086093251 z_k1_s1141
                                 - 0.3179995085438 z_k1_s1142
                                 - 0.601001189399766 z_k1_s1143
                                 - 0.200000082550364 z_k1_s1144
                                 - 0.265001205787224 z_k1_s1145
                                 - 0.24100016852872 z_k1_s1146
                                 - 0.210000050332866 z_k1_s1147
                                 - 0.92599843979457 z_k1_s1150
                                 - 1.0349992045425 z_k1_s1151
                                 - 1.51100114584382 z_k1_s1152
                                 - 0.597000919140532 z_k1_s1153
                                 - 0.814000117726034 z_k1_s1154
                                 - 0.921999277364412 z_k1_s1155
                                 - 0.535000837058616 z_k1_s1156
                                 - 0.918000114934254 z_k1_s1157
                                 - 0.552998055500369 z_k1_s1160
                                 - 0.250999285046407 z_k1_s1161
                                 - 0.255999470654406 z_k1_s1162
                                 - 0.541998449760782 z_k1_s1163
                                 - 0.386999116696905 z_k1_s1164
                                 - 0.622996641022299 z_k1_s1165
                                 - 1.00499717730723 z_k1_s1166
                                 - 0.395999049492298 z_k1_s1167
                                 - 0.890001406282889 z_k1_s1170
                                 - 0.586000617125318 z_k1_s1171
                                 - 0.475000202407805 z_k1_s1172
                                 - 1.57800210737539 z_k1_s1173
                                 - 1.12999877461497 z_k1_s1174
                                 - 1.26300083762291 z_k1_s1175
                                 - 0.459000377852333 z_k1_s1176
                                 - 0.490000295591285 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.23378 w_k1_ADM_s000
                                 + 1.58821 w_k1_AXP_s000
                                 + 1.45587 w_k1_DIS_s000
                                 + 1.19432 w_k1_FDX_s000
                                 + 1.69101 w_k1_IBM_s000
                                 + 1.30008 w_k1_INTC_s000
                                 + 1.45766 w_k1_KO_s000 + 1.81734 w_k1_MAT_s000
                                 + 1.18771 w_k1_MSFT_s000
                                 + 1.13764 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.89816 w_k1_ADM_s000
                                 + 1.14489 w_k1_AXP_s000
                                 + 1.14442 w_k1_DIS_s000
                                 + 0.986048 w_k1_FDX_s000
                                 + 1.23927 w_k1_IBM_s000
                                 + 1.16286 w_k1_INTC_s000
                                 + 1.04303 w_k1_KO_s000 + 1.48734 w_k1_MAT_s000
                                 + 1.03246 w_k1_MSFT_s000
                                 + 0.96439 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.709832 w_k1_ADM_s000
                                 + 0.879124 w_k1_AXP_s000
                                 + 0.695451 w_k1_DIS_s000
                                 + 0.702409 w_k1_FDX_s000
                                 + 1.39589 w_k1_IBM_s000
                                 + 0.950152 w_k1_INTC_s000
                                 + 1.04689 w_k1_KO_s000 + 1.06615 w_k1_MAT_s000
                                 + 0.794189 w_k1_MSFT_s000
                                 + 0.948915 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.723241 w_k1_ADM_s000
                                 + 0.812701 w_k1_AXP_s000
                                 + 0.732386 w_k1_DIS_s000
                                 + 0.816853 w_k1_FDX_s000
                                 + 0.679582 w_k1_IBM_s000
                                 + 0.911204 w_k1_INTC_s000
                                 + 0.760821 w_k1_KO_s000
                                 + 0.924083 w_k1_MAT_s000
                                 + 0.876631 w_k1_MSFT_s000
                                 + 0.809217 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.03132 w_k1_ADM_s000
                                 + 1.00934 w_k1_AXP_s000
                                 + 1.04334 w_k1_DIS_s000
                                 + 0.849865 w_k1_FDX_s000
                                 + 1.75685 w_k1_IBM_s000
                                 + 0.849023 w_k1_INTC_s000
                                 + 1.38128 w_k1_KO_s000 + 1.99812 w_k1_MAT_s000
                                 + 1.03229 w_k1_MSFT_s000
                                 + 1.20468 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.39865 w_k1_ADM_s000
                                 + 1.36573 w_k1_AXP_s000
                                 + 1.45611 w_k1_DIS_s000
                                 + 1.45325 w_k1_FDX_s000
                                 + 0.998021 w_k1_IBM_s000
                                 + 1.1468 w_k1_INTC_s000 + 1.14368 w_k1_KO_s000
                                 + 1.47422 w_k1_MAT_s000
                                 + 1.52865 w_k1_MSFT_s000
                                 + 1.23051 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.00856 w_k1_ADM_s000
                                 + 1.01857 w_k1_AXP_s000
                                 + 0.844334 w_k1_DIS_s000
                                 + 0.91571 w_k1_FDX_s000
                                 + 1.01236 w_k1_IBM_s000
                                 + 1.06207 w_k1_INTC_s000
                                 + 0.96043 w_k1_KO_s000 + 1.02561 w_k1_MAT_s000
                                 + 0.97619 w_k1_MSFT_s000
                                 + 0.857165 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.986889 w_k1_ADM_s000
                                 + 1.02269 w_k1_AXP_s000
                                 + 1.06317 w_k1_DIS_s000
                                 + 1.12844 w_k1_FDX_s000
                                 + 0.676474 w_k1_IBM_s000
                                 + 1.19236 w_k1_INTC_s000
                                 + 0.920957 w_k1_KO_s000
                                 + 0.791842 w_k1_MAT_s000
                                 + 1.02441 w_k1_MSFT_s000
                                 + 0.934207 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.733061 w_k1_ADM_s001
                                 + 0.785987 w_k1_AXP_s001
                                 + 0.677719 w_k1_DIS_s001
                                 + 0.839604 w_k1_FDX_s001
                                 + 0.821305 w_k1_IBM_s001
                                 + 0.772408 w_k1_INTC_s001
                                 + 0.75926 w_k1_KO_s001
                                 + 0.596088 w_k1_MAT_s001
                                 + 0.706189 w_k1_MSFT_s001
                                 + 0.933404 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.973209 w_k1_ADM_s001
                                 + 1.18411 w_k1_AXP_s001
                                 + 0.969893 w_k1_DIS_s001
                                 + 1.11427 w_k1_FDX_s001
                                 + 0.917544 w_k1_IBM_s001
                                 + 1.05642 w_k1_INTC_s001
                                 + 0.978028 w_k1_KO_s001
                                 + 0.808154 w_k1_MAT_s001
                                 + 0.962977 w_k1_MSFT_s001
                                 + 1.12849 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.954457 w_k1_ADM_s001
                                 + 0.949996 w_k1_AXP_s001
                                 + 1.12813 w_k1_DIS_s001
                                 + 1.08488 w_k1_FDX_s001
                                 + 1.10395 w_k1_IBM_s001
                                 + 0.896603 w_k1_INTC_s001
                                 + 1.10465 w_k1_KO_s001 + 1.17227 w_k1_MAT_s001
                                 + 0.97323 w_k1_MSFT_s001
                                 + 1.21138 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.33251 w_k1_ADM_s001
                                 + 1.3379 w_k1_AXP_s001 + 1.16464 w_k1_DIS_s001
                                 + 1.30272 w_k1_FDX_s001
                                 + 1.36539 w_k1_IBM_s001
                                 + 1.14286 w_k1_INTC_s001 + 1.271 w_k1_KO_s001
                                 + 1.57268 w_k1_MAT_s001
                                 + 1.17277 w_k1_MSFT_s001 + 1.2298 w_k1_PG_s001
                                 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.46652 w_k1_ADM_s001
                                 + 1.37795 w_k1_AXP_s001
                                 + 1.34755 w_k1_DIS_s001 + 1.4327 w_k1_FDX_s001
                                 + 1.1192 w_k1_IBM_s001
                                 + 0.934882 w_k1_INTC_s001
                                 + 1.26305 w_k1_KO_s001 + 1.23068 w_k1_MAT_s001
                                 + 1.08178 w_k1_MSFT_s001
                                 + 1.20655 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.07332 w_k1_ADM_s001
                                 + 0.969498 w_k1_AXP_s001
                                 + 1.14935 w_k1_DIS_s001
                                 + 1.26846 w_k1_FDX_s001
                                 + 0.845694 w_k1_IBM_s001
                                 + 1.15557 w_k1_INTC_s001
                                 + 1.02863 w_k1_KO_s001
                                 + 0.719641 w_k1_MAT_s001
                                 + 1.02168 w_k1_MSFT_s001
                                 + 1.17008 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.84397 w_k1_ADM_s001
                                 + 0.862425 w_k1_AXP_s001
                                 + 0.832268 w_k1_DIS_s001
                                 + 0.808237 w_k1_FDX_s001
                                 + 1.23869 w_k1_IBM_s001
                                 + 0.806788 w_k1_INTC_s001
                                 + 0.954617 w_k1_KO_s001
                                 + 1.21005 w_k1_MAT_s001
                                 + 0.760737 w_k1_MSFT_s001
                                 + 1.00447 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.49143 w_k1_ADM_s001
                                 + 1.32864 w_k1_AXP_s001
                                 + 1.55022 w_k1_DIS_s001
                                 + 1.28095 w_k1_FDX_s001
                                 + 1.77026 w_k1_IBM_s001
                                 + 0.996069 w_k1_INTC_s001
                                 + 1.54032 w_k1_KO_s001 + 1.43262 w_k1_MAT_s001
                                 + 1.10583 w_k1_MSFT_s001
                                 + 1.41397 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.831793 w_k1_ADM_s002
                                 + 1.09325 w_k1_AXP_s002
                                 + 0.870608 w_k1_DIS_s002
                                 + 0.942586 w_k1_FDX_s002
                                 + 0.91734 w_k1_IBM_s002
                                 + 0.902365 w_k1_INTC_s002
                                 + 1.0616 w_k1_KO_s002 + 1.05332 w_k1_MAT_s002
                                 + 0.965291 w_k1_MSFT_s002
                                 + 0.943705 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.09364 w_k1_ADM_s002
                                 + 1.24962 w_k1_AXP_s002
                                 + 1.01362 w_k1_DIS_s002
                                 + 1.00916 w_k1_FDX_s002
                                 + 1.45376 w_k1_IBM_s002
                                 + 1.07076 w_k1_INTC_s002
                                 + 1.27342 w_k1_KO_s002 + 1.41388 w_k1_MAT_s002
                                 + 1.08743 w_k1_MSFT_s002
                                 + 1.07902 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.07201 w_k1_ADM_s002
                                 + 0.831951 w_k1_AXP_s002
                                 + 0.793759 w_k1_DIS_s002
                                 + 0.912118 w_k1_FDX_s002
                                 + 0.572038 w_k1_IBM_s002
                                 + 0.999659 w_k1_INTC_s002
                                 + 0.743015 w_k1_KO_s002
                                 + 0.687372 w_k1_MAT_s002
                                 + 0.882842 w_k1_MSFT_s002
                                 + 0.941212 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.761787 w_k1_ADM_s002
                                 + 0.786412 w_k1_AXP_s002
                                 + 0.624413 w_k1_DIS_s002
                                 + 0.703993 w_k1_FDX_s002
                                 + 1.01368 w_k1_IBM_s002
                                 + 0.709354 w_k1_INTC_s002
                                 + 0.750396 w_k1_KO_s002
                                 + 0.962652 w_k1_MAT_s002
                                 + 0.688742 w_k1_MSFT_s002
                                 + 0.837524 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.76103 w_k1_ADM_s002
                                 + 0.721799 w_k1_AXP_s002
                                 + 0.639503 w_k1_DIS_s002
                                 + 0.767469 w_k1_FDX_s002
                                 + 0.63251 w_k1_IBM_s002
                                 + 0.931881 w_k1_INTC_s002
                                 + 0.751973 w_k1_KO_s002
                                 + 0.699224 w_k1_MAT_s002
                                 + 0.853099 w_k1_MSFT_s002
                                 + 0.834206 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.4903 w_k1_ADM_s002
                                 + 1.66023 w_k1_AXP_s002
                                 + 1.73924 w_k1_DIS_s002
                                 + 1.56328 w_k1_FDX_s002
                                 + 1.24027 w_k1_IBM_s002
                                 + 1.51681 w_k1_INTC_s002
                                 + 1.60642 w_k1_KO_s002 + 1.5329 w_k1_MAT_s002
                                 + 1.53543 w_k1_MSFT_s002
                                 + 1.37082 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.28746 w_k1_ADM_s002
                                 + 1.36011 w_k1_AXP_s002
                                 + 1.32412 w_k1_DIS_s002
                                 + 1.22848 w_k1_FDX_s002
                                 + 0.853576 w_k1_IBM_s002
                                 + 1.26038 w_k1_INTC_s002
                                 + 0.991549 w_k1_KO_s002
                                 + 0.775256 w_k1_MAT_s002
                                 + 1.22824 w_k1_MSFT_s002
                                 + 0.973995 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.933303 w_k1_ADM_s002
                                 + 0.974848 w_k1_AXP_s002
                                 + 0.954733 w_k1_DIS_s002
                                 + 0.995616 w_k1_FDX_s002
                                 + 0.742591 w_k1_IBM_s002
                                 + 1.02596 w_k1_INTC_s002
                                 + 0.853597 w_k1_KO_s002
                                 + 0.669528 w_k1_MAT_s002
                                 + 0.92952 w_k1_MSFT_s002
                                 + 0.96856 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.978047 w_k1_ADM_s003
                                 + 1.05833 w_k1_AXP_s003
                                 + 1.13837 w_k1_DIS_s003
                                 + 1.13622 w_k1_FDX_s003
                                 + 1.19534 w_k1_IBM_s003
                                 + 1.18215 w_k1_INTC_s003
                                 + 1.09775 w_k1_KO_s003 + 1.44466 w_k1_MAT_s003
                                 + 1.15803 w_k1_MSFT_s003
                                 + 1.01009 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.04738 w_k1_ADM_s003
                                 + 0.904761 w_k1_AXP_s003
                                 + 1.36976 w_k1_DIS_s003
                                 + 1.15534 w_k1_FDX_s003
                                 + 0.980264 w_k1_IBM_s003
                                 + 1.31723 w_k1_INTC_s003
                                 + 1.27453 w_k1_KO_s003
                                 + 0.989719 w_k1_MAT_s003
                                 + 1.1236 w_k1_MSFT_s003 + 1.05394 w_k1_PG_s003
                                 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.810499 w_k1_ADM_s003
                                 + 0.753917 w_k1_AXP_s003
                                 + 0.65988 w_k1_DIS_s003
                                 + 0.833765 w_k1_FDX_s003
                                 + 0.750169 w_k1_IBM_s003
                                 + 0.921055 w_k1_INTC_s003
                                 + 0.666845 w_k1_KO_s003
                                 + 0.754341 w_k1_MAT_s003
                                 + 0.8411 w_k1_MSFT_s003
                                 + 0.860592 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.0747 w_k1_ADM_s003
                                 + 1.16305 w_k1_AXP_s003
                                 + 1.10404 w_k1_DIS_s003
                                 + 1.25694 w_k1_FDX_s003
                                 + 0.764053 w_k1_IBM_s003
                                 + 1.19901 w_k1_INTC_s003
                                 + 0.840336 w_k1_KO_s003
                                 + 0.888821 w_k1_MAT_s003
                                 + 1.00797 w_k1_MSFT_s003
                                 + 0.917014 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.28957 w_k1_ADM_s003
                                 + 1.04524 w_k1_AXP_s003
                                 + 1.43393 w_k1_DIS_s003
                                 + 1.03608 w_k1_FDX_s003 + 1.5914 w_k1_IBM_s003
                                 + 1.19256 w_k1_INTC_s003
                                 + 1.28957 w_k1_KO_s003 + 1.20971 w_k1_MAT_s003
                                 + 1.11123 w_k1_MSFT_s003
                                 + 1.04323 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.18955 w_k1_ADM_s003
                                 + 0.976466 w_k1_AXP_s003
                                 + 0.770527 w_k1_DIS_s003
                                 + 0.878919 w_k1_FDX_s003
                                 + 1.1192 w_k1_IBM_s003
                                 + 1.01588 w_k1_INTC_s003
                                 + 0.917405 w_k1_KO_s003
                                 + 1.28839 w_k1_MAT_s003
                                 + 0.861799 w_k1_MSFT_s003
                                 + 0.944288 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.00959 w_k1_ADM_s003
                                 + 1.01215 w_k1_AXP_s003
                                 + 1.22017 w_k1_DIS_s003
                                 + 1.12017 w_k1_FDX_s003
                                 + 0.872108 w_k1_IBM_s003
                                 + 1.31098 w_k1_INTC_s003
                                 + 0.972321 w_k1_KO_s003
                                 + 0.801705 w_k1_MAT_s003
                                 + 1.04405 w_k1_MSFT_s003
                                 + 0.850177 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.953111 w_k1_ADM_s003
                                 + 0.781045 w_k1_AXP_s003
                                 + 0.964554 w_k1_DIS_s003
                                 + 0.93233 w_k1_FDX_s003
                                 + 1.08605 w_k1_IBM_s003
                                 + 1.14535 w_k1_INTC_s003
                                 + 0.93059 w_k1_KO_s003
                                 + 0.944501 w_k1_MAT_s003
                                 + 0.89101 w_k1_MSFT_s003
                                 + 0.871698 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.23046 w_k1_ADM_s004
                                 + 1.34773 w_k1_AXP_s004
                                 + 1.57313 w_k1_DIS_s004
                                 + 1.13164 w_k1_FDX_s004
                                 + 1.79144 w_k1_IBM_s004
                                 + 0.951596 w_k1_INTC_s004
                                 + 1.34374 w_k1_KO_s004 + 2.71433 w_k1_MAT_s004
                                 + 1.26099 w_k1_MSFT_s004
                                 + 1.20314 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.893931 w_k1_ADM_s004
                                 + 0.858213 w_k1_AXP_s004
                                 + 0.818232 w_k1_DIS_s004
                                 + 0.863234 w_k1_FDX_s004
                                 + 0.815724 w_k1_IBM_s004
                                 + 0.785534 w_k1_INTC_s004
                                 + 0.732868 w_k1_KO_s004
                                 + 0.615359 w_k1_MAT_s004
                                 + 0.793402 w_k1_MSFT_s004
                                 + 0.850571 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.23941 w_k1_ADM_s004
                                 + 1.50303 w_k1_AXP_s004
                                 + 1.78275 w_k1_DIS_s004
                                 + 1.50241 w_k1_FDX_s004 + 1.1605 w_k1_IBM_s004
                                 + 1.52487 w_k1_INTC_s004
                                 + 1.19827 w_k1_KO_s004 + 1.00615 w_k1_MAT_s004
                                 + 1.52345 w_k1_MSFT_s004
                                 + 1.16636 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.22857 w_k1_ADM_s004
                                 + 0.986355 w_k1_AXP_s004
                                 + 1.32905 w_k1_DIS_s004
                                 + 1.29591 w_k1_FDX_s004
                                 + 0.695091 w_k1_IBM_s004
                                 + 1.07503 w_k1_INTC_s004
                                 + 0.931744 w_k1_KO_s004
                                 + 0.684923 w_k1_MAT_s004
                                 + 1.23902 w_k1_MSFT_s004 + 1.055 w_k1_PG_s004
                                 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.3102 w_k1_ADM_s004
                                 + 1.32891 w_k1_AXP_s004
                                 + 1.42795 w_k1_DIS_s004
                                 + 1.16951 w_k1_FDX_s004
                                 + 1.23102 w_k1_IBM_s004
                                 + 0.950798 w_k1_INTC_s004
                                 + 1.27783 w_k1_KO_s004 + 1.26034 w_k1_MAT_s004
                                 + 1.02657 w_k1_MSFT_s004
                                 + 1.18257 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.897162 w_k1_ADM_s004
                                 + 0.931726 w_k1_AXP_s004
                                 + 0.921356 w_k1_DIS_s004
                                 + 0.837728 w_k1_FDX_s004
                                 + 1.56606 w_k1_IBM_s004
                                 + 0.864973 w_k1_INTC_s004
                                 + 1.03249 w_k1_KO_s004 + 1.69875 w_k1_MAT_s004
                                 + 0.863263 w_k1_MSFT_s004
                                 + 0.936251 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.01969 w_k1_ADM_s004
                                 + 1.13383 w_k1_AXP_s004
                                 + 1.04814 w_k1_DIS_s004
                                 + 0.997738 w_k1_FDX_s004
                                 + 1.01327 w_k1_IBM_s004
                                 + 1.10768 w_k1_INTC_s004
                                 + 0.94411 w_k1_KO_s004 + 1.13846 w_k1_MAT_s004
                                 + 0.963072 w_k1_MSFT_s004
                                 + 0.998469 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.668892 w_k1_ADM_s004
                                 + 0.597928 w_k1_AXP_s004
                                 + 0.610277 w_k1_DIS_s004
                                 + 0.634017 w_k1_FDX_s004
                                 + 0.859403 w_k1_IBM_s004
                                 + 0.581266 w_k1_INTC_s004
                                 + 0.652226 w_k1_KO_s004
                                 + 0.790171 w_k1_MAT_s004
                                 + 0.685674 w_k1_MSFT_s004
                                 + 0.758518 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.69679 w_k1_ADM_s005
                                 + 0.693637 w_k1_AXP_s005
                                 + 0.648177 w_k1_DIS_s005
                                 + 0.705207 w_k1_FDX_s005
                                 + 0.802951 w_k1_IBM_s005
                                 + 1.08845 w_k1_INTC_s005
                                 + 0.823457 w_k1_KO_s005
                                 + 0.477093 w_k1_MAT_s005
                                 + 0.890761 w_k1_MSFT_s005
                                 + 0.750506 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.945032 w_k1_ADM_s005
                                 + 1.30274 w_k1_AXP_s005
                                 + 0.993619 w_k1_DIS_s005
                                 + 0.999524 w_k1_FDX_s005
                                 + 1.44129 w_k1_IBM_s005
                                 + 1.10708 w_k1_INTC_s005
                                 + 1.16935 w_k1_KO_s005 + 1.42506 w_k1_MAT_s005
                                 + 1.01153 w_k1_MSFT_s005
                                 + 0.894706 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.07261 w_k1_ADM_s005
                                 + 1.15535 w_k1_AXP_s005
                                 + 1.22864 w_k1_DIS_s005
                                 + 1.08353 w_k1_FDX_s005 + 1.0371 w_k1_IBM_s005
                                 + 0.96373 w_k1_INTC_s005
                                 + 1.00263 w_k1_KO_s005
                                 + 0.844539 w_k1_MAT_s005
                                 + 0.995322 w_k1_MSFT_s005
                                 + 0.933634 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.803047 w_k1_ADM_s005
                                 + 1.07257 w_k1_AXP_s005
                                 + 0.779508 w_k1_DIS_s005
                                 + 0.880983 w_k1_FDX_s005
                                 + 1.00216 w_k1_IBM_s005
                                 + 1.22914 w_k1_INTC_s005
                                 + 0.975732 w_k1_KO_s005
                                 + 0.839402 w_k1_MAT_s005
                                 + 0.901496 w_k1_MSFT_s005
                                 + 0.855947 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.819878 w_k1_ADM_s005
                                 + 0.829597 w_k1_AXP_s005
                                 + 0.837356 w_k1_DIS_s005
                                 + 0.801014 w_k1_FDX_s005
                                 + 1.31395 w_k1_IBM_s005
                                 + 0.970381 w_k1_INTC_s005
                                 + 1.11926 w_k1_KO_s005
                                 + 0.813013 w_k1_MAT_s005
                                 + 0.864299 w_k1_MSFT_s005
                                 + 0.863607 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.42384 w_k1_ADM_s005
                                 + 1.54996 w_k1_AXP_s005
                                 + 1.63054 w_k1_DIS_s005
                                 + 1.23971 w_k1_FDX_s005
                                 + 1.47495 w_k1_IBM_s005
                                 + 1.52308 w_k1_INTC_s005
                                 + 1.39372 w_k1_KO_s005 + 1.38849 w_k1_MAT_s005
                                 + 1.21533 w_k1_MSFT_s005
                                 + 1.21398 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.08459 w_k1_ADM_s005
                                 + 1.05101 w_k1_AXP_s005
                                 + 1.11876 w_k1_DIS_s005
                                 + 1.16193 w_k1_FDX_s005
                                 + 0.677979 w_k1_IBM_s005
                                 + 1.19629 w_k1_INTC_s005
                                 + 0.842734 w_k1_KO_s005
                                 + 0.560329 w_k1_MAT_s005
                                 + 1.00771 w_k1_MSFT_s005
                                 + 0.937584 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.765304 w_k1_ADM_s005
                                 + 0.767925 w_k1_AXP_s005
                                 + 0.748861 w_k1_DIS_s005
                                 + 0.897254 w_k1_FDX_s005
                                 + 0.627534 w_k1_IBM_s005
                                 + 0.930583 w_k1_INTC_s005
                                 + 0.678438 w_k1_KO_s005
                                 + 0.464459 w_k1_MAT_s005
                                 + 0.802102 w_k1_MSFT_s005
                                 + 0.800038 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.898531 w_k1_ADM_s006
                                 + 0.940516 w_k1_AXP_s006
                                 + 1.01647 w_k1_DIS_s006
                                 + 0.903193 w_k1_FDX_s006
                                 + 1.17658 w_k1_IBM_s006
                                 + 0.959038 w_k1_INTC_s006
                                 + 1.00702 w_k1_KO_s006
                                 + 0.965325 w_k1_MAT_s006
                                 + 1.07275 w_k1_MSFT_s006
                                 + 0.948268 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.02891 w_k1_ADM_s006
                                 + 0.828539 w_k1_AXP_s006
                                 + 0.940858 w_k1_DIS_s006
                                 + 0.928085 w_k1_FDX_s006
                                 + 0.753593 w_k1_IBM_s006
                                 + 0.913149 w_k1_INTC_s006
                                 + 0.810892 w_k1_KO_s006
                                 + 0.567823 w_k1_MAT_s006
                                 + 0.96645 w_k1_MSFT_s006
                                 + 0.915302 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.695754 w_k1_ADM_s006
                                 + 0.707471 w_k1_AXP_s006
                                 + 0.680925 w_k1_DIS_s006
                                 + 0.703039 w_k1_FDX_s006
                                 + 1.00974 w_k1_IBM_s006
                                 + 0.849931 w_k1_INTC_s006
                                 + 0.884959 w_k1_KO_s006
                                 + 0.802676 w_k1_MAT_s006
                                 + 0.818061 w_k1_MSFT_s006
                                 + 0.793433 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.4515 w_k1_ADM_s006
                                 + 1.42305 w_k1_AXP_s006
                                 + 1.75423 w_k1_DIS_s006
                                 + 1.37776 w_k1_FDX_s006
                                 + 1.27307 w_k1_IBM_s006
                                 + 1.27823 w_k1_INTC_s006 + 1.2865 w_k1_KO_s006
                                 + 0.916918 w_k1_MAT_s006
                                 + 1.421 w_k1_MSFT_s006 + 1.24133 w_k1_PG_s006
                                 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.51477 w_k1_ADM_s006
                                 + 1.60811 w_k1_AXP_s006
                                 + 1.18307 w_k1_DIS_s006
                                 + 1.21858 w_k1_FDX_s006
                                 + 0.911618 w_k1_IBM_s006
                                 + 0.965636 w_k1_INTC_s006
                                 + 0.972469 w_k1_KO_s006
                                 + 0.982012 w_k1_MAT_s006
                                 + 1.00853 w_k1_MSFT_s006
                                 + 1.02231 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.974491 w_k1_ADM_s006
                                 + 0.941801 w_k1_AXP_s006
                                 + 0.667018 w_k1_DIS_s006
                                 + 0.898826 w_k1_FDX_s006
                                 + 0.71325 w_k1_IBM_s006 + 0.82 w_k1_INTC_s006
                                 + 0.79681 w_k1_KO_s006
                                 + 0.661624 w_k1_MAT_s006
                                 + 0.894721 w_k1_MSFT_s006
                                 + 0.778591 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.15006 w_k1_ADM_s006
                                 + 0.900629 w_k1_AXP_s006
                                 + 1.37854 w_k1_DIS_s006
                                 + 1.22405 w_k1_FDX_s006
                                 + 0.787441 w_k1_IBM_s006
                                 + 1.24164 w_k1_INTC_s006
                                 + 1.05287 w_k1_KO_s006
                                 + 0.579005 w_k1_MAT_s006
                                 + 1.3067 w_k1_MSFT_s006
                                 + 0.992183 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.0191 w_k1_ADM_s006
                                 + 1.33098 w_k1_AXP_s006
                                 + 1.28464 w_k1_DIS_s006
                                 + 1.03903 w_k1_FDX_s006
                                 + 1.75304 w_k1_IBM_s006 + 1.145 w_k1_INTC_s006
                                 + 1.2217 w_k1_KO_s006 + 1.95229 w_k1_MAT_s006
                                 + 1.16521 w_k1_MSFT_s006
                                 + 1.07095 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.38878 w_k1_ADM_s007
                                 + 1.43799 w_k1_AXP_s007
                                 + 1.19373 w_k1_DIS_s007
                                 + 1.29341 w_k1_FDX_s007
                                 + 0.836861 w_k1_IBM_s007
                                 + 1.28628 w_k1_INTC_s007
                                 + 1.08365 w_k1_KO_s007
                                 + 0.723284 w_k1_MAT_s007
                                 + 1.09425 w_k1_MSFT_s007
                                 + 1.13542 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.00462 w_k1_ADM_s007
                                 + 0.778146 w_k1_AXP_s007
                                 + 0.726082 w_k1_DIS_s007
                                 + 0.897204 w_k1_FDX_s007
                                 + 0.683101 w_k1_IBM_s007
                                 + 0.894003 w_k1_INTC_s007
                                 + 0.86644 w_k1_KO_s007
                                 + 0.825683 w_k1_MAT_s007
                                 + 0.96111 w_k1_MSFT_s007
                                 + 0.911627 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.954741 w_k1_ADM_s007
                                 + 1.04374 w_k1_AXP_s007
                                 + 1.03932 w_k1_DIS_s007
                                 + 1.02259 w_k1_FDX_s007
                                 + 1.06406 w_k1_IBM_s007
                                 + 1.15678 w_k1_INTC_s007
                                 + 1.19101 w_k1_KO_s007 + 1.40378 w_k1_MAT_s007
                                 + 1.047 w_k1_MSFT_s007 + 1.05553 w_k1_PG_s007
                                 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.50092 w_k1_ADM_s007
                                 + 1.20553 w_k1_AXP_s007
                                 + 1.45708 w_k1_DIS_s007
                                 + 1.29323 w_k1_FDX_s007
                                 + 0.824969 w_k1_IBM_s007
                                 + 1.37678 w_k1_INTC_s007 + 1.3547 w_k1_KO_s007
                                 + 0.773904 w_k1_MAT_s007
                                 + 1.38155 w_k1_MSFT_s007
                                 + 1.12476 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.16144 w_k1_ADM_s007
                                 + 0.792963 w_k1_AXP_s007
                                 + 0.707709 w_k1_DIS_s007
                                 + 0.977716 w_k1_FDX_s007
                                 + 0.42562 w_k1_IBM_s007
                                 + 0.804191 w_k1_INTC_s007
                                 + 0.648562 w_k1_KO_s007
                                 + 0.594055 w_k1_MAT_s007
                                 + 0.971515 w_k1_MSFT_s007
                                 + 0.941927 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.34355 w_k1_ADM_s007
                                 + 1.37911 w_k1_AXP_s007 + 1.3301 w_k1_DIS_s007
                                 + 1.15569 w_k1_FDX_s007
                                 + 1.56069 w_k1_IBM_s007
                                 + 0.823298 w_k1_INTC_s007
                                 + 1.6249 w_k1_KO_s007 + 2.01252 w_k1_MAT_s007
                                 + 1.05975 w_k1_MSFT_s007
                                 + 1.20719 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.19999 w_k1_ADM_s007
                                 + 1.1443 w_k1_AXP_s007 + 0.74343 w_k1_DIS_s007
                                 + 0.906741 w_k1_FDX_s007
                                 + 1.0224 w_k1_IBM_s007
                                 + 0.931154 w_k1_INTC_s007
                                 + 0.911545 w_k1_KO_s007
                                 + 1.26248 w_k1_MAT_s007
                                 + 0.876963 w_k1_MSFT_s007
                                 + 1.05769 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.852882 w_k1_ADM_s007
                                 + 0.728912 w_k1_AXP_s007
                                 + 0.722562 w_k1_DIS_s007
                                 + 0.765897 w_k1_FDX_s007
                                 + 1.23994 w_k1_IBM_s007
                                 + 0.878159 w_k1_INTC_s007
                                 + 1.10878 w_k1_KO_s007 + 1.48735 w_k1_MAT_s007
                                 + 0.859143 w_k1_MSFT_s007
                                 + 0.982084 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.00743 w_k1_ADM_s010
                                 + 1.05172 w_k1_AXP_s010
                                 + 0.933745 w_k1_DIS_s010
                                 + 0.978008 w_k1_FDX_s010
                                 + 0.832152 w_k1_IBM_s010
                                 + 0.9132 w_k1_INTC_s010
                                 + 0.853308 w_k1_KO_s010
                                 + 0.685275 w_k1_MAT_s010
                                 + 1.08165 w_k1_MSFT_s010
                                 + 0.998959 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.0123 w_k1_ADM_s010
                                 + 1.04574 w_k1_AXP_s010 + 0.9585 w_k1_DIS_s010
                                 + 0.883982 w_k1_FDX_s010
                                 + 1.19466 w_k1_IBM_s010
                                 + 0.944272 w_k1_INTC_s010
                                 + 1.06645 w_k1_KO_s010
                                 + 0.698724 w_k1_MAT_s010
                                 + 1.04463 w_k1_MSFT_s010
                                 + 1.05486 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.30398 w_k1_ADM_s010
                                 + 1.20871 w_k1_AXP_s010
                                 + 1.34062 w_k1_DIS_s010
                                 + 1.18091 w_k1_FDX_s010
                                 + 1.00679 w_k1_IBM_s010
                                 + 0.860483 w_k1_INTC_s010
                                 + 1.02869 w_k1_KO_s010 + 1.30718 w_k1_MAT_s010
                                 + 1.14856 w_k1_MSFT_s010
                                 + 1.20834 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.11408 w_k1_ADM_s010
                                 + 1.39136 w_k1_AXP_s010
                                 + 0.927013 w_k1_DIS_s010
                                 + 1.02688 w_k1_FDX_s010 + 1.1301 w_k1_IBM_s010
                                 + 1.04255 w_k1_INTC_s010
                                 + 1.09258 w_k1_KO_s010 + 1.08767 w_k1_MAT_s010
                                 + 1.03111 w_k1_MSFT_s010
                                 + 1.11238 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.45914 w_k1_ADM_s010
                                 + 1.69454 w_k1_AXP_s010
                                 + 1.12377 w_k1_DIS_s010
                                 + 1.12779 w_k1_FDX_s010
                                 + 1.56978 w_k1_IBM_s010
                                 + 1.03266 w_k1_INTC_s010
                                 + 1.34655 w_k1_KO_s010 + 1.72617 w_k1_MAT_s010
                                 + 1.13309 w_k1_MSFT_s010
                                 + 1.30015 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.783019 w_k1_ADM_s010
                                 + 1.0539 w_k1_AXP_s010
                                 + 0.696388 w_k1_DIS_s010
                                 + 0.781165 w_k1_FDX_s010
                                 + 1.15445 w_k1_IBM_s010
                                 + 0.991643 w_k1_INTC_s010
                                 + 0.892934 w_k1_KO_s010
                                 + 0.963574 w_k1_MAT_s010
                                 + 0.8461 w_k1_MSFT_s010
                                 + 0.982117 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.00748 w_k1_ADM_s010
                                 + 1.01369 w_k1_AXP_s010
                                 + 0.729157 w_k1_DIS_s010
                                 + 0.892707 w_k1_FDX_s010
                                 + 0.679769 w_k1_IBM_s010
                                 + 0.72866 w_k1_INTC_s010
                                 + 0.824675 w_k1_KO_s010
                                 + 0.843891 w_k1_MAT_s010
                                 + 0.868211 w_k1_MSFT_s010
                                 + 0.933546 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.9136 w_k1_ADM_s010
                                 + 0.849795 w_k1_AXP_s010
                                 + 0.621861 w_k1_DIS_s010
                                 + 0.727058 w_k1_FDX_s010
                                 + 0.891972 w_k1_IBM_s010
                                 + 0.863951 w_k1_INTC_s010
                                 + 0.856152 w_k1_KO_s010
                                 + 0.82045 w_k1_MAT_s010
                                 + 0.848021 w_k1_MSFT_s010
                                 + 0.961334 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.02023 w_k1_ADM_s011
                                 + 1.01927 w_k1_AXP_s011
                                 + 0.920228 w_k1_DIS_s011
                                 + 0.872359 w_k1_FDX_s011
                                 + 1.03777 w_k1_IBM_s011
                                 + 1.10317 w_k1_INTC_s011
                                 + 1.10436 w_k1_KO_s011
                                 + 0.895588 w_k1_MAT_s011
                                 + 1.04014 w_k1_MSFT_s011
                                 + 0.928584 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.12574 w_k1_ADM_s011
                                 + 0.953617 w_k1_AXP_s011
                                 + 1.52779 w_k1_DIS_s011
                                 + 1.10211 w_k1_FDX_s011
                                 + 1.11433 w_k1_IBM_s011
                                 + 1.34115 w_k1_INTC_s011 + 1.3839 w_k1_KO_s011
                                 + 1.13901 w_k1_MAT_s011
                                 + 1.23684 w_k1_MSFT_s011
                                 + 1.08641 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.986969 w_k1_ADM_s011
                                 + 1.42828 w_k1_AXP_s011
                                 + 1.12985 w_k1_DIS_s011
                                 + 1.07385 w_k1_FDX_s011
                                 + 1.00675 w_k1_IBM_s011
                                 + 1.43151 w_k1_INTC_s011
                                 + 1.06083 w_k1_KO_s011 + 1.09769 w_k1_MAT_s011
                                 + 1.09294 w_k1_MSFT_s011
                                 + 0.918124 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.903856 w_k1_ADM_s011
                                 + 0.768607 w_k1_AXP_s011
                                 + 0.787615 w_k1_DIS_s011
                                 + 0.7802 w_k1_FDX_s011
                                 + 0.679299 w_k1_IBM_s011
                                 + 0.932756 w_k1_INTC_s011
                                 + 0.880414 w_k1_KO_s011
                                 + 0.757231 w_k1_MAT_s011
                                 + 0.970666 w_k1_MSFT_s011
                                 + 0.838295 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.01555 w_k1_ADM_s011
                                 + 1.11816 w_k1_AXP_s011
                                 + 0.89336 w_k1_DIS_s011
                                 + 0.879489 w_k1_FDX_s011
                                 + 1.40372 w_k1_IBM_s011
                                 + 1.02455 w_k1_INTC_s011
                                 + 1.18121 w_k1_KO_s011 + 1.52751 w_k1_MAT_s011
                                 + 0.912565 w_k1_MSFT_s011
                                 + 0.928811 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.776731 w_k1_ADM_s011
                                 + 0.629354 w_k1_AXP_s011
                                 + 0.914228 w_k1_DIS_s011
                                 + 0.696588 w_k1_FDX_s011
                                 + 0.847318 w_k1_IBM_s011
                                 + 0.951017 w_k1_INTC_s011
                                 + 1.06233 w_k1_KO_s011
                                 + 0.682763 w_k1_MAT_s011
                                 + 0.888403 w_k1_MSFT_s011
                                 + 0.791615 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.29679 w_k1_ADM_s011
                                 + 0.952506 w_k1_AXP_s011
                                 + 0.841166 w_k1_DIS_s011
                                 + 1.06466 w_k1_FDX_s011
                                 + 0.717021 w_k1_IBM_s011
                                 + 1.05752 w_k1_INTC_s011
                                 + 0.904034 w_k1_KO_s011
                                 + 0.691554 w_k1_MAT_s011
                                 + 1.03734 w_k1_MSFT_s011
                                 + 0.944479 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.64735 w_k1_ADM_s011
                                 + 1.67514 w_k1_AXP_s011 + 1.4773 w_k1_DIS_s011
                                 + 1.41004 w_k1_FDX_s011
                                 + 1.08033 w_k1_IBM_s011
                                 + 1.27192 w_k1_INTC_s011
                                 + 1.27846 w_k1_KO_s011 + 1.61696 w_k1_MAT_s011
                                 + 1.2754 w_k1_MSFT_s011 + 1.2868 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.83561 w_k1_ADM_s012
                                 + 0.828072 w_k1_AXP_s012
                                 + 0.738272 w_k1_DIS_s012
                                 + 0.751884 w_k1_FDX_s012
                                 + 1.25431 w_k1_IBM_s012
                                 + 0.851802 w_k1_INTC_s012
                                 + 0.970998 w_k1_KO_s012
                                 + 1.18845 w_k1_MAT_s012
                                 + 0.818812 w_k1_MSFT_s012
                                 + 0.959756 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.25112 w_k1_ADM_s012
                                 + 1.31094 w_k1_AXP_s012
                                 + 1.34534 w_k1_DIS_s012
                                 + 1.33293 w_k1_FDX_s012 + 1.0224 w_k1_IBM_s012
                                 + 1.42513 w_k1_INTC_s012 + 1.2793 w_k1_KO_s012
                                 + 1.14671 w_k1_MAT_s012
                                 + 1.37471 w_k1_MSFT_s012
                                 + 1.27063 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.97038 w_k1_ADM_s012
                                 + 1.1653 w_k1_AXP_s012 + 1.31536 w_k1_DIS_s012
                                 + 1.04737 w_k1_FDX_s012
                                 + 1.55269 w_k1_IBM_s012
                                 + 1.06121 w_k1_INTC_s012
                                 + 1.33078 w_k1_KO_s012 + 1.2705 w_k1_MAT_s012
                                 + 1.01291 w_k1_MSFT_s012
                                 + 1.16352 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.11231 w_k1_ADM_s012
                                 + 1.12859 w_k1_AXP_s012
                                 + 1.35988 w_k1_DIS_s012
                                 + 1.13785 w_k1_FDX_s012
                                 + 1.22669 w_k1_IBM_s012
                                 + 1.00096 w_k1_INTC_s012
                                 + 1.24672 w_k1_KO_s012 + 1.00283 w_k1_MAT_s012
                                 + 1.07857 w_k1_MSFT_s012
                                 + 0.938165 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.968462 w_k1_ADM_s012
                                 + 1.2177 w_k1_AXP_s012
                                 + 0.877071 w_k1_DIS_s012
                                 + 1.00773 w_k1_FDX_s012
                                 + 1.00376 w_k1_IBM_s012
                                 + 0.943384 w_k1_INTC_s012
                                 + 0.969811 w_k1_KO_s012
                                 + 1.08736 w_k1_MAT_s012
                                 + 0.961883 w_k1_MSFT_s012
                                 + 0.827273 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.731703 w_k1_ADM_s012
                                 + 0.651181 w_k1_AXP_s012
                                 + 0.673463 w_k1_DIS_s012
                                 + 0.710446 w_k1_FDX_s012
                                 + 0.800784 w_k1_IBM_s012
                                 + 0.773515 w_k1_INTC_s012
                                 + 0.716025 w_k1_KO_s012
                                 + 0.813917 w_k1_MAT_s012
                                 + 0.801343 w_k1_MSFT_s012
                                 + 0.916179 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.21782 w_k1_ADM_s012
                                 + 1.54612 w_k1_AXP_s012
                                 + 1.61883 w_k1_DIS_s012
                                 + 1.41947 w_k1_FDX_s012
                                 + 1.43164 w_k1_IBM_s012
                                 + 1.21295 w_k1_INTC_s012
                                 + 1.32986 w_k1_KO_s012 + 1.68726 w_k1_MAT_s012
                                 + 1.17823 w_k1_MSFT_s012
                                 + 1.19081 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.12685 w_k1_ADM_s012
                                 + 1.0043 w_k1_AXP_s012 + 1.15727 w_k1_DIS_s012
                                 + 1.08824 w_k1_FDX_s012
                                 + 0.838744 w_k1_IBM_s012
                                 + 1.02631 w_k1_INTC_s012
                                 + 1.10239 w_k1_KO_s012
                                 + 0.727134 w_k1_MAT_s012
                                 + 1.11384 w_k1_MSFT_s012
                                 + 1.00348 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.948074 w_k1_ADM_s013
                                 + 0.876051 w_k1_AXP_s013
                                 + 0.873999 w_k1_DIS_s013
                                 + 0.906519 w_k1_FDX_s013
                                 + 0.881293 w_k1_IBM_s013
                                 + 1.09592 w_k1_INTC_s013
                                 + 0.952933 w_k1_KO_s013
                                 + 0.810089 w_k1_MAT_s013
                                 + 0.945875 w_k1_MSFT_s013
                                 + 0.897938 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.10164 w_k1_ADM_s013
                                 + 1.76454 w_k1_AXP_s013
                                 + 1.30862 w_k1_DIS_s013
                                 + 1.38154 w_k1_FDX_s013
                                 + 1.21918 w_k1_IBM_s013
                                 + 1.52747 w_k1_INTC_s013
                                 + 1.11448 w_k1_KO_s013 + 1.01709 w_k1_MAT_s013
                                 + 1.20277 w_k1_MSFT_s013
                                 + 1.17179 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.870993 w_k1_ADM_s013
                                 + 0.871644 w_k1_AXP_s013
                                 + 0.935028 w_k1_DIS_s013
                                 + 0.938576 w_k1_FDX_s013
                                 + 1.07783 w_k1_IBM_s013
                                 + 1.15703 w_k1_INTC_s013
                                 + 1.02722 w_k1_KO_s013
                                 + 0.926928 w_k1_MAT_s013
                                 + 1.04268 w_k1_MSFT_s013
                                 + 0.918459 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.02825 w_k1_ADM_s013
                                 + 0.89585 w_k1_AXP_s013
                                 + 1.17449 w_k1_DIS_s013
                                 + 1.07647 w_k1_FDX_s013
                                 + 1.14817 w_k1_IBM_s013
                                 + 1.28119 w_k1_INTC_s013
                                 + 1.19267 w_k1_KO_s013 + 1.30273 w_k1_MAT_s013
                                 + 1.33541 w_k1_MSFT_s013
                                 + 1.20949 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.15728 w_k1_ADM_s013
                                 + 1.02002 w_k1_AXP_s013
                                 + 1.03642 w_k1_DIS_s013
                                 + 1.08311 w_k1_FDX_s013
                                 + 0.805052 w_k1_IBM_s013
                                 + 1.38442 w_k1_INTC_s013
                                 + 0.995989 w_k1_KO_s013
                                 + 0.95729 w_k1_MAT_s013
                                 + 1.12488 w_k1_MSFT_s013
                                 + 1.05903 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.923673 w_k1_ADM_s013
                                 + 0.901399 w_k1_AXP_s013
                                 + 0.825317 w_k1_DIS_s013
                                 + 0.851364 w_k1_FDX_s013
                                 + 1.37891 w_k1_IBM_s013
                                 + 1.06994 w_k1_INTC_s013
                                 + 1.16691 w_k1_KO_s013
                                 + 0.966026 w_k1_MAT_s013
                                 + 0.985176 w_k1_MSFT_s013
                                 + 1.05346 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.901044 w_k1_ADM_s013
                                 + 0.830056 w_k1_AXP_s013
                                 + 0.65821 w_k1_DIS_s013
                                 + 0.802226 w_k1_FDX_s013
                                 + 0.674876 w_k1_IBM_s013
                                 + 0.997479 w_k1_INTC_s013
                                 + 0.8481 w_k1_KO_s013 + 0.934229 w_k1_MAT_s013
                                 + 0.940656 w_k1_MSFT_s013
                                 + 0.961709 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.873205 w_k1_ADM_s013
                                 + 1.38784 w_k1_AXP_s013
                                 + 1.18177 w_k1_DIS_s013
                                 + 1.03768 w_k1_FDX_s013
                                 + 1.60875 w_k1_IBM_s013
                                 + 1.2114 w_k1_INTC_s013 + 1.08673 w_k1_KO_s013
                                 + 1.7124 w_k1_MAT_s013
                                 + 1.02659 w_k1_MSFT_s013
                                 + 1.16663 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.947528 w_k1_ADM_s014
                                 + 0.984969 w_k1_AXP_s014
                                 + 0.625942 w_k1_DIS_s014
                                 + 0.80545 w_k1_FDX_s014
                                 + 0.907284 w_k1_IBM_s014
                                 + 0.914126 w_k1_INTC_s014
                                 + 0.694068 w_k1_KO_s014
                                 + 0.722003 w_k1_MAT_s014
                                 + 0.727986 w_k1_MSFT_s014
                                 + 0.899114 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.887173 w_k1_ADM_s014
                                 + 1.07283 w_k1_AXP_s014
                                 + 1.31198 w_k1_DIS_s014
                                 + 1.21804 w_k1_FDX_s014
                                 + 0.972775 w_k1_IBM_s014
                                 + 1.30733 w_k1_INTC_s014
                                 + 1.07537 w_k1_KO_s014
                                 + 0.700136 w_k1_MAT_s014
                                 + 1.08457 w_k1_MSFT_s014
                                 + 1.07219 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.869732 w_k1_ADM_s014
                                 + 1.14335 w_k1_AXP_s014
                                 + 1.22471 w_k1_DIS_s014
                                 + 1.10455 w_k1_FDX_s014
                                 + 1.97306 w_k1_IBM_s014
                                 + 1.03514 w_k1_INTC_s014
                                 + 1.42804 w_k1_KO_s014 + 1.58807 w_k1_MAT_s014
                                 + 0.991686 w_k1_MSFT_s014
                                 + 1.11649 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.835603 w_k1_ADM_s014
                                 + 1.5441 w_k1_AXP_s014 + 1.66581 w_k1_DIS_s014
                                 + 1.42625 w_k1_FDX_s014
                                 + 1.24889 w_k1_IBM_s014
                                 + 1.45106 w_k1_INTC_s014
                                 + 1.24425 w_k1_KO_s014 + 1.11688 w_k1_MAT_s014
                                 + 1.15279 w_k1_MSFT_s014
                                 + 1.17647 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.787973 w_k1_ADM_s014
                                 + 0.731627 w_k1_AXP_s014
                                 + 0.832884 w_k1_DIS_s014
                                 + 0.972522 w_k1_FDX_s014
                                 + 0.656352 w_k1_IBM_s014
                                 + 0.881447 w_k1_INTC_s014
                                 + 0.688172 w_k1_KO_s014
                                 + 0.54426 w_k1_MAT_s014
                                 + 0.770291 w_k1_MSFT_s014
                                 + 0.951174 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.1679 w_k1_ADM_s014
                                 + 1.07809 w_k1_AXP_s014 + 1.1023 w_k1_DIS_s014
                                 + 1.11121 w_k1_FDX_s014
                                 + 0.732845 w_k1_IBM_s014
                                 + 1.34992 w_k1_INTC_s014
                                 + 0.967769 w_k1_KO_s014
                                 + 0.539771 w_k1_MAT_s014
                                 + 0.909052 w_k1_MSFT_s014
                                 + 1.14623 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.974188 w_k1_ADM_s014
                                 + 1.29536 w_k1_AXP_s014
                                 + 1.26445 w_k1_DIS_s014
                                 + 1.21883 w_k1_FDX_s014
                                 + 1.02416 w_k1_IBM_s014
                                 + 1.10668 w_k1_INTC_s014
                                 + 0.929288 w_k1_KO_s014
                                 + 0.997063 w_k1_MAT_s014
                                 + 0.85509 w_k1_MSFT_s014 + 1.042 w_k1_PG_s014
                                 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.06292 w_k1_ADM_s014
                                 + 1.11799 w_k1_AXP_s014
                                 + 0.995821 w_k1_DIS_s014
                                 + 0.977863 w_k1_FDX_s014
                                 + 1.24218 w_k1_IBM_s014
                                 + 1.01674 w_k1_INTC_s014
                                 + 1.08767 w_k1_KO_s014 + 1.26055 w_k1_MAT_s014
                                 + 0.894155 w_k1_MSFT_s014
                                 + 1.12544 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.965075 w_k1_ADM_s015
                                 + 0.731319 w_k1_AXP_s015
                                 + 0.69157 w_k1_DIS_s015
                                 + 0.867796 w_k1_FDX_s015
                                 + 0.602682 w_k1_IBM_s015
                                 + 0.867548 w_k1_INTC_s015
                                 + 0.777644 w_k1_KO_s015
                                 + 0.690621 w_k1_MAT_s015
                                 + 0.96548 w_k1_MSFT_s015
                                 + 0.76589 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.05403 w_k1_ADM_s015
                                 + 1.18614 w_k1_AXP_s015
                                 + 1.12435 w_k1_DIS_s015
                                 + 1.18671 w_k1_FDX_s015
                                 + 0.863691 w_k1_IBM_s015
                                 + 1.11098 w_k1_INTC_s015
                                 + 0.927783 w_k1_KO_s015
                                 + 0.915433 w_k1_MAT_s015
                                 + 1.06588 w_k1_MSFT_s015
                                 + 0.969297 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.83763 w_k1_ADM_s015
                                 + 0.742596 w_k1_AXP_s015
                                 + 0.935049 w_k1_DIS_s015
                                 + 0.923781 w_k1_FDX_s015
                                 + 1.13191 w_k1_IBM_s015
                                 + 0.940456 w_k1_INTC_s015
                                 + 1.13826 w_k1_KO_s015 + 1.16655 w_k1_MAT_s015
                                 + 0.978463 w_k1_MSFT_s015
                                 + 1.00576 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.12877 w_k1_ADM_s015
                                 + 1.39484 w_k1_AXP_s015
                                 + 1.61601 w_k1_DIS_s015
                                 + 1.35644 w_k1_FDX_s015
                                 + 1.17957 w_k1_IBM_s015
                                 + 1.14069 w_k1_INTC_s015
                                 + 1.13662 w_k1_KO_s015 + 1.48526 w_k1_MAT_s015
                                 + 1.34099 w_k1_MSFT_s015
                                 + 1.17253 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.16054 w_k1_ADM_s015
                                 + 0.863017 w_k1_AXP_s015
                                 + 1.18423 w_k1_DIS_s015 + 1.0141 w_k1_FDX_s015
                                 + 1.46772 w_k1_IBM_s015
                                 + 1.01784 w_k1_INTC_s015
                                 + 1.36521 w_k1_KO_s015 + 1.46261 w_k1_MAT_s015
                                 + 0.973062 w_k1_MSFT_s015
                                 + 1.11594 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.902707 w_k1_ADM_s015
                                 + 0.772384 w_k1_AXP_s015
                                 + 0.699725 w_k1_DIS_s015
                                 + 0.79521 w_k1_FDX_s015
                                 + 1.07221 w_k1_IBM_s015
                                 + 0.790993 w_k1_INTC_s015
                                 + 0.912533 w_k1_KO_s015
                                 + 1.12817 w_k1_MAT_s015
                                 + 0.889554 w_k1_MSFT_s015
                                 + 0.891209 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.13458 w_k1_ADM_s015
                                 + 0.85136 w_k1_AXP_s015
                                 + 1.00228 w_k1_DIS_s015
                                 + 1.03425 w_k1_FDX_s015
                                 + 0.770829 w_k1_IBM_s015
                                 + 0.852529 w_k1_INTC_s015
                                 + 0.961895 w_k1_KO_s015
                                 + 0.703184 w_k1_MAT_s015
                                 + 0.967176 w_k1_MSFT_s015
                                 + 1.02287 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.17808 w_k1_ADM_s015
                                 + 1.38633 w_k1_AXP_s015
                                 + 1.11963 w_k1_DIS_s015
                                 + 1.22165 w_k1_FDX_s015
                                 + 1.32437 w_k1_IBM_s015
                                 + 1.11554 w_k1_INTC_s015
                                 + 1.24419 w_k1_KO_s015 + 1.63108 w_k1_MAT_s015
                                 + 1.01421 w_k1_MSFT_s015
                                 + 1.01139 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.15884 w_k1_ADM_s016
                                 + 1.01522 w_k1_AXP_s016
                                 + 1.01259 w_k1_DIS_s016
                                 + 1.10211 w_k1_FDX_s016
                                 + 1.15999 w_k1_IBM_s016
                                 + 1.38125 w_k1_INTC_s016
                                 + 1.00178 w_k1_KO_s016
                                 + 0.782141 w_k1_MAT_s016
                                 + 1.03614 w_k1_MSFT_s016 + 1.0036 w_k1_PG_s016
                                 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.841313 w_k1_ADM_s016
                                 + 0.728705 w_k1_AXP_s016
                                 + 0.707209 w_k1_DIS_s016
                                 + 0.804978 w_k1_FDX_s016
                                 + 0.671659 w_k1_IBM_s016
                                 + 0.911912 w_k1_INTC_s016
                                 + 0.728612 w_k1_KO_s016
                                 + 0.642423 w_k1_MAT_s016
                                 + 0.808563 w_k1_MSFT_s016
                                 + 0.798738 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.39345 w_k1_ADM_s016
                                 + 1.29133 w_k1_AXP_s016
                                 + 1.43144 w_k1_DIS_s016 + 1.3159 w_k1_FDX_s016
                                 + 1.34921 w_k1_IBM_s016
                                 + 1.12273 w_k1_INTC_s016
                                 + 1.28929 w_k1_KO_s016 + 1.17265 w_k1_MAT_s016
                                 + 1.30122 w_k1_MSFT_s016
                                 + 1.25166 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.33795 w_k1_ADM_s016
                                 + 0.964156 w_k1_AXP_s016
                                 + 1.45729 w_k1_DIS_s016
                                 + 1.17338 w_k1_FDX_s016
                                 + 0.89697 w_k1_IBM_s016
                                 + 1.3976 w_k1_INTC_s016 + 1.15292 w_k1_KO_s016
                                 + 0.762422 w_k1_MAT_s016
                                 + 1.40538 w_k1_MSFT_s016
                                 + 1.18036 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.820764 w_k1_ADM_s016
                                 + 0.819304 w_k1_AXP_s016
                                 + 0.925317 w_k1_DIS_s016
                                 + 0.915417 w_k1_FDX_s016
                                 + 0.931909 w_k1_IBM_s016
                                 + 1.14739 w_k1_INTC_s016
                                 + 0.870608 w_k1_KO_s016
                                 + 0.619399 w_k1_MAT_s016
                                 + 0.961374 w_k1_MSFT_s016
                                 + 0.871346 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.898298 w_k1_ADM_s016
                                 + 0.877338 w_k1_AXP_s016
                                 + 1.03638 w_k1_DIS_s016
                                 + 0.97336 w_k1_FDX_s016
                                 + 1.34616 w_k1_IBM_s016
                                 + 1.17883 w_k1_INTC_s016
                                 + 1.22767 w_k1_KO_s016 + 1.30238 w_k1_MAT_s016
                                 + 1.02794 w_k1_MSFT_s016
                                 + 0.961793 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.32455 w_k1_ADM_s016
                                 + 1.04486 w_k1_AXP_s016
                                 + 1.07955 w_k1_DIS_s016
                                 + 1.11823 w_k1_FDX_s016
                                 + 0.771914 w_k1_IBM_s016
                                 + 0.989853 w_k1_INTC_s016
                                 + 0.871546 w_k1_KO_s016
                                 + 0.624613 w_k1_MAT_s016
                                 + 1.09186 w_k1_MSFT_s016
                                 + 1.04992 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.910486 w_k1_ADM_s016
                                 + 0.815522 w_k1_AXP_s016
                                 + 0.865702 w_k1_DIS_s016
                                 + 0.792444 w_k1_FDX_s016
                                 + 1.21305 w_k1_IBM_s016
                                 + 0.814326 w_k1_INTC_s016
                                 + 0.989281 w_k1_KO_s016
                                 + 1.11875 w_k1_MAT_s016
                                 + 0.825512 w_k1_MSFT_s016
                                 + 0.977311 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.54647 w_k1_ADM_s017
                                 + 1.5929 w_k1_AXP_s017 + 1.35711 w_k1_DIS_s017
                                 + 1.15232 w_k1_FDX_s017
                                 + 1.52137 w_k1_IBM_s017
                                 + 0.944047 w_k1_INTC_s017
                                 + 1.28826 w_k1_KO_s017 + 1.56171 w_k1_MAT_s017
                                 + 0.99143 w_k1_MSFT_s017
                                 + 1.20652 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.01591 w_k1_ADM_s017
                                 + 0.852848 w_k1_AXP_s017
                                 + 0.920925 w_k1_DIS_s017
                                 + 1.05523 w_k1_FDX_s017
                                 + 0.751423 w_k1_IBM_s017
                                 + 0.812325 w_k1_INTC_s017
                                 + 0.807174 w_k1_KO_s017
                                 + 0.675781 w_k1_MAT_s017
                                 + 0.942874 w_k1_MSFT_s017
                                 + 0.981775 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.00143 w_k1_ADM_s017
                                 + 0.610055 w_k1_AXP_s017
                                 + 0.60266 w_k1_DIS_s017
                                 + 0.79361 w_k1_FDX_s017
                                 + 0.643128 w_k1_IBM_s017
                                 + 0.789888 w_k1_INTC_s017
                                 + 0.695759 w_k1_KO_s017
                                 + 0.614732 w_k1_MAT_s017
                                 + 0.807026 w_k1_MSFT_s017
                                 + 0.913983 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.00584 w_k1_ADM_s017
                                 + 1.32758 w_k1_AXP_s017
                                 + 0.875482 w_k1_DIS_s017
                                 + 0.89087 w_k1_FDX_s017
                                 + 1.67696 w_k1_IBM_s017
                                 + 0.81737 w_k1_INTC_s017
                                 + 0.956088 w_k1_KO_s017
                                 + 1.89654 w_k1_MAT_s017
                                 + 0.865514 w_k1_MSFT_s017
                                 + 0.943015 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.30171 w_k1_ADM_s017
                                 + 1.16067 w_k1_AXP_s017
                                 + 1.61035 w_k1_DIS_s017
                                 + 1.33627 w_k1_FDX_s017
                                 + 1.03263 w_k1_IBM_s017
                                 + 1.12993 w_k1_INTC_s017
                                 + 1.11947 w_k1_KO_s017 + 1.34459 w_k1_MAT_s017
                                 + 1.32594 w_k1_MSFT_s017
                                 + 1.16148 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.891902 w_k1_ADM_s017
                                 + 0.763532 w_k1_AXP_s017
                                 + 1.12717 w_k1_DIS_s017
                                 + 1.15172 w_k1_FDX_s017
                                 + 0.689298 w_k1_IBM_s017
                                 + 0.924953 w_k1_INTC_s017
                                 + 0.833982 w_k1_KO_s017
                                 + 0.551403 w_k1_MAT_s017
                                 + 1.04919 w_k1_MSFT_s017
                                 + 0.913763 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.07266 w_k1_ADM_s017
                                 + 0.987548 w_k1_AXP_s017
                                 + 0.685404 w_k1_DIS_s017
                                 + 0.875594 w_k1_FDX_s017
                                 + 1.03928 w_k1_IBM_s017
                                 + 0.748745 w_k1_INTC_s017
                                 + 0.752616 w_k1_KO_s017
                                 + 1.02519 w_k1_MAT_s017
                                 + 0.779177 w_k1_MSFT_s017
                                 + 0.949751 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.0256 w_k1_ADM_s017
                                 + 0.915289 w_k1_AXP_s017
                                 + 1.01209 w_k1_DIS_s017
                                 + 0.878214 w_k1_FDX_s017
                                 + 1.18061 w_k1_IBM_s017
                                 + 0.940924 w_k1_INTC_s017
                                 + 0.887705 w_k1_KO_s017
                                 + 0.689245 w_k1_MAT_s017
                                 + 0.923039 w_k1_MSFT_s017
                                 + 0.904705 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.762432 w_k1_ADM_s100
                                 + 0.87638 w_k1_AXP_s100
                                 + 0.724145 w_k1_DIS_s100
                                 + 0.760828 w_k1_FDX_s100
                                 + 1.09558 w_k1_IBM_s100
                                 + 0.802024 w_k1_INTC_s100
                                 + 0.819076 w_k1_KO_s100
                                 + 1.02952 w_k1_MAT_s100
                                 + 0.848467 w_k1_MSFT_s100
                                 + 0.863732 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.71878 w_k1_ADM_s100
                                 + 0.650904 w_k1_AXP_s100
                                 + 0.761583 w_k1_DIS_s100
                                 + 0.718397 w_k1_FDX_s100
                                 + 0.638792 w_k1_IBM_s100
                                 + 0.723354 w_k1_INTC_s100
                                 + 0.740701 w_k1_KO_s100
                                 + 0.390987 w_k1_MAT_s100
                                 + 0.809134 w_k1_MSFT_s100
                                 + 0.780047 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.25384 w_k1_ADM_s100
                                 + 1.71767 w_k1_AXP_s100
                                 + 1.90271 w_k1_DIS_s100 + 1.3873 w_k1_FDX_s100
                                 + 1.39132 w_k1_IBM_s100
                                 + 1.46921 w_k1_INTC_s100
                                 + 1.68057 w_k1_KO_s100 + 1.96377 w_k1_MAT_s100
                                 + 1.43955 w_k1_MSFT_s100
                                 + 1.40437 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.844004 w_k1_ADM_s100
                                 + 1.04081 w_k1_AXP_s100
                                 + 0.903965 w_k1_DIS_s100
                                 + 1.01099 w_k1_FDX_s100
                                 + 0.813607 w_k1_IBM_s100
                                 + 1.04527 w_k1_INTC_s100
                                 + 0.850398 w_k1_KO_s100
                                 + 0.979862 w_k1_MAT_s100
                                 + 0.903311 w_k1_MSFT_s100
                                 + 0.949506 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.1832 w_k1_ADM_s100
                                 + 1.20941 w_k1_AXP_s100
                                 + 1.17288 w_k1_DIS_s100
                                 + 1.22193 w_k1_FDX_s100
                                 + 0.706956 w_k1_IBM_s100
                                 + 1.00294 w_k1_INTC_s100
                                 + 0.941878 w_k1_KO_s100
                                 + 0.82775 w_k1_MAT_s100
                                 + 1.18916 w_k1_MSFT_s100
                                 + 0.97801 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.0657 w_k1_ADM_s100
                                 + 1.57774 w_k1_AXP_s100
                                 + 1.03778 w_k1_DIS_s100
                                 + 1.11569 w_k1_FDX_s100
                                 + 1.16176 w_k1_IBM_s100
                                 + 1.10806 w_k1_INTC_s100
                                 + 0.934508 w_k1_KO_s100
                                 + 1.34583 w_k1_MAT_s100
                                 + 1.07113 w_k1_MSFT_s100
                                 + 1.10834 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.16085 w_k1_ADM_s100
                                 + 1.21632 w_k1_AXP_s100
                                 + 1.53766 w_k1_DIS_s100
                                 + 1.10239 w_k1_FDX_s100
                                 + 1.27576 w_k1_IBM_s100
                                 + 1.30406 w_k1_INTC_s100
                                 + 1.31948 w_k1_KO_s100 + 1.05599 w_k1_MAT_s100
                                 + 1.29751 w_k1_MSFT_s100
                                 + 1.11696 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.748701 w_k1_ADM_s100
                                 + 1.07129 w_k1_AXP_s100
                                 + 1.09895 w_k1_DIS_s100
                                 + 0.896775 w_k1_FDX_s100
                                 + 1.54712 w_k1_IBM_s100
                                 + 0.837608 w_k1_INTC_s100
                                 + 1.29099 w_k1_KO_s100 + 1.90843 w_k1_MAT_s100
                                 + 1.02919 w_k1_MSFT_s100
                                 + 0.983261 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.05856 w_k1_ADM_s101
                                 + 1.0255 w_k1_AXP_s101 + 1.00783 w_k1_DIS_s101
                                 + 0.989827 w_k1_FDX_s101
                                 + 1.05337 w_k1_IBM_s101
                                 + 1.0765 w_k1_INTC_s101 + 1.1336 w_k1_KO_s101
                                 + 0.902987 w_k1_MAT_s101
                                 + 1.01691 w_k1_MSFT_s101
                                 + 0.92864 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.15549 w_k1_ADM_s101
                                 + 1.51035 w_k1_AXP_s101
                                 + 1.51582 w_k1_DIS_s101
                                 + 1.25006 w_k1_FDX_s101
                                 + 1.56316 w_k1_IBM_s101
                                 + 1.33168 w_k1_INTC_s101
                                 + 1.24322 w_k1_KO_s101 + 1.18926 w_k1_MAT_s101
                                 + 1.1219 w_k1_MSFT_s101 + 1.19021 w_k1_PG_s101
                                 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.970745 w_k1_ADM_s101
                                 + 0.903038 w_k1_AXP_s101
                                 + 0.784034 w_k1_DIS_s101
                                 + 0.906653 w_k1_FDX_s101
                                 + 0.846332 w_k1_IBM_s101
                                 + 0.898018 w_k1_INTC_s101
                                 + 0.823928 w_k1_KO_s101
                                 + 0.848195 w_k1_MAT_s101
                                 + 0.900241 w_k1_MSFT_s101
                                 + 0.925345 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.764701 w_k1_ADM_s101
                                 + 0.704328 w_k1_AXP_s101
                                 + 0.682374 w_k1_DIS_s101
                                 + 0.78883 w_k1_FDX_s101
                                 + 0.923399 w_k1_IBM_s101
                                 + 0.812678 w_k1_INTC_s101
                                 + 0.752567 w_k1_KO_s101
                                 + 0.748696 w_k1_MAT_s101
                                 + 0.764306 w_k1_MSFT_s101
                                 + 0.809105 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.95402 w_k1_ADM_s101
                                 + 0.92766 w_k1_AXP_s101
                                 + 1.00222 w_k1_DIS_s101
                                 + 1.03674 w_k1_FDX_s101
                                 + 1.31362 w_k1_IBM_s101
                                 + 0.923797 w_k1_INTC_s101
                                 + 1.15561 w_k1_KO_s101 + 1.31011 w_k1_MAT_s101
                                 + 0.970304 w_k1_MSFT_s101
                                 + 0.957803 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.889355 w_k1_ADM_s101
                                 + 0.774237 w_k1_AXP_s101
                                 + 0.950147 w_k1_DIS_s101
                                 + 1.01702 w_k1_FDX_s101
                                 + 0.748645 w_k1_IBM_s101
                                 + 0.999755 w_k1_INTC_s101
                                 + 0.927618 w_k1_KO_s101
                                 + 0.646742 w_k1_MAT_s101
                                 + 1.08394 w_k1_MSFT_s101
                                 + 0.930562 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.38322 w_k1_ADM_s101
                                 + 0.944605 w_k1_AXP_s101
                                 + 1.01614 w_k1_DIS_s101
                                 + 1.24798 w_k1_FDX_s101
                                 + 0.83532 w_k1_IBM_s101
                                 + 1.04216 w_k1_INTC_s101
                                 + 1.03568 w_k1_KO_s101
                                 + 0.741888 w_k1_MAT_s101
                                 + 1.19807 w_k1_MSFT_s101
                                 + 1.07115 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.21385 w_k1_ADM_s101
                                 + 1.2472 w_k1_AXP_s101 + 1.37627 w_k1_DIS_s101
                                 + 1.29764 w_k1_FDX_s101
                                 + 1.03931 w_k1_IBM_s101
                                 + 0.986179 w_k1_INTC_s101
                                 + 1.16085 w_k1_KO_s101 + 1.27944 w_k1_MAT_s101
                                 + 1.39653 w_k1_MSFT_s101
                                 + 1.15465 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.823768 w_k1_ADM_s102
                                 + 0.916009 w_k1_AXP_s102
                                 + 1.25658 w_k1_DIS_s102
                                 + 1.03928 w_k1_FDX_s102
                                 + 1.01317 w_k1_IBM_s102
                                 + 1.1207 w_k1_INTC_s102 + 1.02761 w_k1_KO_s102
                                 + 0.852747 w_k1_MAT_s102
                                 + 1.13711 w_k1_MSFT_s102
                                 + 1.11516 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.24588 w_k1_ADM_s102
                                 + 1.10126 w_k1_AXP_s102
                                 + 1.32505 w_k1_DIS_s102
                                 + 1.27091 w_k1_FDX_s102
                                 + 0.804461 w_k1_IBM_s102
                                 + 1.32007 w_k1_INTC_s102 + 1.0244 w_k1_KO_s102
                                 + 0.63037 w_k1_MAT_s102
                                 + 1.09605 w_k1_MSFT_s102 + 1.1534 w_k1_PG_s102
                                 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.97399 w_k1_ADM_s102
                                 + 1.09102 w_k1_AXP_s102 + 1.2355 w_k1_DIS_s102
                                 + 0.994902 w_k1_FDX_s102
                                 + 1.59776 w_k1_IBM_s102
                                 + 1.1935 w_k1_INTC_s102 + 1.20897 w_k1_KO_s102
                                 + 1.25992 w_k1_MAT_s102
                                 + 1.11464 w_k1_MSFT_s102
                                 + 1.19417 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.26242 w_k1_ADM_s102
                                 + 1.88245 w_k1_AXP_s102
                                 + 1.56685 w_k1_DIS_s102
                                 + 1.34516 w_k1_FDX_s102
                                 + 1.11499 w_k1_IBM_s102
                                 + 1.51824 w_k1_INTC_s102
                                 + 1.20422 w_k1_KO_s102 + 1.27232 w_k1_MAT_s102
                                 + 1.31358 w_k1_MSFT_s102
                                 + 1.37564 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.782167 w_k1_ADM_s102
                                 + 0.694568 w_k1_AXP_s102
                                 + 0.664528 w_k1_DIS_s102
                                 + 0.759282 w_k1_FDX_s102
                                 + 0.756757 w_k1_IBM_s102
                                 + 0.829557 w_k1_INTC_s102
                                 + 0.701524 w_k1_KO_s102
                                 + 0.649458 w_k1_MAT_s102
                                 + 0.752623 w_k1_MSFT_s102
                                 + 0.875378 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.31095 w_k1_ADM_s102
                                 + 1.32034 w_k1_AXP_s102
                                 + 0.918273 w_k1_DIS_s102
                                 + 0.977264 w_k1_FDX_s102
                                 + 1.12888 w_k1_IBM_s102
                                 + 0.857485 w_k1_INTC_s102
                                 + 0.954529 w_k1_KO_s102
                                 + 1.59294 w_k1_MAT_s102
                                 + 0.760637 w_k1_MSFT_s102
                                 + 1.09965 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.00271 w_k1_ADM_s102
                                 + 1.11472 w_k1_AXP_s102
                                 + 1.05096 w_k1_DIS_s102
                                 + 0.896182 w_k1_FDX_s102
                                 + 1.28083 w_k1_IBM_s102
                                 + 1.05112 w_k1_INTC_s102
                                 + 1.04148 w_k1_KO_s102 + 1.23011 w_k1_MAT_s102
                                 + 0.922739 w_k1_MSFT_s102
                                 + 1.08864 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.768999 w_k1_ADM_s102
                                 + 0.752631 w_k1_AXP_s102
                                 + 0.87396 w_k1_DIS_s102
                                 + 0.788038 w_k1_FDX_s102
                                 + 0.974503 w_k1_IBM_s102
                                 + 0.855774 w_k1_INTC_s102
                                 + 0.841901 w_k1_KO_s102
                                 + 0.986777 w_k1_MAT_s102
                                 + 0.891852 w_k1_MSFT_s102
                                 + 1.03701 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.963285 w_k1_ADM_s103
                                 + 0.9769 w_k1_AXP_s103 + 1.09287 w_k1_DIS_s103
                                 + 1.09488 w_k1_FDX_s103
                                 + 0.775302 w_k1_IBM_s103
                                 + 1.09835 w_k1_INTC_s103
                                 + 0.952468 w_k1_KO_s103
                                 + 0.599227 w_k1_MAT_s103
                                 + 1.13491 w_k1_MSFT_s103
                                 + 1.02223 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.824758 w_k1_ADM_s103
                                 + 0.810661 w_k1_AXP_s103
                                 + 1.08941 w_k1_DIS_s103
                                 + 0.944475 w_k1_FDX_s103
                                 + 1.08081 w_k1_IBM_s103
                                 + 1.06859 w_k1_INTC_s103 + 1.0586 w_k1_KO_s103
                                 + 0.767408 w_k1_MAT_s103
                                 + 1.09301 w_k1_MSFT_s103
                                 + 0.999401 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.9774 w_k1_ADM_s103
                                 + 0.759831 w_k1_AXP_s103
                                 + 0.816896 w_k1_DIS_s103
                                 + 0.924773 w_k1_FDX_s103
                                 + 0.554421 w_k1_IBM_s103
                                 + 0.884673 w_k1_INTC_s103
                                 + 0.686947 w_k1_KO_s103
                                 + 0.434671 w_k1_MAT_s103
                                 + 1.01057 w_k1_MSFT_s103
                                 + 0.891326 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.860893 w_k1_ADM_s103
                                 + 0.833903 w_k1_AXP_s103
                                 + 0.680367 w_k1_DIS_s103
                                 + 0.840375 w_k1_FDX_s103
                                 + 0.806809 w_k1_IBM_s103
                                 + 0.777351 w_k1_INTC_s103
                                 + 0.761325 w_k1_KO_s103
                                 + 0.606975 w_k1_MAT_s103
                                 + 0.82269 w_k1_MSFT_s103
                                 + 0.834292 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.0478 w_k1_ADM_s103
                                 + 1.26458 w_k1_AXP_s103
                                 + 1.41786 w_k1_DIS_s103
                                 + 1.39273 w_k1_FDX_s103
                                 + 1.09231 w_k1_IBM_s103
                                 + 1.25158 w_k1_INTC_s103
                                 + 1.08484 w_k1_KO_s103
                                 + 0.918282 w_k1_MAT_s103
                                 + 1.39922 w_k1_MSFT_s103
                                 + 1.02279 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.991448 w_k1_ADM_s103
                                 + 1.10594 w_k1_AXP_s103
                                 + 0.762866 w_k1_DIS_s103
                                 + 1.07614 w_k1_FDX_s103
                                 + 0.756184 w_k1_IBM_s103
                                 + 1.09828 w_k1_INTC_s103
                                 + 0.800624 w_k1_KO_s103
                                 + 0.876996 w_k1_MAT_s103
                                 + 1.02161 w_k1_MSFT_s103
                                 + 1.03982 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.13307 w_k1_ADM_s103
                                 + 1.24018 w_k1_AXP_s103
                                 + 1.29191 w_k1_DIS_s103
                                 + 1.03364 w_k1_FDX_s103
                                 + 1.52185 w_k1_IBM_s103
                                 + 1.11358 w_k1_INTC_s103
                                 + 1.41302 w_k1_KO_s103 + 1.31902 w_k1_MAT_s103
                                 + 1.29994 w_k1_MSFT_s103
                                 + 1.13409 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.886563 w_k1_ADM_s103
                                 + 1.29094 w_k1_AXP_s103
                                 + 0.977086 w_k1_DIS_s103
                                 + 1.02405 w_k1_FDX_s103
                                 + 1.11696 w_k1_IBM_s103
                                 + 1.03364 w_k1_INTC_s103
                                 + 0.850388 w_k1_KO_s103
                                 + 1.14586 w_k1_MAT_s103
                                 + 1.01619 w_k1_MSFT_s103
                                 + 0.925983 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.03237 w_k1_ADM_s104
                                 + 1.36854 w_k1_AXP_s104
                                 + 1.48032 w_k1_DIS_s104 + 1.2887 w_k1_FDX_s104
                                 + 1.01699 w_k1_IBM_s104
                                 + 1.25371 w_k1_INTC_s104
                                 + 1.13866 w_k1_KO_s104
                                 + 0.939249 w_k1_MAT_s104
                                 + 1.17012 w_k1_MSFT_s104
                                 + 1.12795 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.18784 w_k1_ADM_s104
                                 + 0.974026 w_k1_AXP_s104
                                 + 1.21062 w_k1_DIS_s104
                                 + 0.958331 w_k1_FDX_s104
                                 + 0.953762 w_k1_IBM_s104
                                 + 0.886448 w_k1_INTC_s104
                                 + 1.11792 w_k1_KO_s104
                                 + 0.796787 w_k1_MAT_s104
                                 + 1.04355 w_k1_MSFT_s104
                                 + 1.04673 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.02158 w_k1_ADM_s104
                                 + 0.98732 w_k1_AXP_s104
                                 + 0.870842 w_k1_DIS_s104
                                 + 0.895041 w_k1_FDX_s104
                                 + 1.12747 w_k1_IBM_s104
                                 + 0.9066 w_k1_INTC_s104
                                 + 0.925148 w_k1_KO_s104
                                 + 0.835352 w_k1_MAT_s104
                                 + 0.878736 w_k1_MSFT_s104
                                 + 1.03945 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.97868 w_k1_ADM_s104
                                 + 1.20972 w_k1_AXP_s104
                                 + 0.951286 w_k1_DIS_s104
                                 + 0.836748 w_k1_FDX_s104
                                 + 1.43452 w_k1_IBM_s104
                                 + 0.905962 w_k1_INTC_s104
                                 + 1.19879 w_k1_KO_s104 + 1.03213 w_k1_MAT_s104
                                 + 0.962602 w_k1_MSFT_s104
                                 + 1.05749 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.03124 w_k1_ADM_s104
                                 + 1.14206 w_k1_AXP_s104
                                 + 1.01424 w_k1_DIS_s104
                                 + 1.09425 w_k1_FDX_s104
                                 + 0.691931 w_k1_IBM_s104
                                 + 1.11616 w_k1_INTC_s104
                                 + 0.884511 w_k1_KO_s104
                                 + 0.682129 w_k1_MAT_s104
                                 + 0.966905 w_k1_MSFT_s104
                                 + 0.945794 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.845701 w_k1_ADM_s104
                                 + 0.860424 w_k1_AXP_s104
                                 + 0.733043 w_k1_DIS_s104
                                 + 0.712856 w_k1_FDX_s104
                                 + 0.951352 w_k1_IBM_s104
                                 + 0.591786 w_k1_INTC_s104
                                 + 0.728969 w_k1_KO_s104
                                 + 0.772407 w_k1_MAT_s104
                                 + 0.722162 w_k1_MSFT_s104
                                 + 0.835855 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.31566 w_k1_ADM_s104
                                 + 1.49531 w_k1_AXP_s104
                                 + 1.76616 w_k1_DIS_s104
                                 + 1.17558 w_k1_FDX_s104
                                 + 1.38651 w_k1_IBM_s104
                                 + 0.970813 w_k1_INTC_s104
                                 + 1.51817 w_k1_KO_s104 + 1.40543 w_k1_MAT_s104
                                 + 1.21279 w_k1_MSFT_s104
                                 + 1.17378 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.901866 w_k1_ADM_s104
                                 + 0.801761 w_k1_AXP_s104
                                 + 0.795693 w_k1_DIS_s104
                                 + 0.842986 w_k1_FDX_s104
                                 + 0.757832 w_k1_IBM_s104
                                 + 0.728473 w_k1_INTC_s104
                                 + 0.884602 w_k1_KO_s104
                                 + 0.744836 w_k1_MAT_s104
                                 + 0.883744 w_k1_MSFT_s104
                                 + 0.911652 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.960943 w_k1_ADM_s105
                                 + 1.22591 w_k1_AXP_s105
                                 + 0.891413 w_k1_DIS_s105
                                 + 0.939816 w_k1_FDX_s105
                                 + 1.42894 w_k1_IBM_s105
                                 + 1.09832 w_k1_INTC_s105
                                 + 1.14955 w_k1_KO_s105 + 1.39535 w_k1_MAT_s105
                                 + 0.932803 w_k1_MSFT_s105
                                 + 1.10176 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.898353 w_k1_ADM_s105
                                 + 0.790742 w_k1_AXP_s105
                                 + 0.600579 w_k1_DIS_s105
                                 + 0.716232 w_k1_FDX_s105
                                 + 1.0886 w_k1_IBM_s105
                                 + 0.817715 w_k1_INTC_s105
                                 + 0.899109 w_k1_KO_s105
                                 + 1.04347 w_k1_MAT_s105
                                 + 0.773713 w_k1_MSFT_s105
                                 + 0.837838 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.56354 w_k1_ADM_s105
                                 + 1.36074 w_k1_AXP_s105
                                 + 1.44941 w_k1_DIS_s105
                                 + 1.29891 w_k1_FDX_s105
                                 + 0.984848 w_k1_IBM_s105
                                 + 1.15662 w_k1_INTC_s105
                                 + 1.14048 w_k1_KO_s105
                                 + 0.888158 w_k1_MAT_s105
                                 + 1.31042 w_k1_MSFT_s105
                                 + 1.09193 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.1611 w_k1_ADM_s105
                                 + 1.18972 w_k1_AXP_s105
                                 + 1.17116 w_k1_DIS_s105
                                 + 0.994812 w_k1_FDX_s105
                                 + 1.87088 w_k1_IBM_s105
                                 + 0.969325 w_k1_INTC_s105
                                 + 1.4487 w_k1_KO_s105 + 1.98555 w_k1_MAT_s105
                                 + 0.962236 w_k1_MSFT_s105
                                 + 1.05478 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.50089 w_k1_ADM_s105
                                 + 1.32983 w_k1_AXP_s105
                                 + 1.89361 w_k1_DIS_s105
                                 + 1.32058 w_k1_FDX_s105
                                 + 1.40612 w_k1_IBM_s105
                                 + 1.09215 w_k1_INTC_s105
                                 + 1.54313 w_k1_KO_s105 + 1.83703 w_k1_MAT_s105
                                 + 1.35744 w_k1_MSFT_s105
                                 + 1.22066 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.16388 w_k1_ADM_s105
                                 + 1.28341 w_k1_AXP_s105
                                 + 1.12202 w_k1_DIS_s105
                                 + 1.09458 w_k1_FDX_s105
                                 + 0.96052 w_k1_IBM_s105
                                 + 1.05732 w_k1_INTC_s105 + 1.1621 w_k1_KO_s105
                                 + 1.35395 w_k1_MAT_s105
                                 + 1.01954 w_k1_MSFT_s105
                                 + 1.05389 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.999645 w_k1_ADM_s105
                                 + 0.691508 w_k1_AXP_s105
                                 + 0.659175 w_k1_DIS_s105
                                 + 0.855116 w_k1_FDX_s105
                                 + 0.723689 w_k1_IBM_s105
                                 + 0.907242 w_k1_INTC_s105
                                 + 0.821695 w_k1_KO_s105
                                 + 0.887656 w_k1_MAT_s105
                                 + 0.957084 w_k1_MSFT_s105
                                 + 0.839265 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.816967 w_k1_ADM_s105
                                 + 0.990973 w_k1_AXP_s105
                                 + 0.865736 w_k1_DIS_s105
                                 + 0.947178 w_k1_FDX_s105
                                 + 1.10547 w_k1_IBM_s105
                                 + 0.989028 w_k1_INTC_s105
                                 + 0.927481 w_k1_KO_s105
                                 + 1.21053 w_k1_MAT_s105
                                 + 0.949285 w_k1_MSFT_s105 + 1.009 w_k1_PG_s105
                                 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.03465 w_k1_ADM_s106
                                 + 1.09828 w_k1_AXP_s106
                                 + 1.33941 w_k1_DIS_s106
                                 + 1.20612 w_k1_FDX_s106
                                 + 0.871235 w_k1_IBM_s106
                                 + 1.3805 w_k1_INTC_s106 + 1.25259 w_k1_KO_s106
                                 + 0.969868 w_k1_MAT_s106
                                 + 1.17363 w_k1_MSFT_s106
                                 + 0.970844 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.819969 w_k1_ADM_s106
                                 + 0.819192 w_k1_AXP_s106
                                 + 0.783702 w_k1_DIS_s106
                                 + 0.775955 w_k1_FDX_s106
                                 + 0.969573 w_k1_IBM_s106
                                 + 0.924179 w_k1_INTC_s106
                                 + 0.813525 w_k1_KO_s106
                                 + 0.966391 w_k1_MAT_s106
                                 + 0.839399 w_k1_MSFT_s106
                                 + 0.771178 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.867965 w_k1_ADM_s106
                                 + 1.39927 w_k1_AXP_s106
                                 + 1.30704 w_k1_DIS_s106
                                 + 1.20557 w_k1_FDX_s106 + 1.6329 w_k1_IBM_s106
                                 + 1.16423 w_k1_INTC_s106 + 1.548 w_k1_KO_s106
                                 + 2.07462 w_k1_MAT_s106
                                 + 1.18954 w_k1_MSFT_s106
                                 + 1.08149 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.00132 w_k1_ADM_s106
                                 + 1.09412 w_k1_AXP_s106
                                 + 1.39423 w_k1_DIS_s106
                                 + 1.10738 w_k1_FDX_s106
                                 + 1.16949 w_k1_IBM_s106
                                 + 1.28594 w_k1_INTC_s106
                                 + 1.25687 w_k1_KO_s106 + 1.34841 w_k1_MAT_s106
                                 + 1.21887 w_k1_MSFT_s106
                                 + 1.02114 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.875791 w_k1_ADM_s106
                                 + 0.732007 w_k1_AXP_s106
                                 + 0.875591 w_k1_DIS_s106
                                 + 0.946983 w_k1_FDX_s106
                                 + 0.869528 w_k1_IBM_s106
                                 + 1.1898 w_k1_INTC_s106
                                 + 0.994076 w_k1_KO_s106
                                 + 0.763583 w_k1_MAT_s106
                                 + 1.16968 w_k1_MSFT_s106
                                 + 0.862176 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.785713 w_k1_ADM_s106
                                 + 0.637559 w_k1_AXP_s106
                                 + 0.653323 w_k1_DIS_s106
                                 + 0.846913 w_k1_FDX_s106
                                 + 0.714055 w_k1_IBM_s106
                                 + 0.95313 w_k1_INTC_s106
                                 + 0.726558 w_k1_KO_s106
                                 + 0.642007 w_k1_MAT_s106
                                 + 0.875591 w_k1_MSFT_s106
                                 + 0.80484 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.959139 w_k1_ADM_s106
                                 + 1.01395 w_k1_AXP_s106
                                 + 0.94062 w_k1_DIS_s106
                                 + 0.994951 w_k1_FDX_s106
                                 + 0.7838 w_k1_IBM_s106
                                 + 0.990357 w_k1_INTC_s106
                                 + 0.914067 w_k1_KO_s106
                                 + 1.03723 w_k1_MAT_s106
                                 + 1.08132 w_k1_MSFT_s106
                                 + 0.877642 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.12159 w_k1_ADM_s106
                                 + 1.36397 w_k1_AXP_s106
                                 + 1.86417 w_k1_DIS_s106
                                 + 1.39666 w_k1_FDX_s106
                                 + 0.881386 w_k1_IBM_s106
                                 + 1.57261 w_k1_INTC_s106
                                 + 1.23165 w_k1_KO_s106 + 1.36475 w_k1_MAT_s106
                                 + 1.51024 w_k1_MSFT_s106
                                 + 1.21702 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.682628 w_k1_ADM_s107
                                 + 0.710127 w_k1_AXP_s107
                                 + 0.573974 w_k1_DIS_s107
                                 + 0.741735 w_k1_FDX_s107
                                 + 0.78089 w_k1_IBM_s107
                                 + 0.767978 w_k1_INTC_s107
                                 + 0.732214 w_k1_KO_s107
                                 + 0.908237 w_k1_MAT_s107
                                 + 0.723734 w_k1_MSFT_s107
                                 + 0.851757 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.928007 w_k1_ADM_s107
                                 + 0.967365 w_k1_AXP_s107
                                 + 0.763073 w_k1_DIS_s107
                                 + 0.819171 w_k1_FDX_s107
                                 + 0.928451 w_k1_IBM_s107
                                 + 0.763753 w_k1_INTC_s107
                                 + 0.749032 w_k1_KO_s107
                                 + 0.808909 w_k1_MAT_s107
                                 + 0.757495 w_k1_MSFT_s107
                                 + 0.901164 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.972815 w_k1_ADM_s107
                                 + 0.77986 w_k1_AXP_s107
                                 + 0.693618 w_k1_DIS_s107
                                 + 0.788107 w_k1_FDX_s107
                                 + 0.558358 w_k1_IBM_s107
                                 + 0.766394 w_k1_INTC_s107
                                 + 0.65968 w_k1_KO_s107
                                 + 0.595186 w_k1_MAT_s107
                                 + 0.747933 w_k1_MSFT_s107
                                 + 0.947094 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.802505 w_k1_ADM_s107
                                 + 0.958835 w_k1_AXP_s107
                                 + 0.866716 w_k1_DIS_s107
                                 + 0.855475 w_k1_FDX_s107
                                 + 1.22366 w_k1_IBM_s107
                                 + 0.936952 w_k1_INTC_s107
                                 + 0.99166 w_k1_KO_s107
                                 + 0.932122 w_k1_MAT_s107
                                 + 0.859291 w_k1_MSFT_s107
                                 + 0.946578 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.769427 w_k1_ADM_s107
                                 + 0.750297 w_k1_AXP_s107
                                 + 0.853583 w_k1_DIS_s107
                                 + 0.86442 w_k1_FDX_s107
                                 + 0.82026 w_k1_IBM_s107
                                 + 1.05748 w_k1_INTC_s107
                                 + 1.00807 w_k1_KO_s107
                                 + 0.977871 w_k1_MAT_s107
                                 + 1.04303 w_k1_MSFT_s107
                                 + 1.02179 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.29776 w_k1_ADM_s107
                                 + 1.35548 w_k1_AXP_s107
                                 + 1.52849 w_k1_DIS_s107
                                 + 1.32244 w_k1_FDX_s107
                                 + 1.10658 w_k1_IBM_s107
                                 + 1.13477 w_k1_INTC_s107
                                 + 1.34937 w_k1_KO_s107 + 1.15078 w_k1_MAT_s107
                                 + 1.16374 w_k1_MSFT_s107 + 1.3096 w_k1_PG_s107
                                 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.05445 w_k1_ADM_s107
                                 + 1.21801 w_k1_AXP_s107
                                 + 1.15522 w_k1_DIS_s107
                                 + 1.02806 w_k1_FDX_s107
                                 + 1.42519 w_k1_IBM_s107
                                 + 0.878068 w_k1_INTC_s107
                                 + 1.18225 w_k1_KO_s107 + 1.81363 w_k1_MAT_s107
                                 + 0.896954 w_k1_MSFT_s107
                                 + 1.19496 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.12578 w_k1_ADM_s107
                                 + 0.929829 w_k1_AXP_s107
                                 + 0.975725 w_k1_DIS_s107
                                 + 1.12443 w_k1_FDX_s107
                                 + 0.853629 w_k1_IBM_s107
                                 + 1.13954 w_k1_INTC_s107
                                 + 0.892293 w_k1_KO_s107
                                 + 0.74367 w_k1_MAT_s107
                                 + 0.954715 w_k1_MSFT_s107
                                 + 1.04989 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.20837 w_k1_ADM_s110
                                 + 0.916516 w_k1_AXP_s110
                                 + 0.917095 w_k1_DIS_s110
                                 + 0.996301 w_k1_FDX_s110
                                 + 0.943838 w_k1_IBM_s110
                                 + 1.06667 w_k1_INTC_s110
                                 + 0.94579 w_k1_KO_s110
                                 + 0.766139 w_k1_MAT_s110
                                 + 1.08263 w_k1_MSFT_s110
                                 + 0.905803 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.711672 w_k1_ADM_s110
                                 + 0.873221 w_k1_AXP_s110
                                 + 0.767937 w_k1_DIS_s110
                                 + 0.775861 w_k1_FDX_s110
                                 + 1.19804 w_k1_IBM_s110
                                 + 0.954818 w_k1_INTC_s110
                                 + 0.815055 w_k1_KO_s110
                                 + 0.993447 w_k1_MAT_s110
                                 + 0.841059 w_k1_MSFT_s110
                                 + 0.840553 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.44936 w_k1_ADM_s110
                                 + 1.38492 w_k1_AXP_s110
                                 + 1.82886 w_k1_DIS_s110 + 1.3958 w_k1_FDX_s110
                                 + 1.25254 w_k1_IBM_s110
                                 + 1.34263 w_k1_INTC_s110
                                 + 1.34223 w_k1_KO_s110 + 1.51993 w_k1_MAT_s110
                                 + 1.42826 w_k1_MSFT_s110
                                 + 1.20137 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.994575 w_k1_ADM_s110
                                 + 0.984548 w_k1_AXP_s110
                                 + 1.0713 w_k1_DIS_s110
                                 + 0.983704 w_k1_FDX_s110
                                 + 1.4133 w_k1_IBM_s110
                                 + 1.10454 w_k1_INTC_s110
                                 + 1.08457 w_k1_KO_s110 + 1.3977 w_k1_MAT_s110
                                 + 1.04735 w_k1_MSFT_s110 + 1.0976 w_k1_PG_s110
                                 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.878876 w_k1_ADM_s110
                                 + 1.08255 w_k1_AXP_s110
                                 + 0.979104 w_k1_DIS_s110
                                 + 0.928292 w_k1_FDX_s110
                                 + 1.01065 w_k1_IBM_s110
                                 + 0.968374 w_k1_INTC_s110
                                 + 1.07842 w_k1_KO_s110 + 1.55397 w_k1_MAT_s110
                                 + 1.07997 w_k1_MSFT_s110
                                 + 1.01011 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.92631 w_k1_ADM_s110
                                 + 1.2362 w_k1_AXP_s110
                                 + 0.872704 w_k1_DIS_s110
                                 + 0.993832 w_k1_FDX_s110
                                 + 0.692998 w_k1_IBM_s110
                                 + 1.20505 w_k1_INTC_s110
                                 + 0.836369 w_k1_KO_s110
                                 + 0.723283 w_k1_MAT_s110
                                 + 0.990756 w_k1_MSFT_s110
                                 + 0.941748 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.767896 w_k1_ADM_s110
                                 + 0.815105 w_k1_AXP_s110
                                 + 0.731533 w_k1_DIS_s110
                                 + 0.776095 w_k1_FDX_s110
                                 + 0.771892 w_k1_IBM_s110
                                 + 0.916388 w_k1_INTC_s110
                                 + 0.804614 w_k1_KO_s110
                                 + 0.86772 w_k1_MAT_s110
                                 + 0.868747 w_k1_MSFT_s110
                                 + 0.786061 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.24334 w_k1_ADM_s110
                                 + 1.22963 w_k1_AXP_s110 + 1.2268 w_k1_DIS_s110
                                 + 1.14091 w_k1_FDX_s110
                                 + 1.13113 w_k1_IBM_s110
                                 + 1.15561 w_k1_INTC_s110
                                 + 1.11803 w_k1_KO_s110 + 1.30019 w_k1_MAT_s110
                                 + 1.32789 w_k1_MSFT_s110
                                 + 1.15138 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.28226 w_k1_ADM_s111
                                 + 1.30041 w_k1_AXP_s111
                                 + 1.21921 w_k1_DIS_s111
                                 + 1.24133 w_k1_FDX_s111 + 1.0223 w_k1_IBM_s111
                                 + 1.23999 w_k1_INTC_s111
                                 + 1.10061 w_k1_KO_s111 + 1.04626 w_k1_MAT_s111
                                 + 1.23653 w_k1_MSFT_s111
                                 + 1.19287 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.00612 w_k1_ADM_s111
                                 + 0.926673 w_k1_AXP_s111
                                 + 1.33987 w_k1_DIS_s111 + 1.1047 w_k1_FDX_s111
                                 + 0.805528 w_k1_IBM_s111
                                 + 1.17578 w_k1_INTC_s111 + 1.0477 w_k1_KO_s111
                                 + 0.744451 w_k1_MAT_s111
                                 + 1.2771 w_k1_MSFT_s111 + 1.05509 w_k1_PG_s111
                                 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.12692 w_k1_ADM_s111
                                 + 1.5671 w_k1_AXP_s111 + 1.14914 w_k1_DIS_s111
                                 + 1.13798 w_k1_FDX_s111
                                 + 1.50868 w_k1_IBM_s111
                                 + 1.13479 w_k1_INTC_s111
                                 + 1.31194 w_k1_KO_s111 + 2.08149 w_k1_MAT_s111
                                 + 1.10824 w_k1_MSFT_s111
                                 + 1.27244 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.06592 w_k1_ADM_s111
                                 + 1.09279 w_k1_AXP_s111
                                 + 1.03356 w_k1_DIS_s111
                                 + 1.02489 w_k1_FDX_s111
                                 + 1.29872 w_k1_IBM_s111
                                 + 1.28021 w_k1_INTC_s111
                                 + 1.07609 w_k1_KO_s111 + 1.29678 w_k1_MAT_s111
                                 + 1.09715 w_k1_MSFT_s111
                                 + 1.15503 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.09998 w_k1_ADM_s111
                                 + 1.28912 w_k1_AXP_s111
                                 + 1.57998 w_k1_DIS_s111
                                 + 1.22434 w_k1_FDX_s111
                                 + 1.49285 w_k1_IBM_s111
                                 + 1.45461 w_k1_INTC_s111
                                 + 1.45727 w_k1_KO_s111 + 1.12365 w_k1_MAT_s111
                                 + 1.24043 w_k1_MSFT_s111
                                 + 1.25499 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.46819 w_k1_ADM_s111
                                 + 1.50158 w_k1_AXP_s111
                                 + 1.65542 w_k1_DIS_s111
                                 + 1.55343 w_k1_FDX_s111
                                 + 0.816267 w_k1_IBM_s111
                                 + 1.73099 w_k1_INTC_s111
                                 + 1.21959 w_k1_KO_s111
                                 + 0.858519 w_k1_MAT_s111
                                 + 1.39715 w_k1_MSFT_s111
                                 + 1.25677 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.945801 w_k1_ADM_s111
                                 + 0.939643 w_k1_AXP_s111
                                 + 0.860565 w_k1_DIS_s111
                                 + 0.995757 w_k1_FDX_s111
                                 + 0.708595 w_k1_IBM_s111
                                 + 1.06315 w_k1_INTC_s111
                                 + 0.870017 w_k1_KO_s111
                                 + 0.933539 w_k1_MAT_s111
                                 + 0.93147 w_k1_MSFT_s111
                                 + 0.958435 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.770336 w_k1_ADM_s111
                                 + 0.889434 w_k1_AXP_s111
                                 + 0.724676 w_k1_DIS_s111
                                 + 0.801647 w_k1_FDX_s111
                                 + 1.14507 w_k1_IBM_s111
                                 + 1.01952 w_k1_INTC_s111
                                 + 1.09876 w_k1_KO_s111 + 1.14372 w_k1_MAT_s111
                                 + 0.83836 w_k1_MSFT_s111
                                 + 0.969232 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.06916 w_k1_ADM_s112
                                 + 0.95096 w_k1_AXP_s112
                                 + 0.944733 w_k1_DIS_s112
                                 + 0.912631 w_k1_FDX_s112
                                 + 1.13525 w_k1_IBM_s112
                                 + 0.980618 w_k1_INTC_s112
                                 + 1.26154 w_k1_KO_s112 + 1.28329 w_k1_MAT_s112
                                 + 1.05777 w_k1_MSFT_s112
                                 + 1.04373 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.809386 w_k1_ADM_s112
                                 + 0.599768 w_k1_AXP_s112
                                 + 0.620474 w_k1_DIS_s112
                                 + 0.720437 w_k1_FDX_s112
                                 + 0.845265 w_k1_IBM_s112
                                 + 0.862798 w_k1_INTC_s112
                                 + 0.913976 w_k1_KO_s112
                                 + 0.938177 w_k1_MAT_s112
                                 + 0.919155 w_k1_MSFT_s112
                                 + 0.755834 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.0532 w_k1_ADM_s112
                                 + 0.61194 w_k1_AXP_s112
                                 + 0.758989 w_k1_DIS_s112
                                 + 0.828306 w_k1_FDX_s112
                                 + 0.68893 w_k1_IBM_s112
                                 + 0.759707 w_k1_INTC_s112
                                 + 0.789007 w_k1_KO_s112
                                 + 0.815413 w_k1_MAT_s112
                                 + 0.826091 w_k1_MSFT_s112
                                 + 0.800098 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.2112 w_k1_ADM_s112
                                 + 0.807664 w_k1_AXP_s112
                                 + 1.0393 w_k1_DIS_s112 + 1.11474 w_k1_FDX_s112
                                 + 0.810114 w_k1_IBM_s112
                                 + 1.12027 w_k1_INTC_s112
                                 + 1.07138 w_k1_KO_s112 + 1.24688 w_k1_MAT_s112
                                 + 1.14485 w_k1_MSFT_s112
                                 + 1.02839 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.10161 w_k1_ADM_s112
                                 + 1.15683 w_k1_AXP_s112
                                 + 1.16594 w_k1_DIS_s112
                                 + 1.20855 w_k1_FDX_s112
                                 + 0.97899 w_k1_IBM_s112
                                 + 0.981151 w_k1_INTC_s112
                                 + 1.2391 w_k1_KO_s112 + 1.30725 w_k1_MAT_s112
                                 + 1.20712 w_k1_MSFT_s112
                                 + 1.00926 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.25041 w_k1_ADM_s112
                                 + 1.41509 w_k1_AXP_s112
                                 + 1.41167 w_k1_DIS_s112
                                 + 1.23978 w_k1_FDX_s112
                                 + 1.51613 w_k1_IBM_s112
                                 + 1.16014 w_k1_INTC_s112
                                 + 1.62367 w_k1_KO_s112 + 2.36556 w_k1_MAT_s112
                                 + 1.2397 w_k1_MSFT_s112 + 1.16973 w_k1_PG_s112
                                 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.25926 w_k1_ADM_s112
                                 + 1.01029 w_k1_AXP_s112
                                 + 1.47088 w_k1_DIS_s112
                                 + 1.19223 w_k1_FDX_s112
                                 + 1.28223 w_k1_IBM_s112
                                 + 1.05396 w_k1_INTC_s112
                                 + 1.49312 w_k1_KO_s112 + 1.7503 w_k1_MAT_s112
                                 + 1.28459 w_k1_MSFT_s112
                                 + 1.14772 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.20643 w_k1_ADM_s112
                                 + 1.23367 w_k1_AXP_s112
                                 + 0.81191 w_k1_DIS_s112
                                 + 1.09929 w_k1_FDX_s112
                                 + 1.03478 w_k1_IBM_s112
                                 + 0.974871 w_k1_INTC_s112
                                 + 1.10445 w_k1_KO_s112 + 1.96289 w_k1_MAT_s112
                                 + 1.05013 w_k1_MSFT_s112 + 1.1384 w_k1_PG_s112
                                 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.06641 w_k1_ADM_s113
                                 + 1.36393 w_k1_AXP_s113
                                 + 1.16003 w_k1_DIS_s113
                                 + 1.20926 w_k1_FDX_s113
                                 + 0.904953 w_k1_IBM_s113
                                 + 1.04083 w_k1_INTC_s113
                                 + 0.99291 w_k1_KO_s113
                                 + 0.921038 w_k1_MAT_s113
                                 + 1.03991 w_k1_MSFT_s113
                                 + 0.976471 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.944126 w_k1_ADM_s113
                                 + 1.16513 w_k1_AXP_s113 + 1.2377 w_k1_DIS_s113
                                 + 1.18562 w_k1_FDX_s113
                                 + 0.971201 w_k1_IBM_s113
                                 + 1.34908 w_k1_INTC_s113
                                 + 1.15004 w_k1_KO_s113
                                 + 0.954962 w_k1_MAT_s113
                                 + 1.19321 w_k1_MSFT_s113
                                 + 1.18702 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.700108 w_k1_ADM_s113
                                 + 1.16781 w_k1_AXP_s113
                                 + 0.650823 w_k1_DIS_s113
                                 + 0.885827 w_k1_FDX_s113
                                 + 0.723101 w_k1_IBM_s113
                                 + 0.941376 w_k1_INTC_s113
                                 + 0.668002 w_k1_KO_s113 + 1.2398 w_k1_MAT_s113
                                 + 0.756316 w_k1_MSFT_s113
                                 + 0.917892 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.715748 w_k1_ADM_s113
                                 + 0.892012 w_k1_AXP_s113
                                 + 0.914726 w_k1_DIS_s113
                                 + 0.982009 w_k1_FDX_s113
                                 + 0.759122 w_k1_IBM_s113
                                 + 1.05976 w_k1_INTC_s113
                                 + 0.933494 w_k1_KO_s113
                                 + 0.996142 w_k1_MAT_s113
                                 + 0.961513 w_k1_MSFT_s113
                                 + 0.942352 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.639515 w_k1_ADM_s113
                                 + 0.679278 w_k1_AXP_s113
                                 + 0.739187 w_k1_DIS_s113
                                 + 0.774917 w_k1_FDX_s113
                                 + 0.934902 w_k1_IBM_s113
                                 + 0.902686 w_k1_INTC_s113
                                 + 0.848743 w_k1_KO_s113
                                 + 0.871319 w_k1_MAT_s113
                                 + 0.939192 w_k1_MSFT_s113
                                 + 0.891413 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.852624 w_k1_ADM_s113
                                 + 0.753007 w_k1_AXP_s113
                                 + 0.568239 w_k1_DIS_s113
                                 + 0.75166 w_k1_FDX_s113
                                 + 0.67657 w_k1_IBM_s113
                                 + 0.800539 w_k1_INTC_s113
                                 + 0.630015 w_k1_KO_s113
                                 + 0.588784 w_k1_MAT_s113
                                 + 0.681464 w_k1_MSFT_s113
                                 + 0.782151 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.775803 w_k1_ADM_s113
                                 + 1.04591 w_k1_AXP_s113
                                 + 0.95869 w_k1_DIS_s113
                                 + 0.892934 w_k1_FDX_s113
                                 + 1.19249 w_k1_IBM_s113
                                 + 0.97955 w_k1_INTC_s113
                                 + 0.918299 w_k1_KO_s113
                                 + 1.40775 w_k1_MAT_s113
                                 + 0.809539 w_k1_MSFT_s113
                                 + 1.05213 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.01676 w_k1_ADM_s113
                                 + 1.1946 w_k1_AXP_s113 + 1.37782 w_k1_DIS_s113
                                 + 1.25918 w_k1_FDX_s113
                                 + 1.51311 w_k1_IBM_s113
                                 + 1.30467 w_k1_INTC_s113
                                 + 1.53141 w_k1_KO_s113 + 1.42739 w_k1_MAT_s113
                                 + 1.2372 w_k1_MSFT_s113 + 1.21966 w_k1_PG_s113
                                 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.604569 w_k1_ADM_s114
                                 + 0.558583 w_k1_AXP_s114
                                 + 0.591332 w_k1_DIS_s114
                                 + 0.686878 w_k1_FDX_s114
                                 + 0.749785 w_k1_IBM_s114
                                 + 0.891637 w_k1_INTC_s114
                                 + 0.850757 w_k1_KO_s114 + 0.4955 w_k1_MAT_s114
                                 + 0.750837 w_k1_MSFT_s114
                                 + 0.841909 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.985901 w_k1_ADM_s114
                                 + 1.20877 w_k1_AXP_s114
                                 + 1.12086 w_k1_DIS_s114
                                 + 1.03162 w_k1_FDX_s114 + 1.448 w_k1_IBM_s114
                                 + 1.0167 w_k1_INTC_s114 + 1.1517 w_k1_KO_s114
                                 + 1.07158 w_k1_MAT_s114
                                 + 0.926465 w_k1_MSFT_s114
                                 + 0.990259 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.99313 w_k1_ADM_s114
                                 + 1.13637 w_k1_AXP_s114
                                 + 0.89736 w_k1_DIS_s114
                                 + 1.12555 w_k1_FDX_s114
                                 + 0.890921 w_k1_IBM_s114
                                 + 1.05775 w_k1_INTC_s114 + 0.98 w_k1_KO_s114
                                 + 0.951487 w_k1_MAT_s114
                                 + 0.894435 w_k1_MSFT_s114
                                 + 1.10402 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.960039 w_k1_ADM_s114
                                 + 0.875875 w_k1_AXP_s114
                                 + 1.17084 w_k1_DIS_s114
                                 + 1.05866 w_k1_FDX_s114 + 1.017 w_k1_IBM_s114
                                 + 1.20379 w_k1_INTC_s114
                                 + 1.03469 w_k1_KO_s114
                                 + 0.702406 w_k1_MAT_s114
                                 + 1.1566 w_k1_MSFT_s114 + 1.07911 w_k1_PG_s114
                                 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.962375 w_k1_ADM_s114
                                 + 0.889845 w_k1_AXP_s114
                                 + 0.828405 w_k1_DIS_s114
                                 + 0.883902 w_k1_FDX_s114
                                 + 0.750233 w_k1_IBM_s114
                                 + 0.888613 w_k1_INTC_s114
                                 + 1.02797 w_k1_KO_s114
                                 + 0.684813 w_k1_MAT_s114
                                 + 0.81667 w_k1_MSFT_s114
                                 + 1.02346 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.39528 w_k1_ADM_s114
                                 + 1.63565 w_k1_AXP_s114
                                 + 1.72144 w_k1_DIS_s114
                                 + 1.57424 w_k1_FDX_s114
                                 + 1.13265 w_k1_IBM_s114
                                 + 1.55962 w_k1_INTC_s114
                                 + 1.40478 w_k1_KO_s114
                                 + 0.856696 w_k1_MAT_s114
                                 + 1.27755 w_k1_MSFT_s114
                                 + 1.24958 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.98861 w_k1_ADM_s114
                                 + 0.664219 w_k1_AXP_s114
                                 + 0.994153 w_k1_DIS_s114
                                 + 0.979508 w_k1_FDX_s114
                                 + 0.614552 w_k1_IBM_s114
                                 + 1.08495 w_k1_INTC_s114
                                 + 0.880737 w_k1_KO_s114
                                 + 0.422197 w_k1_MAT_s114
                                 + 1.08238 w_k1_MSFT_s114
                                 + 0.880666 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.922225 w_k1_ADM_s114
                                 + 0.741272 w_k1_AXP_s114
                                 + 0.699417 w_k1_DIS_s114
                                 + 0.782476 w_k1_FDX_s114
                                 + 1.11916 w_k1_IBM_s114
                                 + 0.816695 w_k1_INTC_s114
                                 + 0.995224 w_k1_KO_s114
                                 + 0.909289 w_k1_MAT_s114
                                 + 0.844986 w_k1_MSFT_s114
                                 + 0.95523 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.06998 w_k1_ADM_s115
                                 + 0.996847 w_k1_AXP_s115
                                 + 1.16424 w_k1_DIS_s115
                                 + 0.952185 w_k1_FDX_s115
                                 + 1.51836 w_k1_IBM_s115
                                 + 1.17486 w_k1_INTC_s115
                                 + 1.16502 w_k1_KO_s115 + 1.03536 w_k1_MAT_s115
                                 + 0.929834 w_k1_MSFT_s115
                                 + 1.02363 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.18305 w_k1_ADM_s115
                                 + 1.26129 w_k1_AXP_s115
                                 + 1.43643 w_k1_DIS_s115
                                 + 1.30675 w_k1_FDX_s115
                                 + 0.954243 w_k1_IBM_s115
                                 + 1.28596 w_k1_INTC_s115
                                 + 1.13955 w_k1_KO_s115 + 1.04098 w_k1_MAT_s115
                                 + 1.18985 w_k1_MSFT_s115
                                 + 1.06733 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.944196 w_k1_ADM_s115
                                 + 0.99427 w_k1_AXP_s115
                                 + 0.895349 w_k1_DIS_s115
                                 + 0.977405 w_k1_FDX_s115
                                 + 0.791636 w_k1_IBM_s115
                                 + 1.19089 w_k1_INTC_s115
                                 + 0.847807 w_k1_KO_s115
                                 + 0.812178 w_k1_MAT_s115
                                 + 0.938477 w_k1_MSFT_s115
                                 + 0.895505 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.993993 w_k1_ADM_s115
                                 + 0.816763 w_k1_AXP_s115
                                 + 0.784474 w_k1_DIS_s115
                                 + 0.839007 w_k1_FDX_s115
                                 + 1.10575 w_k1_IBM_s115
                                 + 0.961235 w_k1_INTC_s115
                                 + 1.03023 w_k1_KO_s115 + 1.35711 w_k1_MAT_s115
                                 + 0.891489 w_k1_MSFT_s115
                                 + 0.917198 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.00522 w_k1_ADM_s115
                                 + 1.07366 w_k1_AXP_s115
                                 + 1.08795 w_k1_DIS_s115
                                 + 1.23296 w_k1_FDX_s115
                                 + 0.73154 w_k1_IBM_s115
                                 + 1.31201 w_k1_INTC_s115
                                 + 0.839145 w_k1_KO_s115
                                 + 0.655413 w_k1_MAT_s115
                                 + 1.13503 w_k1_MSFT_s115
                                 + 0.902083 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.15636 w_k1_ADM_s115
                                 + 1.23708 w_k1_AXP_s115
                                 + 1.09609 w_k1_DIS_s115
                                 + 1.08871 w_k1_FDX_s115 + 1.0721 w_k1_IBM_s115
                                 + 1.27012 w_k1_INTC_s115
                                 + 1.01735 w_k1_KO_s115 + 1.30612 w_k1_MAT_s115
                                 + 1.07526 w_k1_MSFT_s115
                                 + 0.974263 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.31225 w_k1_ADM_s115
                                 + 1.22784 w_k1_AXP_s115 + 1.5805 w_k1_DIS_s115
                                 + 1.31603 w_k1_FDX_s115
                                 + 1.56089 w_k1_IBM_s115
                                 + 1.48217 w_k1_INTC_s115
                                 + 1.41172 w_k1_KO_s115 + 1.33647 w_k1_MAT_s115
                                 + 1.31939 w_k1_MSFT_s115
                                 + 1.15924 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.680932 w_k1_ADM_s115
                                 + 0.81199 w_k1_AXP_s115
                                 + 0.661203 w_k1_DIS_s115
                                 + 0.773953 w_k1_FDX_s115
                                 + 0.774841 w_k1_IBM_s115
                                 + 0.92904 w_k1_INTC_s115
                                 + 0.688764 w_k1_KO_s115
                                 + 0.852486 w_k1_MAT_s115
                                 + 0.76153 w_k1_MSFT_s115
                                 + 0.729105 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.10735 w_k1_ADM_s116
                                 + 1.43025 w_k1_AXP_s116 + 1.4392 w_k1_DIS_s116
                                 + 1.30513 w_k1_FDX_s116
                                 + 1.23164 w_k1_IBM_s116
                                 + 1.21757 w_k1_INTC_s116
                                 + 1.10325 w_k1_KO_s116 + 1.3829 w_k1_MAT_s116
                                 + 1.2101 w_k1_MSFT_s116 + 1.09735 w_k1_PG_s116
                                 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.25676 w_k1_ADM_s116
                                 + 1.03781 w_k1_AXP_s116
                                 + 0.913661 w_k1_DIS_s116
                                 + 1.06425 w_k1_FDX_s116
                                 + 0.695181 w_k1_IBM_s116
                                 + 1.14677 w_k1_INTC_s116
                                 + 0.895058 w_k1_KO_s116
                                 + 0.935918 w_k1_MAT_s116
                                 + 1.18925 w_k1_MSFT_s116
                                 + 0.962554 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.921367 w_k1_ADM_s116
                                 + 0.917428 w_k1_AXP_s116
                                 + 0.71922 w_k1_DIS_s116
                                 + 0.867447 w_k1_FDX_s116
                                 + 0.629805 w_k1_IBM_s116
                                 + 0.951541 w_k1_INTC_s116
                                 + 0.757806 w_k1_KO_s116
                                 + 0.743256 w_k1_MAT_s116
                                 + 0.888096 w_k1_MSFT_s116
                                 + 0.903101 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.801661 w_k1_ADM_s116
                                 + 0.754904 w_k1_AXP_s116
                                 + 0.658651 w_k1_DIS_s116
                                 + 0.767285 w_k1_FDX_s116
                                 + 0.78685 w_k1_IBM_s116
                                 + 1.00225 w_k1_INTC_s116
                                 + 0.796057 w_k1_KO_s116
                                 + 0.734153 w_k1_MAT_s116
                                 + 0.987578 w_k1_MSFT_s116
                                 + 0.825949 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.1067 w_k1_ADM_s116
                                 + 1.36657 w_k1_AXP_s116 + 1.0643 w_k1_DIS_s116
                                 + 1.15039 w_k1_FDX_s116
                                 + 0.795403 w_k1_IBM_s116
                                 + 1.04195 w_k1_INTC_s116
                                 + 0.852682 w_k1_KO_s116
                                 + 1.08824 w_k1_MAT_s116
                                 + 1.10539 w_k1_MSFT_s116
                                 + 0.874366 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.999247 w_k1_ADM_s116
                                 + 0.961569 w_k1_AXP_s116
                                 + 1.31371 w_k1_DIS_s116
                                 + 0.983018 w_k1_FDX_s116
                                 + 1.46814 w_k1_IBM_s116
                                 + 1.03242 w_k1_INTC_s116
                                 + 1.40636 w_k1_KO_s116 + 1.37683 w_k1_MAT_s116
                                 + 1.237 w_k1_MSFT_s116 + 0.980991 w_k1_PG_s116
                                 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.885563 w_k1_ADM_s116
                                 + 0.901374 w_k1_AXP_s116
                                 + 0.929436 w_k1_DIS_s116
                                 + 0.944088 w_k1_FDX_s116
                                 + 1.03741 w_k1_IBM_s116
                                 + 1.05867 w_k1_INTC_s116
                                 + 1.07877 w_k1_KO_s116 + 1.20353 w_k1_MAT_s116
                                 + 1.06044 w_k1_MSFT_s116
                                 + 1.04019 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.849508 w_k1_ADM_s116
                                 + 0.863776 w_k1_AXP_s116
                                 + 1.06787 w_k1_DIS_s116
                                 + 1.01903 w_k1_FDX_s116
                                 + 0.868973 w_k1_IBM_s116
                                 + 1.0503 w_k1_INTC_s116
                                 + 0.979584 w_k1_KO_s116
                                 + 0.861716 w_k1_MAT_s116
                                 + 1.0779 w_k1_MSFT_s116 + 0.96256 w_k1_PG_s116
                                 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.85331 w_k1_ADM_s117
                                 + 0.964257 w_k1_AXP_s117
                                 + 1.20651 w_k1_DIS_s117
                                 + 1.14075 w_k1_FDX_s117
                                 + 0.850667 w_k1_IBM_s117
                                 + 1.08327 w_k1_INTC_s117
                                 + 0.992388 w_k1_KO_s117
                                 + 0.868754 w_k1_MAT_s117
                                 + 1.30296 w_k1_MSFT_s117
                                 + 1.17295 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.09982 w_k1_ADM_s117
                                 + 1.38517 w_k1_AXP_s117
                                 + 1.54018 w_k1_DIS_s117
                                 + 1.34361 w_k1_FDX_s117
                                 + 1.43064 w_k1_IBM_s117
                                 + 1.37384 w_k1_INTC_s117
                                 + 1.39297 w_k1_KO_s117 + 1.36213 w_k1_MAT_s117
                                 + 1.52539 w_k1_MSFT_s117
                                 + 1.32296 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.944695 w_k1_ADM_s117
                                 + 1.50502 w_k1_AXP_s117 + 1.2126 w_k1_DIS_s117
                                 + 1.24359 w_k1_FDX_s117
                                 + 1.08199 w_k1_IBM_s117
                                 + 1.27732 w_k1_INTC_s117
                                 + 1.11692 w_k1_KO_s117 + 1.38625 w_k1_MAT_s117
                                 + 1.3444 w_k1_MSFT_s117 + 1.15407 w_k1_PG_s117
                                 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.832765 w_k1_ADM_s117
                                 + 0.856175 w_k1_AXP_s117
                                 + 0.920369 w_k1_DIS_s117
                                 + 0.926035 w_k1_FDX_s117
                                 + 0.827692 w_k1_IBM_s117
                                 + 1.19006 w_k1_INTC_s117
                                 + 1.00592 w_k1_KO_s117
                                 + 0.680842 w_k1_MAT_s117
                                 + 1.16055 w_k1_MSFT_s117
                                 + 0.947338 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.494702 w_k1_ADM_s117
                                 + 0.89976 w_k1_AXP_s117
                                 + 0.724558 w_k1_DIS_s117
                                 + 0.745428 w_k1_FDX_s117
                                 + 1.26846 w_k1_IBM_s117
                                 + 1.0521 w_k1_INTC_s117
                                 + 0.937441 w_k1_KO_s117
                                 + 1.21884 w_k1_MAT_s117
                                 + 1.00891 w_k1_MSFT_s117
                                 + 0.94747 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.581836 w_k1_ADM_s117
                                 + 0.693546 w_k1_AXP_s117
                                 + 0.554788 w_k1_DIS_s117
                                 + 0.696424 w_k1_FDX_s117
                                 + 0.810904 w_k1_IBM_s117
                                 + 0.835737 w_k1_INTC_s117
                                 + 0.71445 w_k1_KO_s117
                                 + 0.786672 w_k1_MAT_s117
                                 + 0.818439 w_k1_MSFT_s117
                                 + 0.791418 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.718272 w_k1_ADM_s117
                                 + 1.09486 w_k1_AXP_s117
                                 + 1.02132 w_k1_DIS_s117
                                 + 0.862915 w_k1_FDX_s117
                                 + 1.7873 w_k1_IBM_s117
                                 + 1.21119 w_k1_INTC_s117
                                 + 1.26151 w_k1_KO_s117 + 1.77373 w_k1_MAT_s117
                                 + 1.12284 w_k1_MSFT_s117
                                 + 1.07075 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.23215 w_k1_ADM_s117
                                 + 1.56381 w_k1_AXP_s117
                                 + 1.80572 w_k1_DIS_s117
                                 + 1.50449 w_k1_FDX_s117
                                 + 0.883756 w_k1_IBM_s117
                                 + 1.62569 w_k1_INTC_s117
                                 + 1.20144 w_k1_KO_s117 + 1.21256 w_k1_MAT_s117
                                 + 1.67542 w_k1_MSFT_s117
                                 + 1.24286 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.046348273256355 w_k1_ADM_s000
                                 + 0.0191114891984057 w_k1_ADM_s001
                                 + 0.0362306580995825 w_k1_ADM_s002
                                 + 0.0216725401711867 w_k1_ADM_s003
                                 + 0.0458980969425489 w_k1_ADM_s004
                                 + 0.0424531560949411 w_k1_ADM_s005
                                 + 0.064141450291788 w_k1_ADM_s006
                                 + 0.0538805482772073 w_k1_ADM_s007
                                 + 0.0296753822356895 w_k1_ADM_s010
                                 + 0.012452440712285 w_k1_ADM_s011
                                 + 0.00815379788414996 w_k1_ADM_s012
                                 + 0.00972774790952845 w_k1_ADM_s013
                                 + 0.00811073558855557 w_k1_ADM_s014
                                 + 0.0174083939195366 w_k1_ADM_s015
                                 + 0.00676519424752059 w_k1_ADM_s016
                                 + 0.0154003379705762 w_k1_ADM_s017
                                 + 0.0388434855474051 w_k1_ADM_s100
                                 + 0.0228141095643206 w_k1_ADM_s101
                                 + 0.0352250038665022 w_k1_ADM_s102
                                 + 0.0226869306496141 w_k1_ADM_s103
                                 + 0.0242434045790166 w_k1_ADM_s104
                                 + 0.0197409812837907 w_k1_ADM_s105
                                 + 0.0261993197467156 w_k1_ADM_s106
                                 + 0.0335348304955543 w_k1_ADM_s107
                                 + 0.0215660416715582 w_k1_ADM_s110
                                 + 0.0733550775053887 w_k1_ADM_s111
                                 + 0.0196799588542165 w_k1_ADM_s112
                                 + 0.030689889763555 w_k1_ADM_s113
                                 + 0.0239649168912816 w_k1_ADM_s114
                                 + 0.0744691437747207 w_k1_ADM_s115
                                 + 0.0387538886460927 w_k1_ADM_s116
                                 + 0.0539409053265643 w_k1_ADM_s117
                                 + 0.0504767960406756 w_k1_AXP_s000
                                 + 0.0193042420847516 w_k1_AXP_s001
                                 + 0.0402677596125039 w_k1_AXP_s002
                                 + 0.0202304241416795 w_k1_AXP_s003
                                 + 0.046173380494453 w_k1_AXP_s004
                                 + 0.0466290314845313 w_k1_AXP_s005
                                 + 0.0634474654672881 w_k1_AXP_s006
                                 + 0.049412278042864 w_k1_AXP_s007
                                 + 0.0312453001867036 w_k1_AXP_s010
                                 + 0.0121495698976644 w_k1_AXP_s011
                                 + 0.00867649526492577 w_k1_AXP_s012
                                 + 0.0102604962837981 w_k1_AXP_s013
                                 + 0.00920180304597671 w_k1_AXP_s014
                                 + 0.0161435868547835 w_k1_AXP_s015
                                 + 0.0060020750810512 w_k1_AXP_s016
                                 + 0.0139476900063332 w_k1_AXP_s017
                                 + 0.046505383953068 w_k1_AXP_s100
                                 + 0.0217596246000711 w_k1_AXP_s101
                                 + 0.036799911926493 w_k1_AXP_s102
                                 + 0.0246388750224613 w_k1_AXP_s103
                                 + 0.0255809501599704 w_k1_AXP_s104
                                 + 0.0187620594556916 w_k1_AXP_s105
                                 + 0.0280779041090108 w_k1_AXP_s106
                                 + 0.0339410153586889 w_k1_AXP_s107
                                 + 0.0232690575815978 w_k1_AXP_s110
                                 + 0.0783523412950497 w_k1_AXP_s111
                                 + 0.0171641122094326 w_k1_AXP_s112
                                 + 0.0380158219794472 w_k1_AXP_s113
                                 + 0.0240605063030678 w_k1_AXP_s114
                                 + 0.0763531285794865 w_k1_AXP_s115
                                 + 0.0407128135766321 w_k1_AXP_s116
                                 + 0.0689731671207313 w_k1_AXP_s117
                                 + 0.0484744611881796 w_k1_DIS_s000
                                 + 0.0190543437582567 w_k1_DIS_s001
                                 + 0.0358365565343498 w_k1_DIS_s002
                                 + 0.0222248602995936 w_k1_DIS_s003
                                 + 0.0484986498213914 w_k1_DIS_s004
                                 + 0.0451825111885607 w_k1_DIS_s005
                                 + 0.06751496802788 w_k1_DIS_s006
                                 + 0.0470562656920434 w_k1_DIS_s007
                                 + 0.0256939524883691 w_k1_DIS_s010
                                 + 0.0119422461832536 w_k1_DIS_s011
                                 + 0.00847492338790392 w_k1_DIS_s012
                                 + 0.00951987560968366 w_k1_DIS_s013
                                 + 0.00894272808760519 w_k1_DIS_s014
                                 + 0.0174232662296791 w_k1_DIS_s015
                                 + 0.00674818475771219 w_k1_DIS_s016
                                 + 0.0140701399464017 w_k1_DIS_s017
                                 + 0.044918289622288 w_k1_DIS_s100
                                 + 0.0226826589757785 w_k1_DIS_s101
                                 + 0.039765799374528 w_k1_DIS_s102
                                 + 0.0259183998681298 w_k1_DIS_s103
                                 + 0.0258241916236121 w_k1_DIS_s104
                                 + 0.0178117622791736 w_k1_DIS_s105
                                 + 0.0321013186104336 w_k1_DIS_s106
                                 + 0.0335898828881558 w_k1_DIS_s107
                                 + 0.0218692636667737 w_k1_DIS_s110
                                 + 0.0749980799918044 w_k1_DIS_s111
                                 + 0.0186315282407405 w_k1_DIS_s112
                                 + 0.0359886321943196 w_k1_DIS_s113
                                 + 0.0255900118702939 w_k1_DIS_s114
                                 + 0.077347322638718 w_k1_DIS_s115
                                 + 0.0415358695811535 w_k1_DIS_s116
                                 + 0.0687770845468499 w_k1_DIS_s117
                                 + 0.0477035912457263 w_k1_FDX_s000
                                 + 0.0203100645387057 w_k1_FDX_s001
                                 + 0.0364938331272595 w_k1_FDX_s002
                                 + 0.0220537120353422 w_k1_FDX_s003
                                 + 0.0449952607106595 w_k1_FDX_s004
                                 + 0.0431680212270415 w_k1_FDX_s005
                                 + 0.0617150939168039 w_k1_FDX_s006
                                 + 0.0485096285039871 w_k1_FDX_s007
                                 + 0.0260840426921399 w_k1_FDX_s010
                                 + 0.0111188285415958 w_k1_FDX_s011
                                 + 0.00826876057206336 w_k1_FDX_s012
                                 + 0.00979899628875435 w_k1_FDX_s013
                                 + 0.00907262303004218 w_k1_FDX_s014
                                 + 0.017350970157765 w_k1_FDX_s015
                                 + 0.00652758023395689 w_k1_FDX_s016
                                 + 0.014096549606675 w_k1_FDX_s017
                                 + 0.041802057625839 w_k1_FDX_s100
                                 + 0.0232970930415071 w_k1_FDX_s101
                                 + 0.036204823758828 w_k1_FDX_s102
                                 + 0.0251731221148077 w_k1_FDX_s103
                                 + 0.0230550354092583 w_k1_FDX_s104
                                 + 0.0178437448978825 w_k1_FDX_s105
                                 + 0.0290280279951085 w_k1_FDX_s106
                                 + 0.0333049224462798 w_k1_FDX_s107
                                 + 0.0212033846273701 w_k1_FDX_s110
                                 + 0.0757151003951928 w_k1_FDX_s111
                                 + 0.0183222219551332 w_k1_FDX_s112
                                 + 0.0364820541108489 w_k1_FDX_s113
                                 + 0.0253167694436832 w_k1_FDX_s114
                                 + 0.0763394057300939 w_k1_FDX_s115
                                 + 0.0403675110708646 w_k1_FDX_s116
                                 + 0.0670981175428162 w_k1_FDX_s117
                                 + 0.0487811638198215 w_k1_IBM_s000
                                 + 0.0201838217621972 w_k1_IBM_s001
                                 + 0.0354313744931181 w_k1_IBM_s002
                                 + 0.0220639364040368 w_k1_IBM_s003
                                 + 0.0476904438338678 w_k1_IBM_s004
                                 + 0.0441178847976558 w_k1_IBM_s005
                                 + 0.0649380690662091 w_k1_IBM_s006
                                 + 0.0466534552470432 w_k1_IBM_s007
                                 + 0.0277450804867968 w_k1_IBM_s010
                                 + 0.0117585108507589 w_k1_IBM_s011
                                 + 0.00872577235475475 w_k1_IBM_s012
                                 + 0.0100647656433307 w_k1_IBM_s013
                                 + 0.00873780574684368 w_k1_IBM_s014
                                 + 0.0170396679667471 w_k1_IBM_s015
                                 + 0.00654162030831889 w_k1_IBM_s016
                                 + 0.0140912642354751 w_k1_IBM_s017
                                 + 0.0435295953531769 w_k1_IBM_s100
                                 + 0.0228926901508667 w_k1_IBM_s101
                                 + 0.0380232053701927 w_k1_IBM_s102
                                 + 0.0245551087342453 w_k1_IBM_s103
                                 + 0.0234590390233358 w_k1_IBM_s104
                                 + 0.0190042315886548 w_k1_IBM_s105
                                 + 0.0282818189485521 w_k1_IBM_s106
                                 + 0.0334541079112274 w_k1_IBM_s107
                                 + 0.0209849760770885 w_k1_IBM_s110
                                 + 0.0753228786129634 w_k1_IBM_s111
                                 + 0.0185379608837876 w_k1_IBM_s112
                                 + 0.0366360464187903 w_k1_IBM_s113
                                 + 0.0245081286387678 w_k1_IBM_s114
                                 + 0.0738026706141348 w_k1_IBM_s115
                                 + 0.040524588365774 w_k1_IBM_s116
                                 + 0.0712643869186952 w_k1_IBM_s117
                                 + 0.0507572981050276 w_k1_INTC_s000
                                 + 0.0178051574185711 w_k1_INTC_s001
                                 + 0.0373710568607285 w_k1_INTC_s002
                                 + 0.0245050205110457 w_k1_INTC_s003
                                 + 0.0426358531697711 w_k1_INTC_s004
                                 + 0.048579083911218 w_k1_INTC_s005
                                 + 0.061918224422903 w_k1_INTC_s006
                                 + 0.0476457259448903 w_k1_INTC_s007
                                 + 0.0240536953307533 w_k1_INTC_s010
                                 + 0.0130050413435366 w_k1_INTC_s011
                                 + 0.00811369667546411 w_k1_INTC_s012
                                 + 0.0118790796944311 w_k1_INTC_s013
                                 + 0.00940304702212143 w_k1_INTC_s014
                                 + 0.0160297598842199 w_k1_INTC_s015
                                 + 0.00722517150637592 w_k1_INTC_s016
                                 + 0.0121500819131766 w_k1_INTC_s017
                                 + 0.0424424790335687 w_k1_INTC_s100
                                 + 0.0222643484071725 w_k1_INTC_s101
                                 + 0.0394218635776635 w_k1_INTC_s102
                                 + 0.0253216241991695 w_k1_INTC_s103
                                 + 0.0216734919628411 w_k1_INTC_s104
                                 + 0.0178588689486387 w_k1_INTC_s105
                                 + 0.0329087397155877 w_k1_INTC_s106
                                 + 0.0320525654231959 w_k1_INTC_s107
                                 + 0.0231299105971941 w_k1_INTC_s110
                                 + 0.0857830380825937 w_k1_INTC_s111
                                 + 0.0174128174743092 w_k1_INTC_s112
                                 + 0.0384310710883303 w_k1_INTC_s113
                                 + 0.0265865329180735 w_k1_INTC_s114
                                 + 0.0867702747120004 w_k1_INTC_s115
                                 + 0.0427427031627515 w_k1_INTC_s116
                                 + 0.0785076084765385 w_k1_INTC_s117
                                 + 0.0477414946696968 w_k1_KO_s000
                                 + 0.0194927404635911 w_k1_KO_s001
                                 + 0.0379706036508844 w_k1_KO_s002
                                 + 0.0207400870722478 w_k1_KO_s003
                                 + 0.0426278473109922 w_k1_KO_s004
                                 + 0.0428170387659948 w_k1_KO_s005
                                 + 0.0610852305696844 w_k1_KO_s006
                                 + 0.0533268903109212 w_k1_KO_s007
                                 + 0.0265555424466073 w_k1_KO_s010
                                 + 0.0128314797726476 w_k1_KO_s011
                                 + 0.00863689824907125 w_k1_KO_s012
                                 + 0.01005996040317 w_k1_KO_s013
                                 + 0.00817935327102682 w_k1_KO_s014
                                 + 0.0172686540240221 w_k1_KO_s015
                                 + 0.00637122235565996 w_k1_KO_s016
                                 + 0.0125030313194036 w_k1_KO_s017
                                 + 0.0427490587459834 w_k1_KO_s100
                                 + 0.0228778554545464 w_k1_KO_s101
                                 + 0.0356502543180233 w_k1_KO_s102
                                 + 0.0240412139615507 w_k1_KO_s103
                                 + 0.0242393544203791 w_k1_KO_s104
                                 + 0.0192484291923029 w_k1_KO_s105
                                 + 0.0306367057441768 w_k1_KO_s106
                                 + 0.0330366948307011 w_k1_KO_s107
                                 + 0.0209791605280412 w_k1_KO_s110
                                 + 0.0780282082098444 w_k1_KO_s111
                                 + 0.0213322088534478 w_k1_KO_s112
                                 + 0.0359004512555616 w_k1_KO_s113
                                 + 0.0255443184036331 w_k1_KO_s114
                                 + 0.0716294255771731 w_k1_KO_s115
                                 + 0.0413843349233885 w_k1_KO_s116
                                 + 0.0694678675265292 w_k1_KO_s117
                                 + 0.0566613104554499 w_k1_MAT_s000
                                 + 0.0187161438321883 w_k1_MAT_s001
                                 + 0.0373290473761666 w_k1_MAT_s002
                                 + 0.0222698800033378 w_k1_MAT_s003
                                 + 0.04929038401309 w_k1_MAT_s004
                                 + 0.0367347851189659 w_k1_MAT_s005
                                 + 0.0563038283343053 w_k1_MAT_s006
                                 + 0.0561024925617917 w_k1_MAT_s007
                                 + 0.028860747278285 w_k1_MAT_s010
                                 + 0.012323728825808 w_k1_MAT_s011
                                 + 0.00851647725796595 w_k1_MAT_s012
                                 + 0.010393381855676 w_k1_MAT_s013
                                 + 0.00752034519452389 w_k1_MAT_s014
                                 + 0.018486160340795 w_k1_MAT_s015
                                 + 0.00532080301499754 w_k1_MAT_s016
                                 + 0.0136453371233971 w_k1_MAT_s017
                                 + 0.0473166695285874 w_k1_MAT_s100
                                 + 0.0210928944674475 w_k1_MAT_s101
                                 + 0.0350560592157921 w_k1_MAT_s102
                                 + 0.0204311568642481 w_k1_MAT_s103
                                 + 0.0205436332195528 w_k1_MAT_s104
                                 + 0.0223348703659203 w_k1_MAT_s105
                                 + 0.0320556391508207 w_k1_MAT_s106
                                 + 0.033076338694355 w_k1_MAT_s107
                                 + 0.0237501336726356 w_k1_MAT_s110
                                 + 0.0806160465907647 w_k1_MAT_s111
                                 + 0.0255198337529739 w_k1_MAT_s112
                                 + 0.0401621892367369 w_k1_MAT_s113
                                 + 0.019054970123691 w_k1_MAT_s114
                                 + 0.0730889579256826 w_k1_MAT_s115
                                 + 0.0441753004401131 w_k1_MAT_s116
                                 + 0.0708339304371269 w_k1_MAT_s117
                                 + 0.0492865948684753 w_k1_MSFT_s000
                                 + 0.0174147441142577 w_k1_MSFT_s001
                                 + 0.0370983671004848 w_k1_MSFT_s002
                                 + 0.0214784974452179 w_k1_MSFT_s003
                                 + 0.0436023005393829 w_k1_MSFT_s004
                                 + 0.041709104178324 w_k1_MSFT_s005
                                 + 0.0662115670361017 w_k1_MSFT_s006
                                 + 0.0485103906340007 w_k1_MSFT_s007
                                 + 0.0269125845819409 w_k1_MSFT_s010
                                 + 0.0120908322280143 w_k1_MSFT_s011
                                 + 0.00826130641139343 w_k1_MSFT_s012
                                 + 0.0105077527549872 w_k1_MSFT_s013
                                 + 0.00767514719559248 w_k1_MSFT_s014
                                 + 0.0172366850827641 w_k1_MSFT_s015
                                 + 0.00672739874942325 w_k1_MSFT_s016
                                 + 0.0132040738713718 w_k1_MSFT_s017
                                 + 0.0431305262659104 w_k1_MSFT_s100
                                 + 0.023224747178547 w_k1_MSFT_s101
                                 + 0.0366780776269044 w_k1_MSFT_s102
                                 + 0.0270694948337236 w_k1_MSFT_s103
                                 + 0.023038884278391 w_k1_MSFT_s104
                                 + 0.018133752944415 w_k1_MSFT_s105
                                 + 0.0312522500897291 w_k1_MSFT_s106
                                 + 0.0305940802937536 w_k1_MSFT_s107
                                 + 0.0230651329564611 w_k1_MSFT_s110
                                 + 0.0758072258939165 w_k1_MSFT_s111
                                 + 0.0193297730393962 w_k1_MSFT_s112
                                 + 0.034712596151951 w_k1_MSFT_s113
                                 + 0.0243218013810894 w_k1_MSFT_s114
                                 + 0.0736304109007868 w_k1_MSFT_s115
                                 + 0.0442132007831219 w_k1_MSFT_s116
                                 + 0.0803355281726603 w_k1_MSFT_s117
                                 + 0.0454194509411807 w_k1_PG_s000
                                 + 0.0211113082250372 w_k1_PG_s001
                                 + 0.0358511010447706 w_k1_PG_s002
                                 + 0.0202011839058776 w_k1_PG_s003
                                 + 0.04393102852271 w_k1_PG_s004
                                 + 0.0396349759088694 w_k1_PG_s005
                                 + 0.0590776082999507 w_k1_PG_s006
                                 + 0.0493172677582678 w_k1_PG_s007
                                 + 0.0289512567020791 w_k1_PG_s010
                                 + 0.011065821839156 w_k1_PG_s011
                                 + 0.00807744205084617 w_k1_PG_s012
                                 + 0.0103466656586791 w_k1_PG_s013
                                 + 0.00902710128496498 w_k1_PG_s014
                                 + 0.0166253171248149 w_k1_PG_s015
                                 + 0.00636855385017909 w_k1_PG_s016
                                 + 0.013803558212121 w_k1_PG_s017
                                 + 0.0416320263772195 w_k1_PG_s100
                                 + 0.0217731347016443 w_k1_PG_s101
                                 + 0.0402074463964087 w_k1_PG_s102
                                 + 0.0236117415138907 w_k1_PG_s103
                                 + 0.0237158620024278 w_k1_PG_s104
                                 + 0.0178128501095349 w_k1_PG_s105
                                 + 0.026322700343147 w_k1_PG_s106
                                 + 0.0351018515712986 w_k1_PG_s107
                                 + 0.0210818616456752 w_k1_PG_s110
                                 + 0.0779708328407175 w_k1_PG_s111
                                 + 0.0178608634217122 w_k1_PG_s112
                                 + 0.0369030950878522 w_k1_PG_s113
                                 + 0.0250327052292366 w_k1_PG_s114
                                 + 0.068753209083665 w_k1_PG_s115
                                 + 0.0390338306403625 w_k1_PG_s116
                                 + 0.0703294532777946 w_k1_PG_s117 >= 1.041
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
