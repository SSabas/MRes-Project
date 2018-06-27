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
 w_k1_ADM_s0_return_constraint:  1.17729 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.965797 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.10378 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.837301 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.875961 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.20665 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.882514 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.712235 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.15757 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.40594 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.635092 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.07593 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.863099 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.910182 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.05161 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.18465 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.982198 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.03793 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.23061 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.730053 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.808754 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.11116 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.33211 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.1262 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.0719 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.03438 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.792899 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.942726 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.23625 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.808241 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.02285 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.688277 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.28153 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.912667 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.911742 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.08254 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.870354 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.19299 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.28768 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.912656 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.17779 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.747013 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.903836 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.35648 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.788397 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.68049 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.985301 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.51046 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.643672 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.11167 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.680072 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.825537 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.07907 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.15957 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.00108 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.88975 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.36424 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.727481 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.758178 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.25555 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.32656 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.967002 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.35741 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.51812 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.704866 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.923034 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.28095 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.722069 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.960044 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.559459 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.47783 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.776207 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.87898 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.2726 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.984603 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.38612 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.32389 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.876241 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.17431 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.723068 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.863864 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.43931 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.840508 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.654853 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.11122 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.89016 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.558299 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.24611 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.851478 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.826107 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.838633 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.06653 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.912068 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.939842 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.24477 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.533956 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.720406 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.51599 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.44397 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.01528 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.17464 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.73022 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.728291 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.07572 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.13995 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.702599 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.21939 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.707421 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.92275 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.665657 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.938511 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.20819 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.808353 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.41351 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.20984 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.926044 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.09266 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.84165 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.913942 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.27862 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.939178 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.688947 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.175 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.60108 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.676762 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.10949 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.909374 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.856566 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.933129 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.979424 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.79712 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.03638 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.39081 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.696558 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.820437 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.2468 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.2054 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.12518 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.08624 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.11665 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.849997 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.965659 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.16945 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.719828 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.14161 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.772406 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.35603 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.868509 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.924658 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.05944 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.852422 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.40521 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.18517 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.960512 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.10023 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.730299 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.998633 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.1388 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.944737 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.37398 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.749457 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.96626 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.776939 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.43606 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.616975 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.930314 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.880396 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.2114 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.47342 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.642815 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.689643 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.907018 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.638944 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.09509 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.09536 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.636919 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.43962 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.98982 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.517294 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.970566 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.887452 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.821223 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.815632 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.548525 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.59369 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.604396 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.874075 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.60722 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.15936 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.916488 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.18662 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.896607 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.993519 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.787409 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.07249 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.11727 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.05412 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.80812 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.07188 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.76723 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.818906 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.07418 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.982009 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.853766 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.985701 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.15807 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.885207 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.04015 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.50666 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.878106 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.976302 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.16215 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.969955 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.01433 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.962492 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.10951 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.803884 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.932411 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.07411 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.684951 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.17018 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.910364 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.38537 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.878076 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.08827 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.19049 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.975648 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.20429 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.19219 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.899514 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.13799 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.671613 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.997888 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.26066 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.793063 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.979005 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.851627 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.26671 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.661214 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.35044 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.821492 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.886831 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.9738 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.25798 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.25304 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.934268 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.16781 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.873533 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.781432 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.31873 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.27328 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.87367 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.1865 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.55049 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.671698 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.08391 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.02886 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.730068 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.986308 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.724143 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.53648 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.752786 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.87403 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.23649 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.888896 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.14246 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.23052 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.909243 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.25633 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.732116 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.09004 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.22683 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.655831 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.00663 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.858612 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.05407 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.590146 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.32367 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.541397 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.924614 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.869254 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.25007 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.37048 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.766063 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.888438 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.719808 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.531005 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.37845 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.24324 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.541015 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.15277 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.93009 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.388285 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.865389 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.04743 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.655392 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.805535 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.431468 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.74415 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.642561 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.901065 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.52803 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.19065 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.14222 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.1974 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.938986 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.05716 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.851569 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.937018 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.1072 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.903513 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.743617 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.10155 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.39248 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.673495 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.19713 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.01566 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.908126 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.799751 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.02101 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.800638 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.04446 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.1635 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.744745 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.790123 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.19015 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.16526 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.04809 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.96323 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.24956 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.807256 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.0176 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.12665 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.758709 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.23749 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.871256 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.40201 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.964441 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.057 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.18806 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.879289 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.39301 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.18933 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.951099 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.01828 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.761825 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.968506 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.12914 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.902689 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.905566 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.929855 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.35513 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.79008 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.09495 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.820458 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.925091 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.0228 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.1943 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.00997 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.984998 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.1782 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.871436 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.913792 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.19821 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.1591 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.91191 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.09293 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.1446 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.819955 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.01471 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.00798 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.796645 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.01686 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.827107 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.40745 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.85336 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.981511 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.07372 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.04716 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.14124 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.311000565772898 z_k1_s0000
                                 - 0.208000498844956 z_k1_s0001
                                 - 0.241999211655474 z_k1_s0002
                                 - 0.226000037744751 z_k1_s0003
                                 - 0.117999994844592 z_k1_s0004
                                 - 0.311000565772898 z_k1_s0005
                                 - 0.28300107492867 z_k1_s0006
                                 - 0.174000100333605 z_k1_s0007
                                 - 0.159000039752219 z_k1_s0010
                                 - 0.432000064315456 z_k1_s0011
                                 - 0.184999931824889 z_k1_s0012
                                 - 0.146000093715884 z_k1_s0013
                                 - 0.178999218792608 z_k1_s0014
                                 - 0.176999532448609 z_k1_s0015
                                 - 0.252000175350706 z_k1_s0016
                                 - 0.124000070531297 z_k1_s0017
                                 - 0.501998296515073 z_k1_s0020
                                 - 0.322999722214908 z_k1_s0021
                                 - 0.744997734421718 z_k1_s0022
                                 - 0.306999847113218 z_k1_s0023
                                 - 0.513998998041177 z_k1_s0024
                                 - 0.655998968768455 z_k1_s0025
                                 - 0.524000339979441 z_k1_s0026
                                 - 0.973001975419562 z_k1_s0027
                                 - 0.0969999708720482 z_k1_s0030
                                 - 0.146000066245598 z_k1_s0031
                                 - 0.169000038369891 z_k1_s0032
                                 - 0.136000070995903 z_k1_s0033
                                 - 0.329999673739873 z_k1_s0034
                                 - 0.368999943363791 z_k1_s0035
                                 - 0.265999839741875 z_k1_s0036
                                 - 0.182000693244742 z_k1_s0037
                                 - 0.193999113273515 z_k1_s0040
                                 - 0.876997208985212 z_k1_s0041
                                 - 0.240999174768574 z_k1_s0042
                                 - 0.401998933951322 z_k1_s0043
                                 - 0.202999089406753 z_k1_s0044
                                 - 0.274998898472662 z_k1_s0045
                                 - 0.317999156707762 z_k1_s0046
                                 - 0.281999252174808 z_k1_s0047
                                 - 1.17399812648488 z_k1_s0050
                                 - 0.796002282127896 z_k1_s0051
                                 - 0.790996076705618 z_k1_s0052
                                 - 0.723999614967824 z_k1_s0053
                                 - 1.74600200343906 z_k1_s0054
                                 - 0.755999498706824 z_k1_s0055
                                 - 0.439999377660359 z_k1_s0056
                                 - 1.38300134537034 z_k1_s0057
                                 - 0.263999667591509 z_k1_s0060
                                 - 0.317000002843102 z_k1_s0061
                                 - 0.163000155614694 z_k1_s0062
                                 - 0.617000410379429 z_k1_s0063
                                 - 0.177000141726368 z_k1_s0064
                                 - 0.574999958444081 z_k1_s0065
                                 - 0.19400023063347 z_k1_s0066
                                 - 0.161000087084408 z_k1_s0067
                                 - 0.107999935929297 z_k1_s0070
                                 - 0.441001290678716 z_k1_s0071
                                 - 0.317000238596385 z_k1_s0072
                                 - 0.342000450709758 z_k1_s0073
                                 - 0.17299999722383 z_k1_s0074
                                 - 0.506999399656833 z_k1_s0075
                                 - 0.234000024580223 z_k1_s0076
                                 - 0.329000340410804 z_k1_s0077
                                 - 0.130000170737627 z_k1_s0100
                                 - 0.112000404190105 z_k1_s0101
                                 - 0.289000893749023 z_k1_s0102
                                 - 0.119000368014304 z_k1_s0103
                                 - 0.0780002007428686 z_k1_s0104
                                 - 0.0580002198449071 z_k1_s0105
                                 - 0.0680002102938878 z_k1_s0106
                                 - 0.129000456763577 z_k1_s0107
                                 - 0.123999409561948 z_k1_s0110
                                 - 0.0819996917294181 z_k1_s0111
                                 - 0.0859997585988892 z_k1_s0112
                                 - 0.265998714737786 z_k1_s0113
                                 - 0.156999990184995 z_k1_s0114
                                 - 0.167999797727219 z_k1_s0115
                                 - 0.188998903945842 z_k1_s0116
                                 - 0.0859997585988892 z_k1_s0117
                                 - 0.105999941958793 z_k1_s0120
                                 - 0.111999874545185 z_k1_s0121
                                 - 0.144999738170175 z_k1_s0122
                                 - 0.139999989681378 z_k1_s0123
                                 - 0.259000098110468 z_k1_s0124
                                 - 0.23499989896523 z_k1_s0125
                                 - 0.0959999761815283 z_k1_s0126
                                 - 0.0789998937202769 z_k1_s0127
                                 - 0.0589999551536027 z_k1_s0130
                                 - 0.134999793656597 z_k1_s0131
                                 - 0.0779999759793229 z_k1_s0132
                                 - 0.137999816271175 z_k1_s0133
                                 - 0.0439999644806538 z_k1_s0134
                                 - 0.0629998629064306 z_k1_s0135
                                 - 0.0549999862008032 z_k1_s0136
                                 - 0.0399999955278543 z_k1_s0137
                                 - 0.0479999797974122 z_k1_s0140
                                 - 0.083000286432965 z_k1_s0141
                                 - 0.0560000178969731 z_k1_s0142
                                 - 0.121000125305944 z_k1_s0143
                                 - 0.103000070681926 z_k1_s0144
                                 - 0.0560000178969731 z_k1_s0145
                                 - 0.0510000096347446 z_k1_s0146
                                 - 0.104000246494339 z_k1_s0147
                                 - 0.0979996575402944 z_k1_s0150
                                 - 0.0829999031003304 z_k1_s0151
                                 - 0.106000112441235 z_k1_s0152
                                 - 0.0589999579966032 z_k1_s0153
                                 - 0.0359999514555691 z_k1_s0154
                                 - 0.165000002837882 z_k1_s0155
                                 - 0.0409999822688187 z_k1_s0156
                                 - 0.0880002719133645 z_k1_s0157
                                 - 0.0419999933311641 z_k1_s0160
                                 - 0.0629999542467769 z_k1_s0161
                                 - 0.161999555491994 z_k1_s0162
                                 - 0.0730000071351024 z_k1_s0163
                                 - 0.179000095851759 z_k1_s0164
                                 - 0.0689999430798092 z_k1_s0165
                                 - 0.038999998914648 z_k1_s0166
                                 - 0.087999979217683 z_k1_s0167
                                 - 0.214999354587155 z_k1_s0170
                                 - 0.0679997996531367 z_k1_s0171
                                 - 0.290999580486547 z_k1_s0172
                                 - 0.165000147914047 z_k1_s0173
                                 - 0.337998606979862 z_k1_s0174
                                 - 0.0889997593150787 z_k1_s0175
                                 - 0.160999528041979 z_k1_s0176
                                 - 0.299998494030166 z_k1_s0177
                                 - 0.735001364721549 z_k1_s1000
                                 - 1.52799859765369 z_k1_s1001
                                 - 0.52699939072395 z_k1_s1002
                                 - 0.394999254439092 z_k1_s1003
                                 - 0.740001212808345 z_k1_s1004
                                 - 0.380999679796062 z_k1_s1005
                                 - 0.875998265567329 z_k1_s1006
                                 - 0.741997597648689 z_k1_s1007
                                 - 0.752997197709339 z_k1_s1010
                                 - 1.42799312548178 z_k1_s1011
                                 - 0.369998980108335 z_k1_s1012
                                 - 1.0009938917635 z_k1_s1013
                                 - 0.364998598959514 z_k1_s1014
                                 - 0.644999485860397 z_k1_s1015
                                 - 0.555997961894178 z_k1_s1016
                                 - 0.726998722724001 z_k1_s1017
                                 - 0.837996369346013 z_k1_s1020
                                 - 0.301999402918668 z_k1_s1021
                                 - 0.687996438022573 z_k1_s1022
                                 - 0.436999341109764 z_k1_s1023
                                 - 0.848997281934779 z_k1_s1024
                                 - 0.456998979020179 z_k1_s1025
                                 - 0.905997679356939 z_k1_s1026
                                 - 0.817000402393632 z_k1_s1027
                                 - 0.264999974206344 z_k1_s1030
                                 - 0.549999072503694 z_k1_s1031
                                 - 0.353000103726947 z_k1_s1032
                                 - 0.412000206355534 z_k1_s1033
                                 - 1.22500193457659 z_k1_s1034
                                 - 0.258000095494484 z_k1_s1035
                                 - 0.349999957193284 z_k1_s1036
                                 - 1.21899886230914 z_k1_s1037
                                 - 0.510998433302061 z_k1_s1040
                                 - 0.493000183201034 z_k1_s1041
                                 - 0.282000029510141 z_k1_s1042
                                 - 0.779001293244419 z_k1_s1043
                                 - 0.743998770827039 z_k1_s1044
                                 - 0.575001217437094 z_k1_s1045
                                 - 0.58399820948807 z_k1_s1046
                                 - 0.298000085644038 z_k1_s1047
                                 - 0.767001346508749 z_k1_s1050
                                 - 2.23500499589849 z_k1_s1051
                                 - 1.10500016042551 z_k1_s1052
                                 - 0.934998116741991 z_k1_s1053
                                 - 1.59799821299729 z_k1_s1054
                                 - 0.991000539757697 z_k1_s1055
                                 - 0.536000319305814 z_k1_s1056
                                 - 0.58200087230253 z_k1_s1057
                                 - 0.382000043416376 z_k1_s1060
                                 - 0.268999878018335 z_k1_s1061
                                 - 1.21900110547531 z_k1_s1062
                                 - 0.577999374708536 z_k1_s1063
                                 - 0.254999925783181 z_k1_s1064
                                 - 0.816001133706203 z_k1_s1065
                                 - 0.318999829858174 z_k1_s1066
                                 - 0.446998352508139 z_k1_s1067
                                 - 1.04800077403804 z_k1_s1070
                                 - 1.48499894258531 z_k1_s1071
                                 - 0.604999396442236 z_k1_s1072
                                 - 1.87399942573294 z_k1_s1073
                                 - 1.12899511896316 z_k1_s1074
                                 - 1.75999909228616 z_k1_s1075
                                 - 0.384000141189568 z_k1_s1076
                                 - 1.04399863467235 z_k1_s1077
                                 - 0.522999146736012 z_k1_s1100
                                 - 0.29799937919556 z_k1_s1101
                                 - 0.482999248239978 z_k1_s1102
                                 - 0.40500066477364 z_k1_s1103
                                 - 0.355999367414914 z_k1_s1104
                                 - 0.115999976438707 z_k1_s1105
                                 - 0.149000231198193 z_k1_s1106
                                 - 0.377999176187629 z_k1_s1107
                                 - 0.102000025387549 z_k1_s1110
                                 - 0.113000071670549 z_k1_s1111
                                 - 0.12800005907459 z_k1_s1112
                                 - 0.0900000632243356 z_k1_s1113
                                 - 0.227000614421689 z_k1_s1114
                                 - 0.099000000146722 z_k1_s1115
                                 - 0.186999815210349 z_k1_s1116
                                 - 0.442000572679693 z_k1_s1117
                                 - 0.0940000049870854 z_k1_s1120
                                 - 0.177000544190818 z_k1_s1121
                                 - 0.226999782843177 z_k1_s1122
                                 - 0.1220000064726 z_k1_s1123
                                 - 0.235999706920629 z_k1_s1124
                                 - 0.0950000814401742 z_k1_s1125
                                 - 0.139000083774555 z_k1_s1126
                                 - 0.437999870437627 z_k1_s1127
                                 - 0.275001036888469 z_k1_s1130
                                 - 0.306002004009607 z_k1_s1131
                                 - 0.407001780715746 z_k1_s1132
                                 - 0.36300153277332 z_k1_s1133
                                 - 0.252999887413879 z_k1_s1134
                                 - 0.148000442432323 z_k1_s1135
                                 - 0.202000578915656 z_k1_s1136
                                 - 0.0970003135313973 z_k1_s1137
                                 - 0.44399930598884 z_k1_s1140
                                 - 0.469999912942029 z_k1_s1141
                                 - 1.3969969251538 z_k1_s1142
                                 - 0.401999168109164 z_k1_s1143
                                 - 0.488999812316448 z_k1_s1144
                                 - 0.314999436700312 z_k1_s1145
                                 - 0.488000298032759 z_k1_s1146
                                 - 0.558997760119826 z_k1_s1147
                                 - 0.168000195022229 z_k1_s1150
                                 - 0.470999708832784 z_k1_s1151
                                 - 0.562001515713452 z_k1_s1152
                                 - 0.525001752070783 z_k1_s1153
                                 - 0.32700136987298 z_k1_s1154
                                 - 0.162000035706974 z_k1_s1155
                                 - 0.478001722902689 z_k1_s1156
                                 - 0.354000867990224 z_k1_s1157
                                 - 0.124000100977312 z_k1_s1160
                                 - 0.230999750689405 z_k1_s1161
                                 - 0.514999996663269 z_k1_s1162
                                 - 0.153000187081768 z_k1_s1163
                                 - 0.286001210665116 z_k1_s1164
                                 - 0.246000148864599 z_k1_s1165
                                 - 0.433999669719674 z_k1_s1166
                                 - 0.290000861044555 z_k1_s1167
                                 - 0.0899999970158038 z_k1_s1170
                                 - 0.150999431240933 z_k1_s1171
                                 - 0.203999556782468 z_k1_s1172
                                 - 0.0920000445117128 z_k1_s1173
                                 - 0.12899966418597 z_k1_s1174
                                 - 0.251000547036324 z_k1_s1175
                                 - 0.209999951070207 z_k1_s1176
                                 - 0.131999861329839 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.662012 w_k1_ADM_s000
                                 + 0.850595 w_k1_AXP_s000
                                 + 0.749992 w_k1_DIS_s000
                                 + 0.777188 w_k1_FDX_s000
                                 + 1.13683 w_k1_IBM_s000
                                 + 0.984325 w_k1_INTC_s000
                                 + 0.997905 w_k1_KO_s000
                                 + 1.23318 w_k1_MAT_s000
                                 + 0.817788 w_k1_MSFT_s000
                                 + 0.809562 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.26518 w_k1_ADM_s000
                                 + 1.3432 w_k1_AXP_s000 + 1.47498 w_k1_DIS_s000
                                 + 1.25643 w_k1_FDX_s000
                                 + 1.40718 w_k1_IBM_s000
                                 + 1.25964 w_k1_INTC_s000
                                 + 1.38208 w_k1_KO_s000 + 1.34291 w_k1_MAT_s000
                                 + 1.26069 w_k1_MSFT_s000
                                 + 1.09742 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.22174 w_k1_ADM_s000
                                 + 1.19995 w_k1_AXP_s000
                                 + 1.10444 w_k1_DIS_s000
                                 + 1.13293 w_k1_FDX_s000
                                 + 0.828801 w_k1_IBM_s000
                                 + 1.26561 w_k1_INTC_s000
                                 + 1.09618 w_k1_KO_s000 + 0.90127 w_k1_MAT_s000
                                 + 1.28784 w_k1_MSFT_s000
                                 + 1.00588 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.824994 w_k1_ADM_s000
                                 + 0.824626 w_k1_AXP_s000
                                 + 0.903033 w_k1_DIS_s000
                                 + 0.874181 w_k1_FDX_s000
                                 + 1.00234 w_k1_IBM_s000
                                 + 0.949905 w_k1_INTC_s000
                                 + 1.17481 w_k1_KO_s000 + 1.50639 w_k1_MAT_s000
                                 + 1.03547 w_k1_MSFT_s000
                                 + 1.08171 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.2444 w_k1_ADM_s000
                                 + 1.44396 w_k1_AXP_s000
                                 + 1.27802 w_k1_DIS_s000
                                 + 1.21493 w_k1_FDX_s000
                                 + 1.11179 w_k1_IBM_s000
                                 + 0.925059 w_k1_INTC_s000
                                 + 1.48448 w_k1_KO_s000 + 1.94567 w_k1_MAT_s000
                                 + 1.17129 w_k1_MSFT_s000
                                 + 1.11195 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.82879 w_k1_ADM_s000
                                 + 0.716445 w_k1_AXP_s000
                                 + 0.78546 w_k1_DIS_s000
                                 + 0.883798 w_k1_FDX_s000
                                 + 0.68107 w_k1_IBM_s000
                                 + 0.909041 w_k1_INTC_s000
                                 + 0.880683 w_k1_KO_s000
                                 + 0.823436 w_k1_MAT_s000
                                 + 0.91361 w_k1_MSFT_s000
                                 + 0.848955 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.950305 w_k1_ADM_s000
                                 + 1.06931 w_k1_AXP_s000
                                 + 1.05157 w_k1_DIS_s000
                                 + 0.964997 w_k1_FDX_s000
                                 + 1.18852 w_k1_IBM_s000
                                 + 0.87919 w_k1_INTC_s000
                                 + 1.06583 w_k1_KO_s000 + 1.08019 w_k1_MAT_s000
                                 + 0.978821 w_k1_MSFT_s000
                                 + 0.905989 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.06305 w_k1_ADM_s000
                                 + 0.904386 w_k1_AXP_s000
                                 + 1.24536 w_k1_DIS_s000 + 1.0018 w_k1_FDX_s000
                                 + 0.92448 w_k1_IBM_s000
                                 + 0.934854 w_k1_INTC_s000
                                 + 1.21042 w_k1_KO_s000 + 1.1886 w_k1_MAT_s000
                                 + 1.05295 w_k1_MSFT_s000
                                 + 1.03411 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.813286 w_k1_ADM_s001
                                 + 0.781104 w_k1_AXP_s001
                                 + 0.882775 w_k1_DIS_s001
                                 + 0.867957 w_k1_FDX_s001
                                 + 0.884014 w_k1_IBM_s001
                                 + 0.926682 w_k1_INTC_s001
                                 + 0.8288 w_k1_KO_s001 + 0.767692 w_k1_MAT_s001
                                 + 0.923597 w_k1_MSFT_s001
                                 + 0.849907 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.35657 w_k1_ADM_s001
                                 + 1.11141 w_k1_AXP_s001
                                 + 1.19981 w_k1_DIS_s001
                                 + 1.12605 w_k1_FDX_s001
                                 + 0.962501 w_k1_IBM_s001
                                 + 1.00842 w_k1_INTC_s001
                                 + 1.01186 w_k1_KO_s001 + 1.14872 w_k1_MAT_s001
                                 + 1.14562 w_k1_MSFT_s001
                                 + 1.20482 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.818183 w_k1_ADM_s001
                                 + 0.755478 w_k1_AXP_s001
                                 + 0.8906 w_k1_DIS_s001
                                 + 0.855419 w_k1_FDX_s001
                                 + 0.706635 w_k1_IBM_s001
                                 + 0.911674 w_k1_INTC_s001
                                 + 0.826631 w_k1_KO_s001
                                 + 0.694189 w_k1_MAT_s001
                                 + 0.837093 w_k1_MSFT_s001
                                 + 0.887253 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.70935 w_k1_ADM_s001
                                 + 0.873272 w_k1_AXP_s001
                                 + 0.762569 w_k1_DIS_s001
                                 + 0.793175 w_k1_FDX_s001
                                 + 1.0976 w_k1_IBM_s001
                                 + 0.999061 w_k1_INTC_s001
                                 + 0.792037 w_k1_KO_s001
                                 + 1.18774 w_k1_MAT_s001
                                 + 0.828611 w_k1_MSFT_s001
                                 + 0.812872 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.01076 w_k1_ADM_s001
                                 + 0.809419 w_k1_AXP_s001
                                 + 0.797686 w_k1_DIS_s001
                                 + 0.893033 w_k1_FDX_s001
                                 + 0.659734 w_k1_IBM_s001
                                 + 1.05752 w_k1_INTC_s001
                                 + 0.732112 w_k1_KO_s001
                                 + 0.654999 w_k1_MAT_s001
                                 + 0.92378 w_k1_MSFT_s001
                                 + 0.912349 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.971042 w_k1_ADM_s001
                                 + 1.21908 w_k1_AXP_s001
                                 + 1.16206 w_k1_DIS_s001
                                 + 1.01797 w_k1_FDX_s001
                                 + 1.61607 w_k1_IBM_s001
                                 + 1.10324 w_k1_INTC_s001
                                 + 1.17119 w_k1_KO_s001 + 1.71118 w_k1_MAT_s001
                                 + 0.918066 w_k1_MSFT_s001
                                 + 1.00593 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.60363 w_k1_ADM_s001
                                 + 1.73795 w_k1_AXP_s001
                                 + 1.57091 w_k1_DIS_s001
                                 + 1.25742 w_k1_FDX_s001
                                 + 1.43218 w_k1_IBM_s001
                                 + 1.26108 w_k1_INTC_s001
                                 + 1.46167 w_k1_KO_s001 + 1.64623 w_k1_MAT_s001
                                 + 1.10898 w_k1_MSFT_s001
                                 + 1.23344 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.871158 w_k1_ADM_s001
                                 + 0.816729 w_k1_AXP_s001
                                 + 1.01143 w_k1_DIS_s001
                                 + 0.923795 w_k1_FDX_s001
                                 + 1.11297 w_k1_IBM_s001
                                 + 1.07984 w_k1_INTC_s001
                                 + 1.06107 w_k1_KO_s001
                                 + 0.936946 w_k1_MAT_s001
                                 + 1.01169 w_k1_MSFT_s001
                                 + 1.06154 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.215 w_k1_ADM_s002
                                 + 1.23453 w_k1_AXP_s002
                                 + 1.53719 w_k1_DIS_s002
                                 + 1.31721 w_k1_FDX_s002
                                 + 1.03122 w_k1_IBM_s002
                                 + 1.2695 w_k1_INTC_s002 + 1.23803 w_k1_KO_s002
                                 + 0.686921 w_k1_MAT_s002
                                 + 1.18258 w_k1_MSFT_s002 + 1.2088 w_k1_PG_s002
                                 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.775427 w_k1_ADM_s002
                                 + 1.08691 w_k1_AXP_s002
                                 + 0.970717 w_k1_DIS_s002
                                 + 0.901037 w_k1_FDX_s002
                                 + 1.15225 w_k1_IBM_s002
                                 + 1.05543 w_k1_INTC_s002 + 1.0408 w_k1_KO_s002
                                 + 0.77779 w_k1_MAT_s002
                                 + 0.812977 w_k1_MSFT_s002
                                 + 0.949138 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.911 w_k1_ADM_s002
                                 + 0.835969 w_k1_AXP_s002
                                 + 0.872836 w_k1_DIS_s002
                                 + 0.897599 w_k1_FDX_s002
                                 + 1.06557 w_k1_IBM_s002
                                 + 1.12259 w_k1_INTC_s002
                                 + 1.06141 w_k1_KO_s002
                                 + 0.759704 w_k1_MAT_s002
                                 + 0.927533 w_k1_MSFT_s002
                                 + 1.04854 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.05922 w_k1_ADM_s002
                                 + 1.46215 w_k1_AXP_s002
                                 + 0.985344 w_k1_DIS_s002
                                 + 1.04351 w_k1_FDX_s002
                                 + 1.12578 w_k1_IBM_s002
                                 + 1.10063 w_k1_INTC_s002
                                 + 1.10245 w_k1_KO_s002 + 1.1094 w_k1_MAT_s002
                                 + 0.97165 w_k1_MSFT_s002
                                 + 1.07951 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.699156 w_k1_ADM_s002
                                 + 0.747887 w_k1_AXP_s002
                                 + 0.601608 w_k1_DIS_s002
                                 + 0.679253 w_k1_FDX_s002
                                 + 0.874575 w_k1_IBM_s002
                                 + 0.860779 w_k1_INTC_s002
                                 + 0.724714 w_k1_KO_s002
                                 + 0.622002 w_k1_MAT_s002
                                 + 0.685761 w_k1_MSFT_s002
                                 + 0.909568 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.989248 w_k1_ADM_s002
                                 + 0.846471 w_k1_AXP_s002
                                 + 0.741171 w_k1_DIS_s002
                                 + 0.911671 w_k1_FDX_s002
                                 + 0.647595 w_k1_IBM_s002
                                 + 0.719927 w_k1_INTC_s002
                                 + 0.843899 w_k1_KO_s002
                                 + 0.545563 w_k1_MAT_s002
                                 + 0.836484 w_k1_MSFT_s002
                                 + 1.04034 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.07822 w_k1_ADM_s002
                                 + 0.953063 w_k1_AXP_s002
                                 + 1.0265 w_k1_DIS_s002 + 1.114 w_k1_FDX_s002
                                 + 0.874885 w_k1_IBM_s002
                                 + 1.21261 w_k1_INTC_s002
                                 + 1.04235 w_k1_KO_s002
                                 + 0.697592 w_k1_MAT_s002
                                 + 1.20351 w_k1_MSFT_s002
                                 + 1.11337 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.09419 w_k1_ADM_s002
                                 + 1.25054 w_k1_AXP_s002
                                 + 1.38176 w_k1_DIS_s002
                                 + 1.15161 w_k1_FDX_s002
                                 + 1.46466 w_k1_IBM_s002
                                 + 1.23223 w_k1_INTC_s002
                                 + 1.48901 w_k1_KO_s002
                                 + 0.970273 w_k1_MAT_s002
                                 + 1.1902 w_k1_MSFT_s002 + 1.27891 w_k1_PG_s002
                                 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.22461 w_k1_ADM_s003
                                 + 0.900152 w_k1_AXP_s003
                                 + 0.700503 w_k1_DIS_s003
                                 + 1.04742 w_k1_FDX_s003
                                 + 0.570228 w_k1_IBM_s003
                                 + 0.765112 w_k1_INTC_s003
                                 + 0.659047 w_k1_KO_s003
                                 + 0.505446 w_k1_MAT_s003
                                 + 1.02326 w_k1_MSFT_s003
                                 + 0.764477 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.821938 w_k1_ADM_s003
                                 + 0.702196 w_k1_AXP_s003
                                 + 0.623844 w_k1_DIS_s003
                                 + 0.764729 w_k1_FDX_s003
                                 + 0.629121 w_k1_IBM_s003
                                 + 0.640265 w_k1_INTC_s003
                                 + 0.59815 w_k1_KO_s003
                                 + 0.723856 w_k1_MAT_s003
                                 + 0.779798 w_k1_MSFT_s003
                                 + 0.708334 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.816192 w_k1_ADM_s003
                                 + 0.995762 w_k1_AXP_s003
                                 + 0.744595 w_k1_DIS_s003
                                 + 0.885347 w_k1_FDX_s003 + 0.911 w_k1_IBM_s003
                                 + 0.713974 w_k1_INTC_s003
                                 + 0.861348 w_k1_KO_s003
                                 + 0.961509 w_k1_MAT_s003
                                 + 0.948643 w_k1_MSFT_s003
                                 + 0.877865 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.13008 w_k1_ADM_s003
                                 + 1.88652 w_k1_AXP_s003
                                 + 1.75707 w_k1_DIS_s003
                                 + 1.35186 w_k1_FDX_s003
                                 + 2.22757 w_k1_IBM_s003
                                 + 1.3024 w_k1_INTC_s003 + 1.68701 w_k1_KO_s003
                                 + 1.94718 w_k1_MAT_s003
                                 + 1.46907 w_k1_MSFT_s003
                                 + 1.31367 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.098 w_k1_ADM_s003
                                 + 1.13817 w_k1_AXP_s003
                                 + 1.13733 w_k1_DIS_s003
                                 + 1.19042 w_k1_FDX_s003
                                 + 1.06608 w_k1_IBM_s003
                                 + 0.924912 w_k1_INTC_s003
                                 + 1.09993 w_k1_KO_s003
                                 + 0.959567 w_k1_MAT_s003
                                 + 1.16926 w_k1_MSFT_s003
                                 + 0.865204 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.946563 w_k1_ADM_s003
                                 + 0.94208 w_k1_AXP_s003
                                 + 0.929806 w_k1_DIS_s003
                                 + 0.971831 w_k1_FDX_s003
                                 + 0.876709 w_k1_IBM_s003
                                 + 0.657269 w_k1_INTC_s003
                                 + 0.958092 w_k1_KO_s003
                                 + 0.872299 w_k1_MAT_s003
                                 + 0.908288 w_k1_MSFT_s003
                                 + 0.966791 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.919357 w_k1_ADM_s003
                                 + 0.991551 w_k1_AXP_s003
                                 + 0.947713 w_k1_DIS_s003
                                 + 0.79897 w_k1_FDX_s003 + 1.4426 w_k1_IBM_s003
                                 + 0.6108 w_k1_INTC_s003 + 1.06319 w_k1_KO_s003
                                 + 1.40788 w_k1_MAT_s003
                                 + 0.892771 w_k1_MSFT_s003
                                 + 0.981303 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.31291 w_k1_ADM_s003
                                 + 1.15328 w_k1_AXP_s003
                                 + 1.60434 w_k1_DIS_s003
                                 + 1.16273 w_k1_FDX_s003
                                 + 1.41476 w_k1_IBM_s003
                                 + 0.918218 w_k1_INTC_s003
                                 + 1.3253 w_k1_KO_s003 + 0.958096 w_k1_MAT_s003
                                 + 1.35111 w_k1_MSFT_s003
                                 + 1.05977 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.3797 w_k1_ADM_s004
                                 + 1.48809 w_k1_AXP_s004 + 1.4488 w_k1_DIS_s004
                                 + 1.42241 w_k1_FDX_s004
                                 + 1.06718 w_k1_IBM_s004
                                 + 1.30523 w_k1_INTC_s004
                                 + 1.29186 w_k1_KO_s004 + 1.63393 w_k1_MAT_s004
                                 + 1.32745 w_k1_MSFT_s004
                                 + 1.28549 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.971239 w_k1_ADM_s004
                                 + 0.648219 w_k1_AXP_s004
                                 + 0.755978 w_k1_DIS_s004
                                 + 0.773978 w_k1_FDX_s004
                                 + 0.949766 w_k1_IBM_s004
                                 + 0.754045 w_k1_INTC_s004
                                 + 0.987712 w_k1_KO_s004
                                 + 0.958276 w_k1_MAT_s004
                                 + 0.867261 w_k1_MSFT_s004
                                 + 0.904952 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.33764 w_k1_ADM_s004
                                 + 1.09309 w_k1_AXP_s004
                                 + 1.70312 w_k1_DIS_s004
                                 + 1.17147 w_k1_FDX_s004
                                 + 1.18177 w_k1_IBM_s004
                                 + 0.98512 w_k1_INTC_s004
                                 + 1.40695 w_k1_KO_s004 + 1.47614 w_k1_MAT_s004
                                 + 1.31117 w_k1_MSFT_s004
                                 + 1.18425 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.43775 w_k1_ADM_s004
                                 + 0.844968 w_k1_AXP_s004
                                 + 0.95952 w_k1_DIS_s004
                                 + 1.02154 w_k1_FDX_s004
                                 + 0.61523 w_k1_IBM_s004
                                 + 1.00308 w_k1_INTC_s004
                                 + 0.797882 w_k1_KO_s004
                                 + 0.76557 w_k1_MAT_s004
                                 + 1.16457 w_k1_MSFT_s004
                                 + 0.88695 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.3989 w_k1_ADM_s004
                                 + 1.18328 w_k1_AXP_s004
                                 + 0.861823 w_k1_DIS_s004
                                 + 1.07836 w_k1_FDX_s004
                                 + 1.02963 w_k1_IBM_s004
                                 + 1.02337 w_k1_INTC_s004
                                 + 0.908055 w_k1_KO_s004
                                 + 0.945641 w_k1_MAT_s004
                                 + 1.00956 w_k1_MSFT_s004
                                 + 0.955493 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.35839 w_k1_ADM_s004
                                 + 0.931661 w_k1_AXP_s004
                                 + 1.28943 w_k1_DIS_s004
                                 + 1.17473 w_k1_FDX_s004
                                 + 0.830276 w_k1_IBM_s004
                                 + 0.965187 w_k1_INTC_s004
                                 + 1.13629 w_k1_KO_s004
                                 + 0.964055 w_k1_MAT_s004
                                 + 1.29749 w_k1_MSFT_s004
                                 + 1.01202 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.2162 w_k1_ADM_s004
                                 + 1.28341 w_k1_AXP_s004 + 1.3618 w_k1_DIS_s004
                                 + 1.14491 w_k1_FDX_s004
                                 + 1.50765 w_k1_IBM_s004
                                 + 1.06696 w_k1_INTC_s004
                                 + 1.33433 w_k1_KO_s004 + 2.10727 w_k1_MAT_s004
                                 + 1.24234 w_k1_MSFT_s004
                                 + 1.25773 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.19022 w_k1_ADM_s004
                                 + 0.737177 w_k1_AXP_s004
                                 + 1.1743 w_k1_DIS_s004
                                 + 0.950359 w_k1_FDX_s004
                                 + 0.901029 w_k1_IBM_s004
                                 + 0.880394 w_k1_INTC_s004
                                 + 1.0472 w_k1_KO_s004 + 0.762075 w_k1_MAT_s004
                                 + 1.04536 w_k1_MSFT_s004
                                 + 0.917571 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.948542 w_k1_ADM_s005
                                 + 1.11561 w_k1_AXP_s005
                                 + 0.883324 w_k1_DIS_s005
                                 + 1.0073 w_k1_FDX_s005
                                 + 0.846657 w_k1_IBM_s005
                                 + 1.02957 w_k1_INTC_s005
                                 + 0.740819 w_k1_KO_s005
                                 + 0.835783 w_k1_MAT_s005
                                 + 0.803964 w_k1_MSFT_s005
                                 + 0.968388 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.988641 w_k1_ADM_s005
                                 + 0.771824 w_k1_AXP_s005
                                 + 1.08995 w_k1_DIS_s005
                                 + 1.05955 w_k1_FDX_s005
                                 + 0.637458 w_k1_IBM_s005
                                 + 1.21537 w_k1_INTC_s005
                                 + 0.795082 w_k1_KO_s005
                                 + 0.601723 w_k1_MAT_s005
                                 + 1.08918 w_k1_MSFT_s005
                                 + 0.956376 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.03109 w_k1_ADM_s005
                                 + 1.39841 w_k1_AXP_s005
                                 + 1.40183 w_k1_DIS_s005
                                 + 1.16151 w_k1_FDX_s005
                                 + 1.83664 w_k1_IBM_s005
                                 + 1.32782 w_k1_INTC_s005
                                 + 1.47666 w_k1_KO_s005 + 2.0413 w_k1_MAT_s005
                                 + 1.06307 w_k1_MSFT_s005
                                 + 1.22099 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.20648 w_k1_ADM_s005
                                 + 1.48793 w_k1_AXP_s005
                                 + 1.33878 w_k1_DIS_s005
                                 + 1.25447 w_k1_FDX_s005
                                 + 0.895168 w_k1_IBM_s005
                                 + 1.09662 w_k1_INTC_s005
                                 + 0.839248 w_k1_KO_s005
                                 + 1.17707 w_k1_MAT_s005
                                 + 0.995208 w_k1_MSFT_s005
                                 + 1.11857 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.983594 w_k1_ADM_s005
                                 + 0.912465 w_k1_AXP_s005
                                 + 0.922579 w_k1_DIS_s005
                                 + 0.816675 w_k1_FDX_s005
                                 + 1.16012 w_k1_IBM_s005
                                 + 0.918231 w_k1_INTC_s005
                                 + 0.954223 w_k1_KO_s005
                                 + 1.10672 w_k1_MAT_s005
                                 + 0.876959 w_k1_MSFT_s005
                                 + 0.925077 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.00806 w_k1_ADM_s005
                                 + 1.27262 w_k1_AXP_s005 + 1.6626 w_k1_DIS_s005
                                 + 1.28864 w_k1_FDX_s005 + 1.2219 w_k1_IBM_s005
                                 + 1.2337 w_k1_INTC_s005 + 1.32343 w_k1_KO_s005
                                 + 1.69101 w_k1_MAT_s005
                                 + 1.18234 w_k1_MSFT_s005
                                 + 1.19996 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.16283 w_k1_ADM_s005
                                 + 1.58611 w_k1_AXP_s005
                                 + 0.972774 w_k1_DIS_s005
                                 + 1.05032 w_k1_FDX_s005
                                 + 1.18309 w_k1_IBM_s005
                                 + 1.47545 w_k1_INTC_s005
                                 + 1.16002 w_k1_KO_s005 + 1.70718 w_k1_MAT_s005
                                 + 1.02102 w_k1_MSFT_s005
                                 + 1.15801 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.833379 w_k1_ADM_s005
                                 + 0.668417 w_k1_AXP_s005
                                 + 0.648195 w_k1_DIS_s005
                                 + 0.721724 w_k1_FDX_s005
                                 + 0.708709 w_k1_IBM_s005
                                 + 0.747059 w_k1_INTC_s005
                                 + 0.761718 w_k1_KO_s005
                                 + 0.691749 w_k1_MAT_s005
                                 + 0.719097 w_k1_MSFT_s005
                                 + 0.853225 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.18198 w_k1_ADM_s006
                                 + 0.941105 w_k1_AXP_s006
                                 + 0.94907 w_k1_DIS_s006
                                 + 1.16817 w_k1_FDX_s006
                                 + 0.727906 w_k1_IBM_s006
                                 + 1.1533 w_k1_INTC_s006
                                 + 0.806406 w_k1_KO_s006
                                 + 0.662841 w_k1_MAT_s006
                                 + 0.944177 w_k1_MSFT_s006
                                 + 1.05098 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.1444 w_k1_ADM_s006
                                 + 1.1943 w_k1_AXP_s006 + 1.62979 w_k1_DIS_s006
                                 + 1.29458 w_k1_FDX_s006
                                 + 1.15661 w_k1_IBM_s006
                                 + 1.31397 w_k1_INTC_s006
                                 + 1.12574 w_k1_KO_s006 + 1.03277 w_k1_MAT_s006
                                 + 1.12945 w_k1_MSFT_s006
                                 + 1.18502 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.904125 w_k1_ADM_s006
                                 + 0.854546 w_k1_AXP_s006
                                 + 1.10255 w_k1_DIS_s006
                                 + 0.935982 w_k1_FDX_s006
                                 + 1.24543 w_k1_IBM_s006
                                 + 1.04291 w_k1_INTC_s006
                                 + 1.01224 w_k1_KO_s006
                                 + 0.801406 w_k1_MAT_s006
                                 + 0.982315 w_k1_MSFT_s006
                                 + 1.10157 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.09309 w_k1_ADM_s006
                                 + 1.13176 w_k1_AXP_s006
                                 + 1.04319 w_k1_DIS_s006
                                 + 1.14998 w_k1_FDX_s006
                                 + 1.10279 w_k1_IBM_s006
                                 + 0.961414 w_k1_INTC_s006
                                 + 1.01526 w_k1_KO_s006 + 1.23575 w_k1_MAT_s006
                                 + 0.956225 w_k1_MSFT_s006
                                 + 1.12245 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.20983 w_k1_ADM_s006
                                 + 1.2427 w_k1_AXP_s006 + 1.43283 w_k1_DIS_s006
                                 + 1.08067 w_k1_FDX_s006
                                 + 1.68662 w_k1_IBM_s006
                                 + 1.04432 w_k1_INTC_s006
                                 + 1.40494 w_k1_KO_s006 + 1.35661 w_k1_MAT_s006
                                 + 1.01144 w_k1_MSFT_s006
                                 + 1.27158 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.962321 w_k1_ADM_s006
                                 + 0.737158 w_k1_AXP_s006
                                 + 0.740371 w_k1_DIS_s006
                                 + 0.888912 w_k1_FDX_s006
                                 + 0.723896 w_k1_IBM_s006
                                 + 0.881798 w_k1_INTC_s006
                                 + 0.75637 w_k1_KO_s006
                                 + 0.560733 w_k1_MAT_s006
                                 + 0.799451 w_k1_MSFT_s006
                                 + 0.834576 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.826469 w_k1_ADM_s006
                                 + 1.00324 w_k1_AXP_s006
                                 + 0.827813 w_k1_DIS_s006
                                 + 0.885904 w_k1_FDX_s006
                                 + 1.26788 w_k1_IBM_s006
                                 + 0.829579 w_k1_INTC_s006
                                 + 0.901179 w_k1_KO_s006
                                 + 1.24088 w_k1_MAT_s006
                                 + 0.854312 w_k1_MSFT_s006
                                 + 1.07333 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.725313 w_k1_ADM_s006
                                 + 0.734792 w_k1_AXP_s006
                                 + 0.64905 w_k1_DIS_s006
                                 + 0.712391 w_k1_FDX_s006
                                 + 0.903753 w_k1_IBM_s006
                                 + 0.74594 w_k1_INTC_s006
                                 + 0.727704 w_k1_KO_s006
                                 + 0.51463 w_k1_MAT_s006
                                 + 0.69087 w_k1_MSFT_s006
                                 + 0.837318 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.679927 w_k1_ADM_s007
                                 + 0.669644 w_k1_AXP_s007
                                 + 0.601407 w_k1_DIS_s007
                                 + 0.603051 w_k1_FDX_s007
                                 + 0.995602 w_k1_IBM_s007
                                 + 0.795018 w_k1_INTC_s007
                                 + 0.843347 w_k1_KO_s007
                                 + 0.932936 w_k1_MAT_s007
                                 + 0.656609 w_k1_MSFT_s007
                                 + 0.858186 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.17982 w_k1_ADM_s007
                                 + 1.17896 w_k1_AXP_s007
                                 + 1.16056 w_k1_DIS_s007
                                 + 1.27933 w_k1_FDX_s007
                                 + 0.913329 w_k1_IBM_s007
                                 + 1.23183 w_k1_INTC_s007
                                 + 1.09396 w_k1_KO_s007
                                 + 0.939176 w_k1_MAT_s007
                                 + 1.139 w_k1_MSFT_s007 + 1.12835 w_k1_PG_s007
                                 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.03445 w_k1_ADM_s007
                                 + 1.39427 w_k1_AXP_s007 + 1.0969 w_k1_DIS_s007
                                 + 1.06048 w_k1_FDX_s007
                                 + 1.08423 w_k1_IBM_s007
                                 + 0.997198 w_k1_INTC_s007
                                 + 0.939473 w_k1_KO_s007
                                 + 1.33343 w_k1_MAT_s007
                                 + 0.932942 w_k1_MSFT_s007
                                 + 0.974153 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.824575 w_k1_ADM_s007
                                 + 0.923463 w_k1_AXP_s007
                                 + 0.81937 w_k1_DIS_s007
                                 + 0.915573 w_k1_FDX_s007
                                 + 1.05781 w_k1_IBM_s007
                                 + 1.06355 w_k1_INTC_s007
                                 + 0.992979 w_k1_KO_s007
                                 + 1.12753 w_k1_MAT_s007
                                 + 0.905436 w_k1_MSFT_s007
                                 + 0.940581 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.960154 w_k1_ADM_s007
                                 + 0.863656 w_k1_AXP_s007
                                 + 1.36557 w_k1_DIS_s007
                                 + 1.01081 w_k1_FDX_s007
                                 + 0.962794 w_k1_IBM_s007
                                 + 1.19002 w_k1_INTC_s007 + 1.1939 w_k1_KO_s007
                                 + 1.05717 w_k1_MAT_s007
                                 + 1.1494 w_k1_MSFT_s007 + 1.06745 w_k1_PG_s007
                                 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.89392 w_k1_ADM_s007
                                 + 0.827131 w_k1_AXP_s007
                                 + 0.716546 w_k1_DIS_s007
                                 + 0.800114 w_k1_FDX_s007
                                 + 0.814618 w_k1_IBM_s007
                                 + 0.813246 w_k1_INTC_s007
                                 + 0.805111 w_k1_KO_s007
                                 + 0.662708 w_k1_MAT_s007
                                 + 0.824115 w_k1_MSFT_s007
                                 + 0.786987 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.10993 w_k1_ADM_s007
                                 + 0.843919 w_k1_AXP_s007
                                 + 1.36284 w_k1_DIS_s007
                                 + 0.955928 w_k1_FDX_s007
                                 + 1.46689 w_k1_IBM_s007
                                 + 1.05555 w_k1_INTC_s007
                                 + 1.34423 w_k1_KO_s007
                                 + 0.878919 w_k1_MAT_s007
                                 + 0.920875 w_k1_MSFT_s007
                                 + 1.06598 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.01371 w_k1_ADM_s007
                                 + 1.30956 w_k1_AXP_s007
                                 + 1.11454 w_k1_DIS_s007
                                 + 1.05309 w_k1_FDX_s007
                                 + 1.41711 w_k1_IBM_s007
                                 + 0.897586 w_k1_INTC_s007
                                 + 1.11359 w_k1_KO_s007 + 1.97066 w_k1_MAT_s007
                                 + 0.912422 w_k1_MSFT_s007
                                 + 1.18974 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.06368 w_k1_ADM_s010
                                 + 1.11552 w_k1_AXP_s010
                                 + 0.920046 w_k1_DIS_s010
                                 + 1.11154 w_k1_FDX_s010
                                 + 1.14807 w_k1_IBM_s010
                                 + 0.942853 w_k1_INTC_s010
                                 + 1.14578 w_k1_KO_s010 + 1.54889 w_k1_MAT_s010
                                 + 1.12185 w_k1_MSFT_s010
                                 + 1.06575 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.44633 w_k1_ADM_s010
                                 + 1.64127 w_k1_AXP_s010
                                 + 1.53025 w_k1_DIS_s010
                                 + 1.28547 w_k1_FDX_s010
                                 + 1.30978 w_k1_IBM_s010
                                 + 1.18749 w_k1_INTC_s010
                                 + 1.30528 w_k1_KO_s010 + 1.672 w_k1_MAT_s010
                                 + 1.21039 w_k1_MSFT_s010
                                 + 1.37385 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.763649 w_k1_ADM_s010
                                 + 0.856978 w_k1_AXP_s010
                                 + 0.799991 w_k1_DIS_s010
                                 + 0.87462 w_k1_FDX_s010
                                 + 0.933632 w_k1_IBM_s010
                                 + 0.859601 w_k1_INTC_s010
                                 + 0.848651 w_k1_KO_s010
                                 + 1.15794 w_k1_MAT_s010
                                 + 0.981769 w_k1_MSFT_s010
                                 + 0.901666 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.15311 w_k1_ADM_s010
                                 + 1.10294 w_k1_AXP_s010 + 1.3579 w_k1_DIS_s010
                                 + 1.20854 w_k1_FDX_s010
                                 + 0.848847 w_k1_IBM_s010
                                 + 1.10609 w_k1_INTC_s010
                                 + 1.11914 w_k1_KO_s010 + 1.06889 w_k1_MAT_s010
                                 + 1.50819 w_k1_MSFT_s010
                                 + 1.20096 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.09031 w_k1_ADM_s010
                                 + 0.990517 w_k1_AXP_s010
                                 + 1.15916 w_k1_DIS_s010
                                 + 1.11889 w_k1_FDX_s010
                                 + 1.30481 w_k1_IBM_s010
                                 + 1.1926 w_k1_INTC_s010 + 1.211 w_k1_KO_s010
                                 + 1.66626 w_k1_MAT_s010
                                 + 1.40965 w_k1_MSFT_s010
                                 + 1.24641 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.833168 w_k1_ADM_s010
                                 + 1.15678 w_k1_AXP_s010
                                 + 0.972292 w_k1_DIS_s010
                                 + 1.09476 w_k1_FDX_s010
                                 + 0.790838 w_k1_IBM_s010
                                 + 1.07933 w_k1_INTC_s010
                                 + 0.911405 w_k1_KO_s010
                                 + 0.809989 w_k1_MAT_s010
                                 + 1.02754 w_k1_MSFT_s010
                                 + 1.09864 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.898293 w_k1_ADM_s010
                                 + 1.12168 w_k1_AXP_s010
                                 + 0.938686 w_k1_DIS_s010
                                 + 0.880697 w_k1_FDX_s010
                                 + 1.51481 w_k1_IBM_s010
                                 + 1.03053 w_k1_INTC_s010
                                 + 1.21498 w_k1_KO_s010 + 1.37594 w_k1_MAT_s010
                                 + 0.965623 w_k1_MSFT_s010
                                 + 1.08061 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.803775 w_k1_ADM_s010
                                 + 0.73671 w_k1_AXP_s010
                                 + 0.778603 w_k1_DIS_s010
                                 + 0.861849 w_k1_FDX_s010
                                 + 0.636114 w_k1_IBM_s010
                                 + 0.868473 w_k1_INTC_s010
                                 + 0.855534 w_k1_KO_s010
                                 + 0.584152 w_k1_MAT_s010
                                 + 0.932762 w_k1_MSFT_s010
                                 + 0.934334 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.00088 w_k1_ADM_s011
                                 + 1.44222 w_k1_AXP_s011
                                 + 1.04721 w_k1_DIS_s011
                                 + 1.14459 w_k1_FDX_s011
                                 + 1.10219 w_k1_IBM_s011
                                 + 0.980833 w_k1_INTC_s011
                                 + 1.14697 w_k1_KO_s011 + 1.50924 w_k1_MAT_s011
                                 + 0.97249 w_k1_MSFT_s011
                                 + 1.07015 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 1.31041 w_k1_ADM_s011
                                 + 1.68731 w_k1_AXP_s011
                                 + 1.66459 w_k1_DIS_s011
                                 + 1.32877 w_k1_FDX_s011
                                 + 1.52344 w_k1_IBM_s011
                                 + 1.22031 w_k1_INTC_s011 + 1.6665 w_k1_KO_s011
                                 + 2.23663 w_k1_MAT_s011
                                 + 1.34788 w_k1_MSFT_s011
                                 + 1.26198 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.897774 w_k1_ADM_s011
                                 + 0.985459 w_k1_AXP_s011
                                 + 1.23269 w_k1_DIS_s011
                                 + 1.09673 w_k1_FDX_s011
                                 + 0.951428 w_k1_IBM_s011
                                 + 1.12951 w_k1_INTC_s011
                                 + 1.12443 w_k1_KO_s011 + 1.00703 w_k1_MAT_s011
                                 + 1.3416 w_k1_MSFT_s011
                                 + 0.977535 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.914771 w_k1_ADM_s011
                                 + 0.640085 w_k1_AXP_s011
                                 + 0.894452 w_k1_DIS_s011
                                 + 0.895081 w_k1_FDX_s011
                                 + 0.651891 w_k1_IBM_s011
                                 + 0.755228 w_k1_INTC_s011
                                 + 0.865656 w_k1_KO_s011
                                 + 0.572934 w_k1_MAT_s011
                                 + 0.8725 w_k1_MSFT_s011
                                 + 0.865107 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.04895 w_k1_ADM_s011
                                 + 0.906691 w_k1_AXP_s011
                                 + 1.34309 w_k1_DIS_s011 + 1.3611 w_k1_FDX_s011
                                 + 1.09824 w_k1_IBM_s011
                                 + 1.13979 w_k1_INTC_s011
                                 + 1.20046 w_k1_KO_s011 + 0.95161 w_k1_MAT_s011
                                 + 1.16235 w_k1_MSFT_s011
                                 + 1.00139 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.93522 w_k1_ADM_s011
                                 + 0.725617 w_k1_AXP_s011
                                 + 0.752101 w_k1_DIS_s011
                                 + 0.9081 w_k1_FDX_s011
                                 + 0.891365 w_k1_IBM_s011
                                 + 0.905898 w_k1_INTC_s011
                                 + 0.84939 w_k1_KO_s011
                                 + 0.599802 w_k1_MAT_s011
                                 + 0.823675 w_k1_MSFT_s011
                                 + 0.830189 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.907563 w_k1_ADM_s011
                                 + 0.820795 w_k1_AXP_s011
                                 + 0.875232 w_k1_DIS_s011
                                 + 0.993153 w_k1_FDX_s011
                                 + 1.14629 w_k1_IBM_s011
                                 + 0.930715 w_k1_INTC_s011
                                 + 1.03613 w_k1_KO_s011 + 1.05531 w_k1_MAT_s011
                                 + 0.939855 w_k1_MSFT_s011
                                 + 0.991056 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.918267 w_k1_ADM_s011
                                 + 0.816278 w_k1_AXP_s011
                                 + 1.24417 w_k1_DIS_s011
                                 + 1.05809 w_k1_FDX_s011 + 1.3868 w_k1_IBM_s011
                                 + 1.07441 w_k1_INTC_s011
                                 + 1.49208 w_k1_KO_s011 + 1.18206 w_k1_MAT_s011
                                 + 1.11045 w_k1_MSFT_s011
                                 + 1.01045 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.891387 w_k1_ADM_s012
                                 + 1.68055 w_k1_AXP_s012
                                 + 1.41445 w_k1_DIS_s012
                                 + 1.28269 w_k1_FDX_s012
                                 + 1.81043 w_k1_IBM_s012
                                 + 1.38735 w_k1_INTC_s012
                                 + 1.38913 w_k1_KO_s012 + 1.96662 w_k1_MAT_s012
                                 + 1.39381 w_k1_MSFT_s012 + 1.2977 w_k1_PG_s012
                                 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.870175 w_k1_ADM_s012
                                 + 1.09966 w_k1_AXP_s012
                                 + 1.13961 w_k1_DIS_s012
                                 + 1.19957 w_k1_FDX_s012
                                 + 1.21987 w_k1_IBM_s012
                                 + 1.47856 w_k1_INTC_s012
                                 + 1.13073 w_k1_KO_s012 + 1.08421 w_k1_MAT_s012
                                 + 1.33495 w_k1_MSFT_s012
                                 + 1.16607 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.02324 w_k1_ADM_s012
                                 + 1.17692 w_k1_AXP_s012
                                 + 0.903639 w_k1_DIS_s012
                                 + 1.0856 w_k1_FDX_s012
                                 + 0.843026 w_k1_IBM_s012
                                 + 1.23329 w_k1_INTC_s012
                                 + 0.879181 w_k1_KO_s012
                                 + 0.695271 w_k1_MAT_s012
                                 + 1.02484 w_k1_MSFT_s012
                                 + 1.05356 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.997208 w_k1_ADM_s012
                                 + 1.08426 w_k1_AXP_s012
                                 + 1.43634 w_k1_DIS_s012
                                 + 1.35016 w_k1_FDX_s012
                                 + 0.937169 w_k1_IBM_s012
                                 + 1.32404 w_k1_INTC_s012
                                 + 1.33129 w_k1_KO_s012 + 1.24163 w_k1_MAT_s012
                                 + 1.52429 w_k1_MSFT_s012
                                 + 1.12002 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.821562 w_k1_ADM_s012
                                 + 0.73529 w_k1_AXP_s012
                                 + 0.883554 w_k1_DIS_s012
                                 + 0.994559 w_k1_FDX_s012
                                 + 0.652262 w_k1_IBM_s012
                                 + 1.14539 w_k1_INTC_s012
                                 + 0.839551 w_k1_KO_s012
                                 + 0.637192 w_k1_MAT_s012
                                 + 1.07997 w_k1_MSFT_s012
                                 + 0.882489 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.758264 w_k1_ADM_s012
                                 + 0.831222 w_k1_AXP_s012
                                 + 0.908695 w_k1_DIS_s012
                                 + 0.940132 w_k1_FDX_s012
                                 + 0.904748 w_k1_IBM_s012
                                 + 0.968616 w_k1_INTC_s012
                                 + 0.878585 w_k1_KO_s012
                                 + 0.883677 w_k1_MAT_s012
                                 + 0.926468 w_k1_MSFT_s012
                                 + 0.885293 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.879319 w_k1_ADM_s012
                                 + 0.848063 w_k1_AXP_s012
                                 + 1.06715 w_k1_DIS_s012
                                 + 0.886137 w_k1_FDX_s012
                                 + 1.34108 w_k1_IBM_s012
                                 + 1.10937 w_k1_INTC_s012
                                 + 1.17696 w_k1_KO_s012 + 1.16335 w_k1_MAT_s012
                                 + 0.924784 w_k1_MSFT_s012
                                 + 0.947186 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.700526 w_k1_ADM_s012
                                 + 0.954689 w_k1_AXP_s012
                                 + 0.615205 w_k1_DIS_s012
                                 + 0.851724 w_k1_FDX_s012
                                 + 1.05084 w_k1_IBM_s012
                                 + 0.950133 w_k1_INTC_s012
                                 + 0.765123 w_k1_KO_s012
                                 + 1.14716 w_k1_MAT_s012
                                 + 0.734983 w_k1_MSFT_s012
                                 + 0.824048 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.0158 w_k1_ADM_s013
                                 + 1.2865 w_k1_AXP_s013 + 1.21101 w_k1_DIS_s013
                                 + 1.14047 w_k1_FDX_s013
                                 + 1.04811 w_k1_IBM_s013
                                 + 1.04899 w_k1_INTC_s013
                                 + 1.19287 w_k1_KO_s013 + 1.10914 w_k1_MAT_s013
                                 + 0.944803 w_k1_MSFT_s013
                                 + 0.9631 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.98977 w_k1_ADM_s013
                                 + 1.01071 w_k1_AXP_s013
                                 + 1.15275 w_k1_DIS_s013
                                 + 0.973363 w_k1_FDX_s013
                                 + 1.03236 w_k1_IBM_s013
                                 + 1.11322 w_k1_INTC_s013
                                 + 1.21015 w_k1_KO_s013
                                 + 0.908239 w_k1_MAT_s013
                                 + 0.95368 w_k1_MSFT_s013
                                 + 1.08192 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.933931 w_k1_ADM_s013
                                 + 1.17791 w_k1_AXP_s013 + 1.1124 w_k1_DIS_s013
                                 + 1.22401 w_k1_FDX_s013
                                 + 0.789368 w_k1_IBM_s013
                                 + 1.29079 w_k1_INTC_s013
                                 + 0.959964 w_k1_KO_s013
                                 + 0.664261 w_k1_MAT_s013
                                 + 1.07496 w_k1_MSFT_s013
                                 + 1.07766 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.704345 w_k1_ADM_s013
                                 + 0.979589 w_k1_AXP_s013
                                 + 0.737443 w_k1_DIS_s013
                                 + 0.848358 w_k1_FDX_s013
                                 + 0.829116 w_k1_IBM_s013
                                 + 0.834684 w_k1_INTC_s013
                                 + 0.795418 w_k1_KO_s013
                                 + 0.903838 w_k1_MAT_s013
                                 + 0.717229 w_k1_MSFT_s013
                                 + 0.893196 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.03865 w_k1_ADM_s013
                                 + 1.81453 w_k1_AXP_s013
                                 + 1.55167 w_k1_DIS_s013
                                 + 1.41874 w_k1_FDX_s013
                                 + 1.12562 w_k1_IBM_s013
                                 + 1.25987 w_k1_INTC_s013
                                 + 1.31548 w_k1_KO_s013 + 1.71401 w_k1_MAT_s013
                                 + 1.22584 w_k1_MSFT_s013
                                 + 1.32558 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.654761 w_k1_ADM_s013
                                 + 0.698016 w_k1_AXP_s013
                                 + 0.919017 w_k1_DIS_s013
                                 + 0.775438 w_k1_FDX_s013
                                 + 1.18001 w_k1_IBM_s013
                                 + 0.908809 w_k1_INTC_s013
                                 + 1.0101 w_k1_KO_s013 + 0.927672 w_k1_MAT_s013
                                 + 0.846769 w_k1_MSFT_s013
                                 + 0.873115 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.0867 w_k1_ADM_s013
                                 + 1.10752 w_k1_AXP_s013
                                 + 1.34342 w_k1_DIS_s013
                                 + 1.13434 w_k1_FDX_s013
                                 + 1.37757 w_k1_IBM_s013
                                 + 1.24561 w_k1_INTC_s013
                                 + 1.46689 w_k1_KO_s013 + 1.12443 w_k1_MAT_s013
                                 + 1.06794 w_k1_MSFT_s013
                                 + 1.23463 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.12218 w_k1_ADM_s013
                                 + 1.82513 w_k1_AXP_s013
                                 + 1.28306 w_k1_DIS_s013
                                 + 1.12374 w_k1_FDX_s013
                                 + 1.63298 w_k1_IBM_s013
                                 + 1.18204 w_k1_INTC_s013
                                 + 1.18805 w_k1_KO_s013 + 1.46464 w_k1_MAT_s013
                                 + 1.13613 w_k1_MSFT_s013 + 1.0999 w_k1_PG_s013
                                 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.08353 w_k1_ADM_s014
                                 + 0.890309 w_k1_AXP_s014
                                 + 0.777889 w_k1_DIS_s014
                                 + 0.899553 w_k1_FDX_s014
                                 + 1.24104 w_k1_IBM_s014
                                 + 0.728265 w_k1_INTC_s014
                                 + 0.903374 w_k1_KO_s014
                                 + 0.97264 w_k1_MAT_s014
                                 + 0.851204 w_k1_MSFT_s014
                                 + 0.94051 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.919474 w_k1_ADM_s014
                                 + 0.657082 w_k1_AXP_s014
                                 + 0.601074 w_k1_DIS_s014
                                 + 0.753209 w_k1_FDX_s014
                                 + 0.855171 w_k1_IBM_s014
                                 + 0.563342 w_k1_INTC_s014
                                 + 0.683879 w_k1_KO_s014
                                 + 0.983061 w_k1_MAT_s014
                                 + 0.752452 w_k1_MSFT_s014
                                 + 0.817575 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.14528 w_k1_ADM_s014
                                 + 0.819317 w_k1_AXP_s014
                                 + 0.920483 w_k1_DIS_s014
                                 + 0.937615 w_k1_FDX_s014
                                 + 0.691702 w_k1_IBM_s014
                                 + 0.717445 w_k1_INTC_s014
                                 + 0.748126 w_k1_KO_s014
                                 + 0.79643 w_k1_MAT_s014
                                 + 0.923381 w_k1_MSFT_s014
                                 + 0.944436 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.22132 w_k1_ADM_s014
                                 + 0.977323 w_k1_AXP_s014
                                 + 1.05025 w_k1_DIS_s014
                                 + 0.93551 w_k1_FDX_s014
                                 + 1.13629 w_k1_IBM_s014
                                 + 0.763096 w_k1_INTC_s014
                                 + 1.10323 w_k1_KO_s014
                                 + 0.817314 w_k1_MAT_s014
                                 + 0.91675 w_k1_MSFT_s014
                                 + 0.978537 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.3443 w_k1_ADM_s014
                                 + 1.09284 w_k1_AXP_s014
                                 + 1.17781 w_k1_DIS_s014
                                 + 1.20416 w_k1_FDX_s014
                                 + 0.867569 w_k1_IBM_s014
                                 + 0.892893 w_k1_INTC_s014
                                 + 0.887869 w_k1_KO_s014
                                 + 0.921125 w_k1_MAT_s014
                                 + 1.15567 w_k1_MSFT_s014
                                 + 0.980458 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.10968 w_k1_ADM_s014
                                 + 1.66945 w_k1_AXP_s014
                                 + 1.30156 w_k1_DIS_s014
                                 + 1.11536 w_k1_FDX_s014
                                 + 1.72557 w_k1_IBM_s014
                                 + 1.01506 w_k1_INTC_s014
                                 + 1.24464 w_k1_KO_s014 + 1.52583 w_k1_MAT_s014
                                 + 1.19725 w_k1_MSFT_s014 + 1.1074 w_k1_PG_s014
                                 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.00997 w_k1_ADM_s014
                                 + 0.842887 w_k1_AXP_s014
                                 + 0.830983 w_k1_DIS_s014
                                 + 0.759936 w_k1_FDX_s014
                                 + 0.994937 w_k1_IBM_s014
                                 + 0.905967 w_k1_INTC_s014
                                 + 0.808499 w_k1_KO_s014
                                 + 0.487823 w_k1_MAT_s014
                                 + 0.911384 w_k1_MSFT_s014
                                 + 0.779144 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.34322 w_k1_ADM_s014
                                 + 1.249 w_k1_AXP_s014 + 1.61697 w_k1_DIS_s014
                                 + 1.20714 w_k1_FDX_s014
                                 + 1.06164 w_k1_IBM_s014
                                 + 1.04107 w_k1_INTC_s014
                                 + 1.29992 w_k1_KO_s014 + 0.8313 w_k1_MAT_s014
                                 + 1.34627 w_k1_MSFT_s014
                                 + 1.14538 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.08496 w_k1_ADM_s015
                                 + 0.643639 w_k1_AXP_s015
                                 + 0.737653 w_k1_DIS_s015
                                 + 0.732198 w_k1_FDX_s015
                                 + 0.841391 w_k1_IBM_s015
                                 + 0.662161 w_k1_INTC_s015
                                 + 0.800243 w_k1_KO_s015
                                 + 0.883481 w_k1_MAT_s015
                                 + 0.704985 w_k1_MSFT_s015
                                 + 0.80053 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.17481 w_k1_ADM_s015
                                 + 0.7946 w_k1_AXP_s015 + 1.23123 w_k1_DIS_s015
                                 + 1.01261 w_k1_FDX_s015 + 1.2345 w_k1_IBM_s015
                                 + 1.08046 w_k1_INTC_s015 + 1.0347 w_k1_KO_s015
                                 + 0.921834 w_k1_MAT_s015
                                 + 1.02175 w_k1_MSFT_s015
                                 + 0.99408 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.962908 w_k1_ADM_s015
                                 + 0.84125 w_k1_AXP_s015
                                 + 1.06802 w_k1_DIS_s015 + 1.0068 w_k1_FDX_s015
                                 + 0.704275 w_k1_IBM_s015
                                 + 0.834224 w_k1_INTC_s015
                                 + 0.78927 w_k1_KO_s015
                                 + 0.815941 w_k1_MAT_s015
                                 + 0.861964 w_k1_MSFT_s015
                                 + 0.851343 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.17255 w_k1_ADM_s015
                                 + 0.997836 w_k1_AXP_s015
                                 + 1.28098 w_k1_DIS_s015
                                 + 0.974335 w_k1_FDX_s015
                                 + 1.38321 w_k1_IBM_s015
                                 + 0.859677 w_k1_INTC_s015
                                 + 1.2645 w_k1_KO_s015 + 1.89142 w_k1_MAT_s015
                                 + 1.03145 w_k1_MSFT_s015
                                 + 1.17771 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.921359 w_k1_ADM_s015
                                 + 0.787695 w_k1_AXP_s015
                                 + 1.31446 w_k1_DIS_s015
                                 + 0.874957 w_k1_FDX_s015
                                 + 1.69604 w_k1_IBM_s015
                                 + 0.847318 w_k1_INTC_s015
                                 + 1.19473 w_k1_KO_s015 + 1.51272 w_k1_MAT_s015
                                 + 0.889311 w_k1_MSFT_s015
                                 + 0.89355 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.970901 w_k1_ADM_s015
                                 + 0.801313 w_k1_AXP_s015
                                 + 1.0154 w_k1_DIS_s015
                                 + 0.937876 w_k1_FDX_s015
                                 + 1.00138 w_k1_IBM_s015
                                 + 0.792454 w_k1_INTC_s015
                                 + 0.948288 w_k1_KO_s015
                                 + 1.27264 w_k1_MAT_s015
                                 + 0.895984 w_k1_MSFT_s015
                                 + 1.04038 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.55009 w_k1_ADM_s015
                                 + 1.33558 w_k1_AXP_s015
                                 + 1.99405 w_k1_DIS_s015
                                 + 1.46083 w_k1_FDX_s015
                                 + 1.29668 w_k1_IBM_s015
                                 + 1.21877 w_k1_INTC_s015
                                 + 1.49217 w_k1_KO_s015 + 1.55162 w_k1_MAT_s015
                                 + 1.36494 w_k1_MSFT_s015
                                 + 1.21101 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.18761 w_k1_ADM_s015
                                 + 1.13646 w_k1_AXP_s015 + 1.6444 w_k1_DIS_s015
                                 + 1.28921 w_k1_FDX_s015
                                 + 0.911694 w_k1_IBM_s015
                                 + 1.29096 w_k1_INTC_s015
                                 + 1.13761 w_k1_KO_s015 + 1.01371 w_k1_MAT_s015
                                 + 1.4532 w_k1_MSFT_s015 + 1.10346 w_k1_PG_s015
                                 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.19344 w_k1_ADM_s016
                                 + 1.54754 w_k1_AXP_s016
                                 + 1.24431 w_k1_DIS_s016
                                 + 1.19376 w_k1_FDX_s016
                                 + 1.45204 w_k1_IBM_s016
                                 + 1.02706 w_k1_INTC_s016
                                 + 1.32596 w_k1_KO_s016 + 2.14851 w_k1_MAT_s016
                                 + 1.0324 w_k1_MSFT_s016 + 1.0232 w_k1_PG_s016
                                 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.58152 w_k1_ADM_s016
                                 + 1.37341 w_k1_AXP_s016
                                 + 1.45464 w_k1_DIS_s016
                                 + 1.40639 w_k1_FDX_s016
                                 + 0.899675 w_k1_IBM_s016
                                 + 1.06392 w_k1_INTC_s016
                                 + 1.25565 w_k1_KO_s016 + 1.32067 w_k1_MAT_s016
                                 + 1.19453 w_k1_MSFT_s016
                                 + 1.19388 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.857588 w_k1_ADM_s016
                                 + 0.662493 w_k1_AXP_s016
                                 + 0.601936 w_k1_DIS_s016
                                 + 0.774492 w_k1_FDX_s016
                                 + 0.893378 w_k1_IBM_s016
                                 + 0.674491 w_k1_INTC_s016
                                 + 0.866286 w_k1_KO_s016
                                 + 1.18361 w_k1_MAT_s016
                                 + 0.819704 w_k1_MSFT_s016
                                 + 0.817283 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.896364 w_k1_ADM_s016
                                 + 0.804321 w_k1_AXP_s016
                                 + 0.723212 w_k1_DIS_s016
                                 + 0.950729 w_k1_FDX_s016
                                 + 0.658259 w_k1_IBM_s016
                                 + 0.719128 w_k1_INTC_s016
                                 + 0.816591 w_k1_KO_s016 + 1.1209 w_k1_MAT_s016
                                 + 1.01663 w_k1_MSFT_s016
                                 + 0.923836 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.10537 w_k1_ADM_s016
                                 + 0.98015 w_k1_AXP_s016
                                 + 0.952109 w_k1_DIS_s016
                                 + 0.97525 w_k1_FDX_s016
                                 + 1.08832 w_k1_IBM_s016
                                 + 0.921944 w_k1_INTC_s016
                                 + 1.17697 w_k1_KO_s016 + 1.37238 w_k1_MAT_s016
                                 + 1.04102 w_k1_MSFT_s016
                                 + 1.04418 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.25798 w_k1_ADM_s016
                                 + 0.941312 w_k1_AXP_s016
                                 + 1.46493 w_k1_DIS_s016
                                 + 1.15672 w_k1_FDX_s016
                                 + 1.45626 w_k1_IBM_s016
                                 + 1.08287 w_k1_INTC_s016
                                 + 1.41315 w_k1_KO_s016 + 1.08272 w_k1_MAT_s016
                                 + 1.24679 w_k1_MSFT_s016
                                 + 1.20717 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.32382 w_k1_ADM_s016
                                 + 1.21016 w_k1_AXP_s016
                                 + 1.71468 w_k1_DIS_s016
                                 + 1.39286 w_k1_FDX_s016
                                 + 1.08449 w_k1_IBM_s016
                                 + 1.30216 w_k1_INTC_s016
                                 + 1.46837 w_k1_KO_s016 + 1.14439 w_k1_MAT_s016
                                 + 1.70435 w_k1_MSFT_s016
                                 + 1.24123 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.29766 w_k1_ADM_s016
                                 + 1.20177 w_k1_AXP_s016
                                 + 1.09674 w_k1_DIS_s016
                                 + 1.12419 w_k1_FDX_s016
                                 + 0.805955 w_k1_IBM_s016
                                 + 1.10476 w_k1_INTC_s016
                                 + 1.03143 w_k1_KO_s016
                                 + 0.837028 w_k1_MAT_s016
                                 + 1.14415 w_k1_MSFT_s016
                                 + 0.979179 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.08831 w_k1_ADM_s017
                                 + 0.681772 w_k1_AXP_s017
                                 + 0.792385 w_k1_DIS_s017
                                 + 0.875825 w_k1_FDX_s017
                                 + 0.755485 w_k1_IBM_s017
                                 + 0.880468 w_k1_INTC_s017
                                 + 0.834157 w_k1_KO_s017
                                 + 0.382962 w_k1_MAT_s017
                                 + 0.893281 w_k1_MSFT_s017
                                 + 0.876195 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.56727 w_k1_ADM_s017
                                 + 2.26972 w_k1_AXP_s017
                                 + 1.50825 w_k1_DIS_s017
                                 + 1.33808 w_k1_FDX_s017
                                 + 2.03813 w_k1_IBM_s017
                                 + 1.53335 w_k1_INTC_s017
                                 + 1.69406 w_k1_KO_s017 + 2.7747 w_k1_MAT_s017
                                 + 1.23248 w_k1_MSFT_s017
                                 + 1.33319 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.36105 w_k1_ADM_s017
                                 + 1.42698 w_k1_AXP_s017
                                 + 1.36575 w_k1_DIS_s017
                                 + 1.22762 w_k1_FDX_s017
                                 + 1.07301 w_k1_IBM_s017
                                 + 1.19307 w_k1_INTC_s017
                                 + 1.26726 w_k1_KO_s017 + 1.26057 w_k1_MAT_s017
                                 + 1.27703 w_k1_MSFT_s017
                                 + 1.31896 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.00772 w_k1_ADM_s017
                                 + 0.966944 w_k1_AXP_s017
                                 + 1.03873 w_k1_DIS_s017
                                 + 0.887694 w_k1_FDX_s017
                                 + 1.16894 w_k1_IBM_s017
                                 + 0.925128 w_k1_INTC_s017
                                 + 1.06691 w_k1_KO_s017
                                 + 0.914351 w_k1_MAT_s017
                                 + 0.907465 w_k1_MSFT_s017
                                 + 0.998536 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.999121 w_k1_ADM_s017
                                 + 0.912763 w_k1_AXP_s017
                                 + 0.702949 w_k1_DIS_s017
                                 + 0.829351 w_k1_FDX_s017
                                 + 0.944773 w_k1_IBM_s017
                                 + 0.89321 w_k1_INTC_s017
                                 + 0.847099 w_k1_KO_s017
                                 + 0.753386 w_k1_MAT_s017
                                 + 0.89373 w_k1_MSFT_s017
                                 + 0.906892 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.726844 w_k1_ADM_s017
                                 + 0.654896 w_k1_AXP_s017
                                 + 0.527187 w_k1_DIS_s017
                                 + 0.684975 w_k1_FDX_s017
                                 + 0.748643 w_k1_IBM_s017
                                 + 0.850994 w_k1_INTC_s017
                                 + 0.731387 w_k1_KO_s017
                                 + 0.777693 w_k1_MAT_s017
                                 + 0.817517 w_k1_MSFT_s017
                                 + 0.79573 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.16445 w_k1_ADM_s017
                                 + 1.19309 w_k1_AXP_s017 + 1.1399 w_k1_DIS_s017
                                 + 0.984326 w_k1_FDX_s017
                                 + 1.58089 w_k1_IBM_s017
                                 + 1.41034 w_k1_INTC_s017 + 1.4859 w_k1_KO_s017
                                 + 1.07434 w_k1_MAT_s017
                                 + 1.20509 w_k1_MSFT_s017
                                 + 1.16568 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.07281 w_k1_ADM_s017
                                 + 0.981418 w_k1_AXP_s017
                                 + 0.939675 w_k1_DIS_s017
                                 + 1.05909 w_k1_FDX_s017
                                 + 0.915624 w_k1_IBM_s017
                                 + 1.13831 w_k1_INTC_s017
                                 + 0.957476 w_k1_KO_s017
                                 + 0.832763 w_k1_MAT_s017
                                 + 0.98973 w_k1_MSFT_s017
                                 + 0.992489 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.04736 w_k1_ADM_s100
                                 + 1.16892 w_k1_AXP_s100
                                 + 1.31308 w_k1_DIS_s100
                                 + 1.43402 w_k1_FDX_s100
                                 + 0.834488 w_k1_IBM_s100
                                 + 1.45085 w_k1_INTC_s100
                                 + 0.992444 w_k1_KO_s100
                                 + 0.784234 w_k1_MAT_s100
                                 + 1.2772 w_k1_MSFT_s100 + 1.11605 w_k1_PG_s100
                                 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.851054 w_k1_ADM_s100
                                 + 0.905792 w_k1_AXP_s100
                                 + 0.906881 w_k1_DIS_s100
                                 + 0.965144 w_k1_FDX_s100
                                 + 1.05921 w_k1_IBM_s100
                                 + 1.05393 w_k1_INTC_s100
                                 + 0.841458 w_k1_KO_s100
                                 + 0.778331 w_k1_MAT_s100
                                 + 0.902177 w_k1_MSFT_s100
                                 + 0.902078 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.29174 w_k1_ADM_s100
                                 + 1.51446 w_k1_AXP_s100
                                 + 1.77551 w_k1_DIS_s100
                                 + 1.48221 w_k1_FDX_s100
                                 + 1.41681 w_k1_IBM_s100
                                 + 1.37407 w_k1_INTC_s100
                                 + 1.31985 w_k1_KO_s100 + 1.62649 w_k1_MAT_s100
                                 + 1.31652 w_k1_MSFT_s100
                                 + 1.17222 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.635633 w_k1_ADM_s100
                                 + 0.730885 w_k1_AXP_s100
                                 + 0.597257 w_k1_DIS_s100
                                 + 0.681088 w_k1_FDX_s100
                                 + 1.03373 w_k1_IBM_s100
                                 + 0.854183 w_k1_INTC_s100
                                 + 0.781941 w_k1_KO_s100
                                 + 0.79918 w_k1_MAT_s100
                                 + 0.735985 w_k1_MSFT_s100
                                 + 0.785103 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.894471 w_k1_ADM_s100
                                 + 0.871803 w_k1_AXP_s100
                                 + 0.957354 w_k1_DIS_s100
                                 + 1.00043 w_k1_FDX_s100
                                 + 0.732563 w_k1_IBM_s100
                                 + 0.995562 w_k1_INTC_s100
                                 + 0.873695 w_k1_KO_s100
                                 + 0.650946 w_k1_MAT_s100
                                 + 1.04187 w_k1_MSFT_s100
                                 + 0.870393 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.816083 w_k1_ADM_s100
                                 + 0.738224 w_k1_AXP_s100
                                 + 0.6354 w_k1_DIS_s100
                                 + 0.816255 w_k1_FDX_s100
                                 + 0.723825 w_k1_IBM_s100
                                 + 0.808337 w_k1_INTC_s100
                                 + 0.588694 w_k1_KO_s100
                                 + 0.520194 w_k1_MAT_s100
                                 + 0.765278 w_k1_MSFT_s100
                                 + 0.838721 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.16154 w_k1_ADM_s100
                                 + 1.18591 w_k1_AXP_s100
                                 + 1.44074 w_k1_DIS_s100
                                 + 1.19526 w_k1_FDX_s100
                                 + 1.12961 w_k1_IBM_s100
                                 + 1.37024 w_k1_INTC_s100
                                 + 1.21101 w_k1_KO_s100 + 1.11554 w_k1_MAT_s100
                                 + 1.25191 w_k1_MSFT_s100 + 1.1316 w_k1_PG_s100
                                 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.894724 w_k1_ADM_s100
                                 + 1.01138 w_k1_AXP_s100 + 1.0829 w_k1_DIS_s100
                                 + 0.97707 w_k1_FDX_s100
                                 + 1.49418 w_k1_IBM_s100
                                 + 1.1355 w_k1_INTC_s100 + 1.15174 w_k1_KO_s100
                                 + 1.40714 w_k1_MAT_s100
                                 + 0.956135 w_k1_MSFT_s100
                                 + 1.07436 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.792119 w_k1_ADM_s101
                                 + 1.04448 w_k1_AXP_s101
                                 + 1.23779 w_k1_DIS_s101
                                 + 1.03037 w_k1_FDX_s101
                                 + 0.902326 w_k1_IBM_s101
                                 + 0.961081 w_k1_INTC_s101
                                 + 0.965291 w_k1_KO_s101
                                 + 1.00334 w_k1_MAT_s101
                                 + 1.11408 w_k1_MSFT_s101
                                 + 1.09041 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.77894 w_k1_ADM_s101
                                 + 0.942323 w_k1_AXP_s101
                                 + 0.894045 w_k1_DIS_s101
                                 + 0.862717 w_k1_FDX_s101
                                 + 1.15938 w_k1_IBM_s101
                                 + 0.829634 w_k1_INTC_s101
                                 + 0.927302 w_k1_KO_s101 + 1.0893 w_k1_MAT_s101
                                 + 0.981374 w_k1_MSFT_s101
                                 + 0.978485 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.06792 w_k1_ADM_s101
                                 + 1.43081 w_k1_AXP_s101
                                 + 1.25911 w_k1_DIS_s101
                                 + 1.31947 w_k1_FDX_s101
                                 + 0.790908 w_k1_IBM_s101
                                 + 1.31375 w_k1_INTC_s101
                                 + 0.89071 w_k1_KO_s101 + 1.14811 w_k1_MAT_s101
                                 + 1.25395 w_k1_MSFT_s101
                                 + 1.21711 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.931446 w_k1_ADM_s101
                                 + 0.818499 w_k1_AXP_s101
                                 + 0.647694 w_k1_DIS_s101
                                 + 0.767762 w_k1_FDX_s101
                                 + 0.793638 w_k1_IBM_s101
                                 + 0.811288 w_k1_INTC_s101
                                 + 0.638931 w_k1_KO_s101
                                 + 0.700063 w_k1_MAT_s101
                                 + 0.695274 w_k1_MSFT_s101
                                 + 0.87147 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.18736 w_k1_ADM_s101
                                 + 1.17381 w_k1_AXP_s101
                                 + 1.54249 w_k1_DIS_s101
                                 + 1.26511 w_k1_FDX_s101
                                 + 0.828245 w_k1_IBM_s101
                                 + 1.17775 w_k1_INTC_s101
                                 + 0.944116 w_k1_KO_s101
                                 + 0.905874 w_k1_MAT_s101
                                 + 1.18111 w_k1_MSFT_s101
                                 + 1.06147 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.00889 w_k1_ADM_s101
                                 + 1.01873 w_k1_AXP_s101
                                 + 0.849311 w_k1_DIS_s101
                                 + 1.00265 w_k1_FDX_s101
                                 + 0.735847 w_k1_IBM_s101
                                 + 0.911034 w_k1_INTC_s101
                                 + 0.859343 w_k1_KO_s101
                                 + 0.907052 w_k1_MAT_s101
                                 + 0.949353 w_k1_MSFT_s101
                                 + 0.931434 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.33478 w_k1_ADM_s101
                                 + 1.5077 w_k1_AXP_s101 + 1.54482 w_k1_DIS_s101
                                 + 1.29498 w_k1_FDX_s101 + 1.4941 w_k1_IBM_s101
                                 + 1.1428 w_k1_INTC_s101 + 1.25053 w_k1_KO_s101
                                 + 1.58591 w_k1_MAT_s101
                                 + 1.20882 w_k1_MSFT_s101
                                 + 1.43194 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.16975 w_k1_ADM_s101
                                 + 1.36552 w_k1_AXP_s101
                                 + 0.951699 w_k1_DIS_s101
                                 + 1.01527 w_k1_FDX_s101
                                 + 1.16821 w_k1_IBM_s101
                                 + 0.870281 w_k1_INTC_s101
                                 + 0.892908 w_k1_KO_s101
                                 + 1.38185 w_k1_MAT_s101
                                 + 0.79048 w_k1_MSFT_s101
                                 + 1.11699 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.01211 w_k1_ADM_s102
                                 + 0.956901 w_k1_AXP_s102
                                 + 1.13879 w_k1_DIS_s102
                                 + 1.12193 w_k1_FDX_s102
                                 + 0.798265 w_k1_IBM_s102
                                 + 1.24594 w_k1_INTC_s102 + 1.0573 w_k1_KO_s102
                                 + 0.904923 w_k1_MAT_s102
                                 + 1.23038 w_k1_MSFT_s102
                                 + 0.942748 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.34214 w_k1_ADM_s102
                                 + 1.63477 w_k1_AXP_s102 + 1.6658 w_k1_DIS_s102
                                 + 1.63025 w_k1_FDX_s102
                                 + 0.965596 w_k1_IBM_s102
                                 + 1.59107 w_k1_INTC_s102
                                 + 1.09558 w_k1_KO_s102 + 1.26357 w_k1_MAT_s102
                                 + 1.46688 w_k1_MSFT_s102
                                 + 1.28328 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.17803 w_k1_ADM_s102
                                 + 1.25868 w_k1_AXP_s102
                                 + 1.46416 w_k1_DIS_s102 + 1.2952 w_k1_FDX_s102
                                 + 1.25412 w_k1_IBM_s102
                                 + 1.11652 w_k1_INTC_s102
                                 + 1.35851 w_k1_KO_s102 + 1.91181 w_k1_MAT_s102
                                 + 1.2525 w_k1_MSFT_s102 + 1.16442 w_k1_PG_s102
                                 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.86714 w_k1_ADM_s102
                                 + 0.591351 w_k1_AXP_s102
                                 + 0.773414 w_k1_DIS_s102
                                 + 0.826966 w_k1_FDX_s102
                                 + 0.818048 w_k1_IBM_s102
                                 + 0.942595 w_k1_INTC_s102
                                 + 0.815828 w_k1_KO_s102
                                 + 0.889593 w_k1_MAT_s102
                                 + 0.949479 w_k1_MSFT_s102
                                 + 0.958705 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.14413 w_k1_ADM_s102
                                 + 1.06934 w_k1_AXP_s102
                                 + 1.04121 w_k1_DIS_s102
                                 + 0.909856 w_k1_FDX_s102
                                 + 1.45649 w_k1_IBM_s102
                                 + 1.0471 w_k1_INTC_s102 + 1.15147 w_k1_KO_s102
                                 + 1.59559 w_k1_MAT_s102
                                 + 1.15794 w_k1_MSFT_s102
                                 + 1.13395 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.815537 w_k1_ADM_s102
                                 + 0.732553 w_k1_AXP_s102
                                 + 0.664668 w_k1_DIS_s102
                                 + 0.792521 w_k1_FDX_s102
                                 + 0.678777 w_k1_IBM_s102
                                 + 0.942717 w_k1_INTC_s102
                                 + 0.717886 w_k1_KO_s102
                                 + 0.774886 w_k1_MAT_s102
                                 + 0.781601 w_k1_MSFT_s102
                                 + 0.788465 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.70022 w_k1_ADM_s102
                                 + 0.668627 w_k1_AXP_s102
                                 + 0.734191 w_k1_DIS_s102
                                 + 0.727032 w_k1_FDX_s102
                                 + 0.988376 w_k1_IBM_s102
                                 + 0.859931 w_k1_INTC_s102
                                 + 0.847693 w_k1_KO_s102
                                 + 1.29519 w_k1_MAT_s102
                                 + 0.875475 w_k1_MSFT_s102
                                 + 0.833564 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.995443 w_k1_ADM_s102
                                 + 0.860456 w_k1_AXP_s102
                                 + 1.03078 w_k1_DIS_s102
                                 + 1.00106 w_k1_FDX_s102
                                 + 1.06296 w_k1_IBM_s102
                                 + 1.08752 w_k1_INTC_s102
                                 + 1.07567 w_k1_KO_s102 + 1.27509 w_k1_MAT_s102
                                 + 1.04451 w_k1_MSFT_s102
                                 + 1.02872 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.05172 w_k1_ADM_s103
                                 + 1.17477 w_k1_AXP_s103
                                 + 0.927042 w_k1_DIS_s103
                                 + 1.03543 w_k1_FDX_s103
                                 + 1.07091 w_k1_IBM_s103
                                 + 0.970494 w_k1_INTC_s103
                                 + 1.05061 w_k1_KO_s103 + 1.29897 w_k1_MAT_s103
                                 + 0.90354 w_k1_MSFT_s103
                                 + 1.12264 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.31597 w_k1_ADM_s103
                                 + 1.3679 w_k1_AXP_s103 + 1.47534 w_k1_DIS_s103
                                 + 1.28251 w_k1_FDX_s103
                                 + 1.52375 w_k1_IBM_s103
                                 + 1.46787 w_k1_INTC_s103
                                 + 1.53178 w_k1_KO_s103 + 1.75875 w_k1_MAT_s103
                                 + 1.28564 w_k1_MSFT_s103
                                 + 1.37285 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.975025 w_k1_ADM_s103
                                 + 0.889013 w_k1_AXP_s103
                                 + 0.787929 w_k1_DIS_s103
                                 + 0.882972 w_k1_FDX_s103
                                 + 1.47571 w_k1_IBM_s103
                                 + 0.908422 w_k1_INTC_s103
                                 + 1.1129 w_k1_KO_s103 + 1.61658 w_k1_MAT_s103
                                 + 0.898799 w_k1_MSFT_s103
                                 + 1.08971 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.0322 w_k1_ADM_s103
                                 + 0.775731 w_k1_AXP_s103
                                 + 0.75526 w_k1_DIS_s103
                                 + 0.905924 w_k1_FDX_s103
                                 + 1.0926 w_k1_IBM_s103
                                 + 0.705386 w_k1_INTC_s103
                                 + 0.918254 w_k1_KO_s103
                                 + 0.919419 w_k1_MAT_s103
                                 + 0.74275 w_k1_MSFT_s103
                                 + 1.03815 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.22334 w_k1_ADM_s103
                                 + 1.06481 w_k1_AXP_s103
                                 + 1.24427 w_k1_DIS_s103
                                 + 1.21633 w_k1_FDX_s103
                                 + 0.966198 w_k1_IBM_s103
                                 + 0.974786 w_k1_INTC_s103
                                 + 1.1361 w_k1_KO_s103 + 0.946484 w_k1_MAT_s103
                                 + 1.03607 w_k1_MSFT_s103
                                 + 1.12765 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.808844 w_k1_ADM_s103
                                 + 0.586605 w_k1_AXP_s103
                                 + 0.551956 w_k1_DIS_s103
                                 + 0.677509 w_k1_FDX_s103
                                 + 1.01926 w_k1_IBM_s103
                                 + 0.713019 w_k1_INTC_s103
                                 + 0.930002 w_k1_KO_s103
                                 + 0.954699 w_k1_MAT_s103
                                 + 0.809611 w_k1_MSFT_s103
                                 + 0.917852 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.770589 w_k1_ADM_s103
                                 + 0.547297 w_k1_AXP_s103
                                 + 0.562315 w_k1_DIS_s103 + 0.714 w_k1_FDX_s103
                                 + 0.753825 w_k1_IBM_s103
                                 + 0.739526 w_k1_INTC_s103
                                 + 0.737171 w_k1_KO_s103
                                 + 0.603686 w_k1_MAT_s103
                                 + 0.764042 w_k1_MSFT_s103
                                 + 0.763382 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.1018 w_k1_ADM_s103
                                 + 0.854159 w_k1_AXP_s103
                                 + 0.73099 w_k1_DIS_s103
                                 + 0.959744 w_k1_FDX_s103
                                 + 0.73463 w_k1_IBM_s103
                                 + 1.01633 w_k1_INTC_s103
                                 + 0.840494 w_k1_KO_s103
                                 + 0.652813 w_k1_MAT_s103
                                 + 0.822066 w_k1_MSFT_s103
                                 + 0.926252 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.39598 w_k1_ADM_s104
                                 + 1.04092 w_k1_AXP_s104
                                 + 1.29464 w_k1_DIS_s104 + 1.2624 w_k1_FDX_s104
                                 + 0.853152 w_k1_IBM_s104
                                 + 1.16533 w_k1_INTC_s104
                                 + 1.05958 w_k1_KO_s104
                                 + 0.739382 w_k1_MAT_s104
                                 + 1.18622 w_k1_MSFT_s104
                                 + 1.05007 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.11371 w_k1_ADM_s104
                                 + 1.00629 w_k1_AXP_s104
                                 + 1.14313 w_k1_DIS_s104
                                 + 1.00252 w_k1_FDX_s104
                                 + 1.52519 w_k1_IBM_s104
                                 + 1.13898 w_k1_INTC_s104
                                 + 1.24545 w_k1_KO_s104 + 1.80222 w_k1_MAT_s104
                                 + 1.1388 w_k1_MSFT_s104 + 1.02462 w_k1_PG_s104
                                 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.06329 w_k1_ADM_s104
                                 + 0.905844 w_k1_AXP_s104
                                 + 0.572972 w_k1_DIS_s104
                                 + 0.864284 w_k1_FDX_s104
                                 + 0.742262 w_k1_IBM_s104
                                 + 0.842374 w_k1_INTC_s104
                                 + 0.697883 w_k1_KO_s104
                                 + 0.867539 w_k1_MAT_s104
                                 + 0.762843 w_k1_MSFT_s104
                                 + 0.819162 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.898504 w_k1_ADM_s104
                                 + 0.626068 w_k1_AXP_s104
                                 + 0.678586 w_k1_DIS_s104
                                 + 0.743211 w_k1_FDX_s104
                                 + 0.907941 w_k1_IBM_s104
                                 + 0.857524 w_k1_INTC_s104
                                 + 0.822117 w_k1_KO_s104
                                 + 0.859152 w_k1_MAT_s104
                                 + 0.84978 w_k1_MSFT_s104
                                 + 0.805902 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.05766 w_k1_ADM_s104
                                 + 1.03 w_k1_AXP_s104 + 0.887529 w_k1_DIS_s104
                                 + 0.952092 w_k1_FDX_s104
                                 + 1.16057 w_k1_IBM_s104
                                 + 0.846146 w_k1_INTC_s104
                                 + 0.93705 w_k1_KO_s104 + 1.16518 w_k1_MAT_s104
                                 + 0.961827 w_k1_MSFT_s104
                                 + 0.923771 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.05222 w_k1_ADM_s104
                                 + 0.823254 w_k1_AXP_s104
                                 + 0.847121 w_k1_DIS_s104
                                 + 0.988306 w_k1_FDX_s104
                                 + 0.721525 w_k1_IBM_s104
                                 + 0.988224 w_k1_INTC_s104
                                 + 0.843209 w_k1_KO_s104
                                 + 0.635524 w_k1_MAT_s104
                                 + 1.01768 w_k1_MSFT_s104
                                 + 0.831833 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.37506 w_k1_ADM_s104
                                 + 1.45928 w_k1_AXP_s104
                                 + 1.32346 w_k1_DIS_s104 + 1.3448 w_k1_FDX_s104
                                 + 1.3079 w_k1_IBM_s104
                                 + 1.25409 w_k1_INTC_s104 + 1.1765 w_k1_KO_s104
                                 + 1.62821 w_k1_MAT_s104
                                 + 1.2524 w_k1_MSFT_s104 + 1.19319 w_k1_PG_s104
                                 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.19337 w_k1_ADM_s104
                                 + 1.10991 w_k1_AXP_s104
                                 + 0.91603 w_k1_DIS_s104 + 1.1052 w_k1_FDX_s104
                                 + 0.984876 w_k1_IBM_s104
                                 + 1.04849 w_k1_INTC_s104
                                 + 0.871333 w_k1_KO_s104
                                 + 1.09526 w_k1_MAT_s104
                                 + 1.04637 w_k1_MSFT_s104
                                 + 0.907451 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.838322 w_k1_ADM_s105
                                 + 0.706253 w_k1_AXP_s105
                                 + 0.844768 w_k1_DIS_s105
                                 + 0.908771 w_k1_FDX_s105
                                 + 0.675949 w_k1_IBM_s105
                                 + 0.886798 w_k1_INTC_s105
                                 + 0.882876 w_k1_KO_s105
                                 + 0.689446 w_k1_MAT_s105
                                 + 0.977598 w_k1_MSFT_s105
                                 + 0.937418 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.06191 w_k1_ADM_s105
                                 + 0.967519 w_k1_AXP_s105
                                 + 0.947198 w_k1_DIS_s105
                                 + 0.959996 w_k1_FDX_s105
                                 + 1.09853 w_k1_IBM_s105
                                 + 0.946555 w_k1_INTC_s105
                                 + 0.991438 w_k1_KO_s105
                                 + 0.938771 w_k1_MAT_s105
                                 + 1.01098 w_k1_MSFT_s105
                                 + 0.945939 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.44894 w_k1_ADM_s105
                                 + 1.16399 w_k1_AXP_s105
                                 + 1.60333 w_k1_DIS_s105 + 1.3628 w_k1_FDX_s105
                                 + 0.925327 w_k1_IBM_s105
                                 + 1.16283 w_k1_INTC_s105
                                 + 1.13541 w_k1_KO_s105
                                 + 0.771848 w_k1_MAT_s105
                                 + 1.35904 w_k1_MSFT_s105
                                 + 1.18165 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.819384 w_k1_ADM_s105
                                 + 0.652755 w_k1_AXP_s105
                                 + 0.650484 w_k1_DIS_s105
                                 + 0.735305 w_k1_FDX_s105
                                 + 0.81306 w_k1_IBM_s105
                                 + 0.853246 w_k1_INTC_s105
                                 + 0.756829 w_k1_KO_s105
                                 + 0.966433 w_k1_MAT_s105
                                 + 0.78081 w_k1_MSFT_s105
                                 + 0.770213 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.36064 w_k1_ADM_s105
                                 + 1.27109 w_k1_AXP_s105
                                 + 1.36448 w_k1_DIS_s105 + 1.0793 w_k1_FDX_s105
                                 + 1.53408 w_k1_IBM_s105
                                 + 1.23151 w_k1_INTC_s105
                                 + 1.19332 w_k1_KO_s105 + 1.34088 w_k1_MAT_s105
                                 + 1.12278 w_k1_MSFT_s105
                                 + 1.16833 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.09649 w_k1_ADM_s105
                                 + 0.952721 w_k1_AXP_s105
                                 + 1.06324 w_k1_DIS_s105
                                 + 1.15407 w_k1_FDX_s105
                                 + 0.803245 w_k1_IBM_s105
                                 + 1.08951 w_k1_INTC_s105
                                 + 0.90478 w_k1_KO_s105
                                 + 0.784814 w_k1_MAT_s105
                                 + 1.16739 w_k1_MSFT_s105
                                 + 0.911865 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.66913 w_k1_ADM_s105
                                 + 0.75189 w_k1_AXP_s105
                                 + 0.709202 w_k1_DIS_s105
                                 + 0.751293 w_k1_FDX_s105
                                 + 1.34976 w_k1_IBM_s105
                                 + 0.831559 w_k1_INTC_s105
                                 + 0.922312 w_k1_KO_s105
                                 + 1.28682 w_k1_MAT_s105
                                 + 0.793372 w_k1_MSFT_s105
                                 + 0.869888 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.11375 w_k1_ADM_s105
                                 + 1.04403 w_k1_AXP_s105
                                 + 0.847674 w_k1_DIS_s105
                                 + 1.01391 w_k1_FDX_s105
                                 + 0.80826 w_k1_IBM_s105
                                 + 0.987692 w_k1_INTC_s105
                                 + 0.732137 w_k1_KO_s105
                                 + 1.05853 w_k1_MAT_s105
                                 + 0.95225 w_k1_MSFT_s105
                                 + 0.92553 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.706932 w_k1_ADM_s106
                                 + 0.805117 w_k1_AXP_s106
                                 + 0.703853 w_k1_DIS_s106
                                 + 0.705416 w_k1_FDX_s106
                                 + 1.19176 w_k1_IBM_s106
                                 + 0.834116 w_k1_INTC_s106
                                 + 0.882702 w_k1_KO_s106
                                 + 0.773002 w_k1_MAT_s106
                                 + 0.739887 w_k1_MSFT_s106
                                 + 0.867669 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.06629 w_k1_ADM_s106
                                 + 1.17128 w_k1_AXP_s106
                                 + 1.46122 w_k1_DIS_s106
                                 + 1.46654 w_k1_FDX_s106
                                 + 0.748313 w_k1_IBM_s106
                                 + 1.25862 w_k1_INTC_s106
                                 + 1.05494 w_k1_KO_s106
                                 + 0.703708 w_k1_MAT_s106
                                 + 1.19994 w_k1_MSFT_s106
                                 + 1.11978 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.848203 w_k1_ADM_s106
                                 + 0.818701 w_k1_AXP_s106
                                 + 0.73954 w_k1_DIS_s106
                                 + 0.895373 w_k1_FDX_s106
                                 + 0.779146 w_k1_IBM_s106
                                 + 0.947118 w_k1_INTC_s106
                                 + 0.672594 w_k1_KO_s106
                                 + 0.525248 w_k1_MAT_s106
                                 + 0.805988 w_k1_MSFT_s106
                                 + 0.905502 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.964452 w_k1_ADM_s106
                                 + 0.975243 w_k1_AXP_s106
                                 + 1.07794 w_k1_DIS_s106
                                 + 0.925592 w_k1_FDX_s106
                                 + 1.01497 w_k1_IBM_s106
                                 + 0.840128 w_k1_INTC_s106
                                 + 1.0437 w_k1_KO_s106 + 0.991682 w_k1_MAT_s106
                                 + 0.937846 w_k1_MSFT_s106
                                 + 0.986841 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.03609 w_k1_ADM_s106
                                 + 1.26878 w_k1_AXP_s106 + 1.7504 w_k1_DIS_s106
                                 + 1.26975 w_k1_FDX_s106
                                 + 1.56786 w_k1_IBM_s106
                                 + 1.20018 w_k1_INTC_s106
                                 + 1.63161 w_k1_KO_s106 + 1.56859 w_k1_MAT_s106
                                 + 1.34345 w_k1_MSFT_s106
                                 + 1.34951 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.04671 w_k1_ADM_s106
                                 + 1.1827 w_k1_AXP_s106 + 1.394 w_k1_DIS_s106
                                 + 1.2552 w_k1_FDX_s106
                                 + 0.890185 w_k1_IBM_s106
                                 + 0.988057 w_k1_INTC_s106
                                 + 0.899317 w_k1_KO_s106
                                 + 0.803047 w_k1_MAT_s106
                                 + 1.08566 w_k1_MSFT_s106
                                 + 1.17766 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.957434 w_k1_ADM_s106
                                 + 1.04037 w_k1_AXP_s106
                                 + 0.981877 w_k1_DIS_s106
                                 + 1.08573 w_k1_FDX_s106
                                 + 1.05412 w_k1_IBM_s106
                                 + 1.06189 w_k1_INTC_s106
                                 + 0.952248 w_k1_KO_s106
                                 + 0.984539 w_k1_MAT_s106
                                 + 1.02298 w_k1_MSFT_s106 + 1.0908 w_k1_PG_s106
                                 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.0588 w_k1_ADM_s106
                                 + 1.03983 w_k1_AXP_s106
                                 + 0.89154 w_k1_DIS_s106
                                 + 0.935351 w_k1_FDX_s106
                                 + 1.42359 w_k1_IBM_s106
                                 + 0.957622 w_k1_INTC_s106
                                 + 1.12569 w_k1_KO_s106 + 1.36447 w_k1_MAT_s106
                                 + 0.906715 w_k1_MSFT_s106
                                 + 0.989154 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.11945 w_k1_ADM_s107
                                 + 1.09942 w_k1_AXP_s107
                                 + 0.891763 w_k1_DIS_s107
                                 + 1.04029 w_k1_FDX_s107
                                 + 1.17577 w_k1_IBM_s107
                                 + 1.14779 w_k1_INTC_s107
                                 + 1.15462 w_k1_KO_s107 + 1.2782 w_k1_MAT_s107
                                 + 1.02081 w_k1_MSFT_s107
                                 + 1.08301 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.915174 w_k1_ADM_s107
                                 + 0.79739 w_k1_AXP_s107
                                 + 1.19845 w_k1_DIS_s107
                                 + 1.00826 w_k1_FDX_s107 + 1.0624 w_k1_IBM_s107
                                 + 1.03241 w_k1_INTC_s107
                                 + 1.20555 w_k1_KO_s107 + 1.08276 w_k1_MAT_s107
                                 + 1.12156 w_k1_MSFT_s107
                                 + 0.996173 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.09965 w_k1_ADM_s107
                                 + 0.88131 w_k1_AXP_s107
                                 + 0.863728 w_k1_DIS_s107
                                 + 0.819165 w_k1_FDX_s107
                                 + 1.66941 w_k1_IBM_s107
                                 + 1.02297 w_k1_INTC_s107
                                 + 1.27101 w_k1_KO_s107 + 1.60665 w_k1_MAT_s107
                                 + 0.883901 w_k1_MSFT_s107
                                 + 1.02447 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.900421 w_k1_ADM_s107
                                 + 0.721486 w_k1_AXP_s107
                                 + 0.55184 w_k1_DIS_s107
                                 + 0.693376 w_k1_FDX_s107
                                 + 0.807597 w_k1_IBM_s107
                                 + 0.863974 w_k1_INTC_s107
                                 + 0.730736 w_k1_KO_s107
                                 + 0.676243 w_k1_MAT_s107
                                 + 0.760003 w_k1_MSFT_s107
                                 + 0.827286 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.12456 w_k1_ADM_s107
                                 + 1.07281 w_k1_AXP_s107 + 1.1505 w_k1_DIS_s107
                                 + 1.22785 w_k1_FDX_s107
                                 + 0.79188 w_k1_IBM_s107
                                 + 1.33276 w_k1_INTC_s107
                                 + 1.10736 w_k1_KO_s107
                                 + 0.850436 w_k1_MAT_s107
                                 + 1.28836 w_k1_MSFT_s107 + 1.1404 w_k1_PG_s107
                                 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.00182 w_k1_ADM_s107
                                 + 1.08079 w_k1_AXP_s107
                                 + 0.947212 w_k1_DIS_s107
                                 + 1.10588 w_k1_FDX_s107
                                 + 0.879971 w_k1_IBM_s107
                                 + 1.12514 w_k1_INTC_s107
                                 + 0.914096 w_k1_KO_s107
                                 + 1.08645 w_k1_MAT_s107
                                 + 1.04127 w_k1_MSFT_s107
                                 + 0.96849 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.1372 w_k1_ADM_s107
                                 + 1.08055 w_k1_AXP_s107
                                 + 1.16824 w_k1_DIS_s107 + 1.2194 w_k1_FDX_s107
                                 + 0.614866 w_k1_IBM_s107
                                 + 1.50671 w_k1_INTC_s107
                                 + 0.875969 w_k1_KO_s107
                                 + 0.532359 w_k1_MAT_s107
                                 + 1.14858 w_k1_MSFT_s107
                                 + 1.00463 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.959678 w_k1_ADM_s107
                                 + 1.46378 w_k1_AXP_s107
                                 + 1.50448 w_k1_DIS_s107
                                 + 1.30798 w_k1_FDX_s107
                                 + 1.56214 w_k1_IBM_s107
                                 + 1.34387 w_k1_INTC_s107
                                 + 1.42755 w_k1_KO_s107 + 2.11957 w_k1_MAT_s107
                                 + 1.33062 w_k1_MSFT_s107
                                 + 1.15143 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.13615 w_k1_ADM_s110
                                 + 1.43599 w_k1_AXP_s110 + 1.3901 w_k1_DIS_s110
                                 + 1.32792 w_k1_FDX_s110
                                 + 1.27037 w_k1_IBM_s110
                                 + 1.18832 w_k1_INTC_s110
                                 + 1.26983 w_k1_KO_s110 + 1.32473 w_k1_MAT_s110
                                 + 1.25776 w_k1_MSFT_s110
                                 + 1.19494 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.961956 w_k1_ADM_s110
                                 + 0.982598 w_k1_AXP_s110
                                 + 0.893042 w_k1_DIS_s110
                                 + 1.05743 w_k1_FDX_s110
                                 + 0.597005 w_k1_IBM_s110
                                 + 1.1284 w_k1_INTC_s110
                                 + 0.763182 w_k1_KO_s110
                                 + 0.619556 w_k1_MAT_s110
                                 + 0.992424 w_k1_MSFT_s110
                                 + 0.8571 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.03002 w_k1_ADM_s110
                                 + 0.972576 w_k1_AXP_s110
                                 + 0.961483 w_k1_DIS_s110
                                 + 1.05075 w_k1_FDX_s110
                                 + 0.839684 w_k1_IBM_s110
                                 + 1.20119 w_k1_INTC_s110
                                 + 0.961225 w_k1_KO_s110
                                 + 0.690656 w_k1_MAT_s110
                                 + 0.98106 w_k1_MSFT_s110
                                 + 1.03034 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.797921 w_k1_ADM_s110
                                 + 0.769806 w_k1_AXP_s110
                                 + 0.660868 w_k1_DIS_s110
                                 + 0.799962 w_k1_FDX_s110
                                 + 0.80213 w_k1_IBM_s110
                                 + 1.13038 w_k1_INTC_s110
                                 + 0.883036 w_k1_KO_s110 + 0.8287 w_k1_MAT_s110
                                 + 0.904036 w_k1_MSFT_s110
                                 + 0.920017 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.803251 w_k1_ADM_s110
                                 + 0.850502 w_k1_AXP_s110
                                 + 1.17676 w_k1_DIS_s110
                                 + 0.929977 w_k1_FDX_s110
                                 + 1.41026 w_k1_IBM_s110
                                 + 1.02416 w_k1_INTC_s110
                                 + 1.30428 w_k1_KO_s110 + 1.39238 w_k1_MAT_s110
                                 + 1.03622 w_k1_MSFT_s110
                                 + 1.04992 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.06532 w_k1_ADM_s110
                                 + 1.21279 w_k1_AXP_s110
                                 + 0.69925 w_k1_DIS_s110
                                 + 0.895503 w_k1_FDX_s110
                                 + 0.707089 w_k1_IBM_s110
                                 + 0.883224 w_k1_INTC_s110
                                 + 0.793072 w_k1_KO_s110
                                 + 0.982168 w_k1_MAT_s110
                                 + 0.796871 w_k1_MSFT_s110
                                 + 0.905265 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.937029 w_k1_ADM_s110
                                 + 1.12578 w_k1_AXP_s110
                                 + 0.829451 w_k1_DIS_s110
                                 + 0.851137 w_k1_FDX_s110
                                 + 1.31792 w_k1_IBM_s110
                                 + 1.09645 w_k1_INTC_s110
                                 + 0.978148 w_k1_KO_s110
                                 + 1.85501 w_k1_MAT_s110
                                 + 0.857807 w_k1_MSFT_s110
                                 + 1.01384 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.949985 w_k1_ADM_s110
                                 + 0.876101 w_k1_AXP_s110
                                 + 1.12843 w_k1_DIS_s110
                                 + 0.95419 w_k1_FDX_s110
                                 + 0.867105 w_k1_IBM_s110
                                 + 1.24679 w_k1_INTC_s110
                                 + 1.22202 w_k1_KO_s110 + 1.08367 w_k1_MAT_s110
                                 + 1.11345 w_k1_MSFT_s110 + 1.1519 w_k1_PG_s110
                                 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.53265 w_k1_ADM_s111
                                 + 1.55175 w_k1_AXP_s111 + 1.7112 w_k1_DIS_s111
                                 + 1.37293 w_k1_FDX_s111 + 1.3284 w_k1_IBM_s111
                                 + 1.32983 w_k1_INTC_s111
                                 + 1.39628 w_k1_KO_s111 + 1.67201 w_k1_MAT_s111
                                 + 1.39306 w_k1_MSFT_s111
                                 + 1.11984 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.30078 w_k1_ADM_s111
                                 + 1.17283 w_k1_AXP_s111
                                 + 1.37331 w_k1_DIS_s111
                                 + 1.07985 w_k1_FDX_s111 + 1.3661 w_k1_IBM_s111
                                 + 1.1375 w_k1_INTC_s111 + 1.28774 w_k1_KO_s111
                                 + 1.53203 w_k1_MAT_s111
                                 + 1.32003 w_k1_MSFT_s111
                                 + 1.04784 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.5256 w_k1_ADM_s111
                                 + 1.02813 w_k1_AXP_s111
                                 + 1.32091 w_k1_DIS_s111
                                 + 1.26882 w_k1_FDX_s111
                                 + 0.788556 w_k1_IBM_s111
                                 + 1.23175 w_k1_INTC_s111
                                 + 0.948129 w_k1_KO_s111
                                 + 0.677175 w_k1_MAT_s111
                                 + 1.34184 w_k1_MSFT_s111
                                 + 1.12036 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.23507 w_k1_ADM_s111
                                 + 1.24006 w_k1_AXP_s111
                                 + 1.17628 w_k1_DIS_s111
                                 + 0.902757 w_k1_FDX_s111
                                 + 1.85191 w_k1_IBM_s111
                                 + 0.920801 w_k1_INTC_s111
                                 + 1.22915 w_k1_KO_s111 + 2.06223 w_k1_MAT_s111
                                 + 1.0143 w_k1_MSFT_s111 + 1.11052 w_k1_PG_s111
                                 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.923716 w_k1_ADM_s111
                                 + 0.852636 w_k1_AXP_s111
                                 + 0.874822 w_k1_DIS_s111
                                 + 0.923949 w_k1_FDX_s111
                                 + 1.07926 w_k1_IBM_s111
                                 + 0.941771 w_k1_INTC_s111
                                 + 0.975827 w_k1_KO_s111
                                 + 1.40156 w_k1_MAT_s111
                                 + 1.11032 w_k1_MSFT_s111
                                 + 1.00919 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.22239 w_k1_ADM_s111
                                 + 1.27809 w_k1_AXP_s111
                                 + 1.04343 w_k1_DIS_s111
                                 + 1.15224 w_k1_FDX_s111
                                 + 1.19599 w_k1_IBM_s111
                                 + 1.1811 w_k1_INTC_s111 + 1.05582 w_k1_KO_s111
                                 + 1.38563 w_k1_MAT_s111
                                 + 1.25474 w_k1_MSFT_s111
                                 + 1.11499 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.30008 w_k1_ADM_s111
                                 + 1.18409 w_k1_AXP_s111
                                 + 0.85447 w_k1_DIS_s111
                                 + 1.10692 w_k1_FDX_s111
                                 + 0.859925 w_k1_IBM_s111
                                 + 1.05973 w_k1_INTC_s111
                                 + 0.820491 w_k1_KO_s111
                                 + 0.965474 w_k1_MAT_s111
                                 + 1.0501 w_k1_MSFT_s111
                                 + 0.960829 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.976927 w_k1_ADM_s111
                                 + 0.830381 w_k1_AXP_s111
                                 + 0.607852 w_k1_DIS_s111
                                 + 0.799021 w_k1_FDX_s111
                                 + 0.816205 w_k1_IBM_s111
                                 + 0.802913 w_k1_INTC_s111
                                 + 0.715771 w_k1_KO_s111
                                 + 0.763345 w_k1_MAT_s111
                                 + 0.841335 w_k1_MSFT_s111
                                 + 0.847922 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.989954 w_k1_ADM_s112
                                 + 1.37909 w_k1_AXP_s112
                                 + 1.15446 w_k1_DIS_s112
                                 + 1.07894 w_k1_FDX_s112
                                 + 1.44098 w_k1_IBM_s112
                                 + 1.39721 w_k1_INTC_s112
                                 + 1.40405 w_k1_KO_s112 + 1.84312 w_k1_MAT_s112
                                 + 1.20901 w_k1_MSFT_s112
                                 + 1.19045 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.951821 w_k1_ADM_s112
                                 + 0.90259 w_k1_AXP_s112
                                 + 0.927415 w_k1_DIS_s112
                                 + 0.922847 w_k1_FDX_s112
                                 + 1.29123 w_k1_IBM_s112
                                 + 1.14834 w_k1_INTC_s112
                                 + 1.17408 w_k1_KO_s112 + 1.13012 w_k1_MAT_s112
                                 + 1.06967 w_k1_MSFT_s112
                                 + 0.90579 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.822762 w_k1_ADM_s112
                                 + 0.732322 w_k1_AXP_s112
                                 + 0.668535 w_k1_DIS_s112
                                 + 0.743837 w_k1_FDX_s112
                                 + 0.731535 w_k1_IBM_s112
                                 + 0.920523 w_k1_INTC_s112
                                 + 0.682752 w_k1_KO_s112
                                 + 0.604158 w_k1_MAT_s112
                                 + 0.798725 w_k1_MSFT_s112
                                 + 0.710221 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.06176 w_k1_ADM_s112
                                 + 1.07803 w_k1_AXP_s112
                                 + 1.39312 w_k1_DIS_s112
                                 + 1.23872 w_k1_FDX_s112
                                 + 0.976564 w_k1_IBM_s112
                                 + 1.07036 w_k1_INTC_s112
                                 + 1.21428 w_k1_KO_s112
                                 + 0.970869 w_k1_MAT_s112
                                 + 1.1995 w_k1_MSFT_s112 + 1.01295 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.857079 w_k1_ADM_s112
                                 + 0.804632 w_k1_AXP_s112
                                 + 0.765033 w_k1_DIS_s112
                                 + 0.815173 w_k1_FDX_s112
                                 + 1.01523 w_k1_IBM_s112
                                 + 0.856908 w_k1_INTC_s112
                                 + 0.920269 w_k1_KO_s112
                                 + 0.767968 w_k1_MAT_s112
                                 + 0.80698 w_k1_MSFT_s112
                                 + 0.799084 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.881674 w_k1_ADM_s112
                                 + 1.00384 w_k1_AXP_s112
                                 + 1.02155 w_k1_DIS_s112
                                 + 0.90449 w_k1_FDX_s112
                                 + 1.86679 w_k1_IBM_s112
                                 + 1.18927 w_k1_INTC_s112 + 1.2825 w_k1_KO_s112
                                 + 1.11528 w_k1_MAT_s112
                                 + 1.05747 w_k1_MSFT_s112
                                 + 0.915161 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.41035 w_k1_ADM_s112
                                 + 1.54836 w_k1_AXP_s112
                                 + 1.67277 w_k1_DIS_s112
                                 + 1.34274 w_k1_FDX_s112
                                 + 1.37497 w_k1_IBM_s112
                                 + 1.26824 w_k1_INTC_s112
                                 + 1.31556 w_k1_KO_s112 + 1.53602 w_k1_MAT_s112
                                 + 1.28432 w_k1_MSFT_s112
                                 + 1.07586 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.07849 w_k1_ADM_s112
                                 + 0.957962 w_k1_AXP_s112
                                 + 0.890421 w_k1_DIS_s112
                                 + 0.97841 w_k1_FDX_s112
                                 + 0.775861 w_k1_IBM_s112
                                 + 0.93204 w_k1_INTC_s112
                                 + 0.969396 w_k1_KO_s112
                                 + 0.836003 w_k1_MAT_s112
                                 + 1.11019 w_k1_MSFT_s112
                                 + 0.890608 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.99345 w_k1_ADM_s113
                                 + 1.20579 w_k1_AXP_s113
                                 + 1.12999 w_k1_DIS_s113 + 1.1134 w_k1_FDX_s113
                                 + 1.08346 w_k1_IBM_s113
                                 + 1.31273 w_k1_INTC_s113
                                 + 0.974031 w_k1_KO_s113
                                 + 1.13811 w_k1_MAT_s113
                                 + 0.958685 w_k1_MSFT_s113
                                 + 1.0194 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.737807 w_k1_ADM_s113
                                 + 0.720755 w_k1_AXP_s113
                                 + 0.65235 w_k1_DIS_s113
                                 + 0.734078 w_k1_FDX_s113
                                 + 0.965891 w_k1_IBM_s113
                                 + 0.892872 w_k1_INTC_s113
                                 + 0.724525 w_k1_KO_s113
                                 + 0.886042 w_k1_MAT_s113
                                 + 0.733049 w_k1_MSFT_s113
                                 + 0.828624 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.07308 w_k1_ADM_s113
                                 + 1.01253 w_k1_AXP_s113
                                 + 1.03874 w_k1_DIS_s113
                                 + 0.964862 w_k1_FDX_s113
                                 + 0.870962 w_k1_IBM_s113
                                 + 1.1019 w_k1_INTC_s113
                                 + 0.819123 w_k1_KO_s113
                                 + 0.609674 w_k1_MAT_s113
                                 + 0.908473 w_k1_MSFT_s113
                                 + 0.944613 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.11059 w_k1_ADM_s113
                                 + 1.31057 w_k1_AXP_s113
                                 + 1.38452 w_k1_DIS_s113 + 1.2057 w_k1_FDX_s113
                                 + 1.25707 w_k1_IBM_s113
                                 + 1.43627 w_k1_INTC_s113
                                 + 1.22206 w_k1_KO_s113 + 1.03263 w_k1_MAT_s113
                                 + 1.18337 w_k1_MSFT_s113
                                 + 1.12714 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.915503 w_k1_ADM_s113
                                 + 1.04741 w_k1_AXP_s113
                                 + 0.980193 w_k1_DIS_s113 + 0.875 w_k1_FDX_s113
                                 + 1.6357 w_k1_IBM_s113
                                 + 1.25106 w_k1_INTC_s113
                                 + 1.10356 w_k1_KO_s113 + 1.54192 w_k1_MAT_s113
                                 + 0.876586 w_k1_MSFT_s113
                                 + 1.13189 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.31875 w_k1_ADM_s113
                                 + 1.09107 w_k1_AXP_s113
                                 + 1.14759 w_k1_DIS_s113
                                 + 1.28962 w_k1_FDX_s113
                                 + 0.667619 w_k1_IBM_s113
                                 + 1.19508 w_k1_INTC_s113
                                 + 0.832172 w_k1_KO_s113
                                 + 0.650288 w_k1_MAT_s113
                                 + 1.16333 w_k1_MSFT_s113
                                 + 0.970014 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.979004 w_k1_ADM_s113
                                 + 1.0936 w_k1_AXP_s113
                                 + 0.727375 w_k1_DIS_s113
                                 + 0.867092 w_k1_FDX_s113
                                 + 0.946548 w_k1_IBM_s113
                                 + 0.998072 w_k1_INTC_s113
                                 + 0.777636 w_k1_KO_s113
                                 + 1.09798 w_k1_MAT_s113
                                 + 0.737659 w_k1_MSFT_s113
                                 + 0.829031 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.30997 w_k1_ADM_s113
                                 + 1.40638 w_k1_AXP_s113
                                 + 1.90011 w_k1_DIS_s113
                                 + 1.53734 w_k1_FDX_s113
                                 + 1.24429 w_k1_IBM_s113
                                 + 1.63861 w_k1_INTC_s113
                                 + 1.28498 w_k1_KO_s113 + 1.84248 w_k1_MAT_s113
                                 + 1.31742 w_k1_MSFT_s113
                                 + 1.39024 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.38239 w_k1_ADM_s114
                                 + 1.43532 w_k1_AXP_s114
                                 + 1.41763 w_k1_DIS_s114
                                 + 1.37409 w_k1_FDX_s114
                                 + 1.02882 w_k1_IBM_s114
                                 + 1.13715 w_k1_INTC_s114
                                 + 1.24529 w_k1_KO_s114
                                 + 0.946879 w_k1_MAT_s114
                                 + 1.22559 w_k1_MSFT_s114
                                 + 1.16241 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.938233 w_k1_ADM_s114
                                 + 1.12595 w_k1_AXP_s114
                                 + 1.00915 w_k1_DIS_s114
                                 + 0.931898 w_k1_FDX_s114
                                 + 1.40193 w_k1_IBM_s114
                                 + 1.09539 w_k1_INTC_s114 + 1.2521 w_k1_KO_s114
                                 + 1.57837 w_k1_MAT_s114
                                 + 0.99218 w_k1_MSFT_s114
                                 + 1.10767 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.718184 w_k1_ADM_s114
                                 + 0.79579 w_k1_AXP_s114
                                 + 0.761838 w_k1_DIS_s114
                                 + 0.827193 w_k1_FDX_s114
                                 + 0.906528 w_k1_IBM_s114
                                 + 0.849134 w_k1_INTC_s114
                                 + 0.793757 w_k1_KO_s114
                                 + 0.739103 w_k1_MAT_s114
                                 + 0.933623 w_k1_MSFT_s114
                                 + 0.839833 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.918897 w_k1_ADM_s114
                                 + 1.18019 w_k1_AXP_s114
                                 + 1.09533 w_k1_DIS_s114
                                 + 1.15755 w_k1_FDX_s114
                                 + 0.963184 w_k1_IBM_s114
                                 + 1.01497 w_k1_INTC_s114
                                 + 1.12761 w_k1_KO_s114 + 1.26774 w_k1_MAT_s114
                                 + 1.02846 w_k1_MSFT_s114
                                 + 1.05205 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.20753 w_k1_ADM_s114
                                 + 1.36831 w_k1_AXP_s114
                                 + 1.47447 w_k1_DIS_s114
                                 + 1.25171 w_k1_FDX_s114
                                 + 1.54371 w_k1_IBM_s114
                                 + 1.11022 w_k1_INTC_s114
                                 + 1.48072 w_k1_KO_s114 + 1.54819 w_k1_MAT_s114
                                 + 1.26732 w_k1_MSFT_s114
                                 + 1.28094 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.00406 w_k1_ADM_s114
                                 + 1.04113 w_k1_AXP_s114
                                 + 0.968669 w_k1_DIS_s114
                                 + 1.04111 w_k1_FDX_s114
                                 + 0.886393 w_k1_IBM_s114
                                 + 1.08677 w_k1_INTC_s114
                                 + 0.915722 w_k1_KO_s114
                                 + 0.728817 w_k1_MAT_s114
                                 + 0.837185 w_k1_MSFT_s114
                                 + 0.901135 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.17698 w_k1_ADM_s114
                                 + 1.10676 w_k1_AXP_s114
                                 + 1.16518 w_k1_DIS_s114
                                 + 1.14933 w_k1_FDX_s114 + 1.2181 w_k1_IBM_s114
                                 + 1.0305 w_k1_INTC_s114 + 1.2607 w_k1_KO_s114
                                 + 1.20329 w_k1_MAT_s114
                                 + 1.09343 w_k1_MSFT_s114
                                 + 0.978296 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.12061 w_k1_ADM_s114
                                 + 0.977564 w_k1_AXP_s114
                                 + 0.892986 w_k1_DIS_s114
                                 + 0.986059 w_k1_FDX_s114
                                 + 0.800316 w_k1_IBM_s114
                                 + 0.803378 w_k1_INTC_s114
                                 + 0.950561 w_k1_KO_s114
                                 + 0.714375 w_k1_MAT_s114
                                 + 0.929816 w_k1_MSFT_s114
                                 + 0.944607 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.17661 w_k1_ADM_s115
                                 + 1.33892 w_k1_AXP_s115
                                 + 1.10179 w_k1_DIS_s115
                                 + 1.26445 w_k1_FDX_s115
                                 + 0.642546 w_k1_IBM_s115
                                 + 1.08131 w_k1_INTC_s115
                                 + 0.890291 w_k1_KO_s115
                                 + 0.819675 w_k1_MAT_s115
                                 + 0.956311 w_k1_MSFT_s115
                                 + 1.16862 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.775989 w_k1_ADM_s115
                                 + 0.805156 w_k1_AXP_s115
                                 + 0.682174 w_k1_DIS_s115
                                 + 0.713597 w_k1_FDX_s115
                                 + 0.758967 w_k1_IBM_s115
                                 + 0.812822 w_k1_INTC_s115
                                 + 0.748908 w_k1_KO_s115
                                 + 0.750899 w_k1_MAT_s115
                                 + 0.690429 w_k1_MSFT_s115
                                 + 0.936858 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.08579 w_k1_ADM_s115
                                 + 1.03862 w_k1_AXP_s115
                                 + 0.915705 w_k1_DIS_s115
                                 + 1.03521 w_k1_FDX_s115
                                 + 0.832442 w_k1_IBM_s115
                                 + 0.910677 w_k1_INTC_s115
                                 + 0.951827 w_k1_KO_s115
                                 + 0.917857 w_k1_MAT_s115
                                 + 0.798889 w_k1_MSFT_s115
                                 + 1.13901 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.899968 w_k1_ADM_s115
                                 + 0.97546 w_k1_AXP_s115
                                 + 1.13475 w_k1_DIS_s115
                                 + 0.970377 w_k1_FDX_s115
                                 + 1.39577 w_k1_IBM_s115
                                 + 0.898301 w_k1_INTC_s115
                                 + 1.36071 w_k1_KO_s115 + 1.61552 w_k1_MAT_s115
                                 + 0.964321 w_k1_MSFT_s115
                                 + 1.11306 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.24882 w_k1_ADM_s115
                                 + 1.13092 w_k1_AXP_s115
                                 + 1.28569 w_k1_DIS_s115
                                 + 1.21678 w_k1_FDX_s115
                                 + 0.956023 w_k1_IBM_s115
                                 + 0.999422 w_k1_INTC_s115
                                 + 1.10538 w_k1_KO_s115 + 1.1172 w_k1_MAT_s115
                                 + 1.15591 w_k1_MSFT_s115
                                 + 1.06362 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.756499 w_k1_ADM_s115
                                 + 0.763528 w_k1_AXP_s115
                                 + 0.849222 w_k1_DIS_s115
                                 + 0.90628 w_k1_FDX_s115
                                 + 0.726009 w_k1_IBM_s115
                                 + 0.898708 w_k1_INTC_s115
                                 + 0.920464 w_k1_KO_s115
                                 + 0.783992 w_k1_MAT_s115
                                 + 0.955183 w_k1_MSFT_s115
                                 + 0.986997 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.744853 w_k1_ADM_s115
                                 + 0.806767 w_k1_AXP_s115
                                 + 0.842124 w_k1_DIS_s115
                                 + 0.816754 w_k1_FDX_s115
                                 + 1.05174 w_k1_IBM_s115
                                 + 0.810614 w_k1_INTC_s115
                                 + 0.867108 w_k1_KO_s115
                                 + 1.11728 w_k1_MAT_s115
                                 + 0.752789 w_k1_MSFT_s115
                                 + 0.876539 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.904104 w_k1_ADM_s115
                                 + 1.46439 w_k1_AXP_s115
                                 + 1.43636 w_k1_DIS_s115
                                 + 1.23399 w_k1_FDX_s115
                                 + 1.55205 w_k1_IBM_s115
                                 + 1.29243 w_k1_INTC_s115
                                 + 1.33834 w_k1_KO_s115 + 2.12835 w_k1_MAT_s115
                                 + 1.16883 w_k1_MSFT_s115
                                 + 1.22522 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.68464 w_k1_ADM_s116
                                 + 1.64239 w_k1_AXP_s116
                                 + 1.43631 w_k1_DIS_s116
                                 + 1.28421 w_k1_FDX_s116 + 1.1126 w_k1_IBM_s116
                                 + 1.05227 w_k1_INTC_s116
                                 + 1.27561 w_k1_KO_s116 + 1.50763 w_k1_MAT_s116
                                 + 1.17443 w_k1_MSFT_s116
                                 + 1.15831 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.05542 w_k1_ADM_s116
                                 + 0.91161 w_k1_AXP_s116
                                 + 1.03758 w_k1_DIS_s116
                                 + 0.946556 w_k1_FDX_s116
                                 + 1.53029 w_k1_IBM_s116
                                 + 1.10752 w_k1_INTC_s116
                                 + 1.33498 w_k1_KO_s116 + 1.29015 w_k1_MAT_s116
                                 + 1.21496 w_k1_MSFT_s116
                                 + 0.941974 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.812603 w_k1_ADM_s116
                                 + 0.772309 w_k1_AXP_s116
                                 + 0.824504 w_k1_DIS_s116
                                 + 0.755937 w_k1_FDX_s116
                                 + 1.02975 w_k1_IBM_s116
                                 + 0.869428 w_k1_INTC_s116
                                 + 1.02779 w_k1_KO_s116 + 0.7804 w_k1_MAT_s116
                                 + 0.878649 w_k1_MSFT_s116
                                 + 0.782491 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.68145 w_k1_ADM_s116
                                 + 1.33666 w_k1_AXP_s116
                                 + 1.65594 w_k1_DIS_s116
                                 + 1.46222 w_k1_FDX_s116
                                 + 0.960706 w_k1_IBM_s116
                                 + 1.47158 w_k1_INTC_s116
                                 + 1.47405 w_k1_KO_s116 + 1.43577 w_k1_MAT_s116
                                 + 1.52888 w_k1_MSFT_s116
                                 + 1.15735 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.726382 w_k1_ADM_s116
                                 + 0.756196 w_k1_AXP_s116
                                 + 0.641125 w_k1_DIS_s116
                                 + 0.814146 w_k1_FDX_s116
                                 + 0.648325 w_k1_IBM_s116
                                 + 1.01785 w_k1_INTC_s116
                                 + 0.719981 w_k1_KO_s116
                                 + 0.659173 w_k1_MAT_s116
                                 + 0.882677 w_k1_MSFT_s116
                                 + 0.702334 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.05115 w_k1_ADM_s116
                                 + 1.06966 w_k1_AXP_s116
                                 + 1.11975 w_k1_DIS_s116
                                 + 1.17606 w_k1_FDX_s116
                                 + 1.02483 w_k1_IBM_s116
                                 + 1.05549 w_k1_INTC_s116
                                 + 1.06435 w_k1_KO_s116
                                 + 0.753681 w_k1_MAT_s116
                                 + 1.16238 w_k1_MSFT_s116
                                 + 0.950759 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.16341 w_k1_ADM_s116
                                 + 1.08004 w_k1_AXP_s116
                                 + 1.01132 w_k1_DIS_s116
                                 + 1.06273 w_k1_FDX_s116
                                 + 0.900082 w_k1_IBM_s116
                                 + 0.885835 w_k1_INTC_s116
                                 + 0.896015 w_k1_KO_s116
                                 + 0.692276 w_k1_MAT_s116
                                 + 0.978537 w_k1_MSFT_s116
                                 + 0.916219 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.25882 w_k1_ADM_s116
                                 + 1.09291 w_k1_AXP_s116
                                 + 1.21611 w_k1_DIS_s116
                                 + 1.07343 w_k1_FDX_s116
                                 + 1.28374 w_k1_IBM_s116
                                 + 1.16837 w_k1_INTC_s116
                                 + 1.25841 w_k1_KO_s116 + 1.02337 w_k1_MAT_s116
                                 + 1.24497 w_k1_MSFT_s116
                                 + 0.974168 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.12885 w_k1_ADM_s117
                                 + 0.941143 w_k1_AXP_s117
                                 + 1.28661 w_k1_DIS_s117
                                 + 1.02018 w_k1_FDX_s117
                                 + 0.925875 w_k1_IBM_s117
                                 + 1.34677 w_k1_INTC_s117
                                 + 0.963514 w_k1_KO_s117
                                 + 0.743232 w_k1_MAT_s117
                                 + 1.10171 w_k1_MSFT_s117
                                 + 1.10939 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.26803 w_k1_ADM_s117
                                 + 0.913279 w_k1_AXP_s117
                                 + 0.804698 w_k1_DIS_s117
                                 + 0.957225 w_k1_FDX_s117
                                 + 0.697281 w_k1_IBM_s117
                                 + 1.03794 w_k1_INTC_s117
                                 + 0.714452 w_k1_KO_s117
                                 + 0.58545 w_k1_MAT_s117
                                 + 0.878445 w_k1_MSFT_s117
                                 + 0.984216 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.13529 w_k1_ADM_s117
                                 + 1.29008 w_k1_AXP_s117
                                 + 1.68926 w_k1_DIS_s117
                                 + 1.37636 w_k1_FDX_s117
                                 + 1.10953 w_k1_IBM_s117
                                 + 1.20525 w_k1_INTC_s117
                                 + 1.34888 w_k1_KO_s117 + 1.18273 w_k1_MAT_s117
                                 + 1.38621 w_k1_MSFT_s117
                                 + 1.28446 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.18114 w_k1_ADM_s117
                                 + 0.836017 w_k1_AXP_s117
                                 + 1.28873 w_k1_DIS_s117
                                 + 1.03601 w_k1_FDX_s117
                                 + 1.04974 w_k1_IBM_s117
                                 + 1.19693 w_k1_INTC_s117
                                 + 1.10504 w_k1_KO_s117
                                 + 0.960224 w_k1_MAT_s117
                                 + 1.25276 w_k1_MSFT_s117
                                 + 1.22652 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.988935 w_k1_ADM_s117
                                 + 0.67858 w_k1_AXP_s117
                                 + 0.818979 w_k1_DIS_s117
                                 + 0.816772 w_k1_FDX_s117
                                 + 0.951359 w_k1_IBM_s117
                                 + 1.14575 w_k1_INTC_s117
                                 + 0.892864 w_k1_KO_s117
                                 + 0.755674 w_k1_MAT_s117
                                 + 0.862906 w_k1_MSFT_s117
                                 + 0.966878 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.985546 w_k1_ADM_s117
                                 + 1.04201 w_k1_AXP_s117
                                 + 1.02152 w_k1_DIS_s117
                                 + 0.969294 w_k1_FDX_s117
                                 + 1.20582 w_k1_IBM_s117
                                 + 1.27718 w_k1_INTC_s117
                                 + 1.21886 w_k1_KO_s117
                                 + 0.907305 w_k1_MAT_s117
                                 + 1.02399 w_k1_MSFT_s117
                                 + 1.04351 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.807276 w_k1_ADM_s117
                                 + 0.806267 w_k1_AXP_s117
                                 + 0.763548 w_k1_DIS_s117
                                 + 0.786044 w_k1_FDX_s117
                                 + 1.06703 w_k1_IBM_s117
                                 + 0.996633 w_k1_INTC_s117
                                 + 0.901463 w_k1_KO_s117
                                 + 0.867465 w_k1_MAT_s117
                                 + 0.850283 w_k1_MSFT_s117
                                 + 1.01262 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.28036 w_k1_ADM_s117
                                 + 1.27052 w_k1_AXP_s117
                                 + 1.03089 w_k1_DIS_s117
                                 + 1.18096 w_k1_FDX_s117
                                 + 0.792205 w_k1_IBM_s117
                                 + 1.14641 w_k1_INTC_s117
                                 + 0.771984 w_k1_KO_s117
                                 + 0.900585 w_k1_MAT_s117
                                 + 1.08974 w_k1_MSFT_s117
                                 + 0.929947 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0180965446532552 w_k1_ADM_s000
                                 + 0.0183521772429232 w_k1_ADM_s001
                                 + 0.045022117793689 w_k1_ADM_s002
                                 + 0.0172553825631781 w_k1_ADM_s003
                                 + 0.0339970147644588 w_k1_ADM_s004
                                 + 0.0773328492705943 w_k1_ADM_s005
                                 + 0.0254121187387979 w_k1_ADM_s006
                                 + 0.0241621825870104 w_k1_ADM_s007
                                 + 0.00956323665154366 w_k1_ADM_s010
                                 + 0.011243990657634 w_k1_ADM_s011
                                 + 0.0101065778748768 w_k1_ADM_s012
                                 + 0.00555203072700554 w_k1_ADM_s013
                                 + 0.0073205038271781 w_k1_ADM_s014
                                 + 0.00736514794096257 w_k1_ADM_s015
                                 + 0.0080460852881443 w_k1_ADM_s016
                                 + 0.018146066537432 w_k1_ADM_s017
                                 + 0.0565626817240757 w_k1_ADM_s100
                                 + 0.0571294540299372 w_k1_ADM_s101
                                 + 0.0523462789967817 w_k1_ADM_s102
                                 + 0.0509201335084242 w_k1_ADM_s103
                                 + 0.0481276887742524 w_k1_ADM_s104
                                 + 0.0965134952289353 w_k1_ADM_s105
                                 + 0.0404531554073084 w_k1_ADM_s106
                                 + 0.0935630543589525 w_k1_ADM_s107
                                 + 0.0260977022029165 w_k1_ADM_s110
                                 + 0.0161536956330394 w_k1_ADM_s111
                                 + 0.0153227678114443 w_k1_ADM_s112
                                 + 0.020904861586645 w_k1_ADM_s113
                                 + 0.0453499164672419 w_k1_ADM_s114
                                 + 0.0285287503887643 w_k1_ADM_s115
                                 + 0.0246475762137328 w_k1_ADM_s116
                                 + 0.013468120258055 w_k1_ADM_s117
                                 + 0.0187385546451668 w_k1_AXP_s000
                                 + 0.0177148604791518 w_k1_AXP_s001
                                 + 0.0469835892067617 w_k1_AXP_s002
                                 + 0.0181155941234615 w_k1_AXP_s003
                                 + 0.0257270158914707 w_k1_AXP_s004
                                 + 0.0828507010929544 w_k1_AXP_s005
                                 + 0.0242138628885489 w_k1_AXP_s006
                                 + 0.0254714323420757 w_k1_AXP_s007
                                 + 0.0102342278920344 w_k1_AXP_s010
                                 + 0.010617874053845 w_k1_AXP_s011
                                 + 0.0116636109065406 w_k1_AXP_s012
                                 + 0.00697142081889279 w_k1_AXP_s013
                                 + 0.00640346160465643 w_k1_AXP_s014
                                 + 0.00592414175634865 w_k1_AXP_s015
                                 + 0.00710909839177674 w_k1_AXP_s016
                                 + 0.0172902658743648 w_k1_AXP_s017
                                 + 0.0604572024674936 w_k1_AXP_s100
                                 + 0.0639736077386732 w_k1_AXP_s101
                                 + 0.0497137661684749 w_k1_AXP_s102
                                 + 0.0438559199604434 w_k1_AXP_s103
                                 + 0.0419382642834742 w_k1_AXP_s104
                                 + 0.0858662685762025 w_k1_AXP_s105
                                 + 0.042696196708708 w_k1_AXP_s106
                                 + 0.0927807503212378 w_k1_AXP_s107
                                 + 0.0276772927926891 w_k1_AXP_s110
                                 + 0.0144254788739116 w_k1_AXP_s111
                                 + 0.0150721718953697 w_k1_AXP_s112
                                 + 0.0222378699535179 w_k1_AXP_s113
                                 + 0.0483624306007491 w_k1_AXP_s114
                                 + 0.0309752658554667 w_k1_AXP_s115
                                 + 0.0228157790612069 w_k1_AXP_s116
                                 + 0.0124880299268915 w_k1_AXP_s117
                                 + 0.0192077667663448 w_k1_DIS_s000
                                 + 0.0180453157632335 w_k1_DIS_s001
                                 + 0.0471574426208421 w_k1_DIS_s002
                                 + 0.0178632765283295 w_k1_DIS_s003
                                 + 0.0303397713032263 w_k1_DIS_s004
                                 + 0.0817496570376657 w_k1_DIS_s005
                                 + 0.0253497950557795 w_k1_DIS_s006
                                 + 0.0248982347732777 w_k1_DIS_s007
                                 + 0.00994862319800145 w_k1_DIS_s010
                                 + 0.0121991755440746 w_k1_DIS_s011
                                 + 0.0120311410858777 w_k1_DIS_s012
                                 + 0.00666986565236296 w_k1_DIS_s013
                                 + 0.00670602732652588 w_k1_DIS_s014
                                 + 0.00804594879211115 w_k1_DIS_s015
                                 + 0.00729104414965651 w_k1_DIS_s016
                                 + 0.0159168598788842 w_k1_DIS_s017
                                 + 0.0653856250341674 w_k1_DIS_s100
                                 + 0.0598456224495579 w_k1_DIS_s101
                                 + 0.0549775083822442 w_k1_DIS_s102
                                 + 0.0440093209443021 w_k1_DIS_s103
                                 + 0.0410861118377029 w_k1_DIS_s104
                                 + 0.0925239919922141 w_k1_DIS_s105
                                 + 0.0448208155944732 w_k1_DIS_s106
                                 + 0.0925624859419142 w_k1_DIS_s107
                                 + 0.0277536742127336 w_k1_DIS_s110
                                 + 0.0133500977014078 w_k1_DIS_s111
                                 + 0.0149450450403862 w_k1_DIS_s112
                                 + 0.0218479472405296 w_k1_DIS_s113
                                 + 0.047022653442602 w_k1_DIS_s114
                                 + 0.0308578382155483 w_k1_DIS_s115
                                 + 0.0234616867938467 w_k1_DIS_s116
                                 + 0.0135894781960513 w_k1_DIS_s117
                                 + 0.018404089809149 w_k1_FDX_s000
                                 + 0.0166996807998912 w_k1_FDX_s001
                                 + 0.0459278428966088 w_k1_FDX_s002
                                 + 0.0172231428068554 w_k1_FDX_s003
                                 + 0.0282174061464145 w_k1_FDX_s004
                                 + 0.0771337097038231 w_k1_FDX_s005
                                 + 0.0256984539852883 w_k1_FDX_s006
                                 + 0.0242929603108146 w_k1_FDX_s007
                                 + 0.0100689267695976 w_k1_FDX_s010
                                 + 0.0122824974138749 w_k1_FDX_s011
                                 + 0.0124762789684825 w_k1_FDX_s012
                                 + 0.00629852758358838 w_k1_FDX_s013
                                 + 0.00622186718524904 w_k1_FDX_s014
                                 + 0.00679601109249049 w_k1_FDX_s015
                                 + 0.00741244693971478 w_k1_FDX_s016
                                 + 0.0160048104726653 w_k1_FDX_s017
                                 + 0.0640223994130263 w_k1_FDX_s100
                                 + 0.0583112564039303 w_k1_FDX_s101
                                 + 0.0529618666335805 w_k1_FDX_s102
                                 + 0.0474932911962555 w_k1_FDX_s103
                                 + 0.0435336032720984 w_k1_FDX_s104
                                 + 0.0889722070210415 w_k1_FDX_s105
                                 + 0.0440289911786356 w_k1_FDX_s106
                                 + 0.0954883387482035 w_k1_FDX_s107
                                 + 0.0276356330667728 w_k1_FDX_s110
                                 + 0.0138968957017911 w_k1_FDX_s111
                                 + 0.0147823111054785 w_k1_FDX_s112
                                 + 0.0209770177381494 w_k1_FDX_s113
                                 + 0.0472112037329224 w_k1_FDX_s114
                                 + 0.030117205095442 w_k1_FDX_s115
                                 + 0.0228560162572609 w_k1_FDX_s116
                                 + 0.0128205881934353 w_k1_FDX_s117
                                 + 0.0191356340153444 w_k1_IBM_s000
                                 + 0.0175038935938098 w_k1_IBM_s001
                                 + 0.0478721807365767 w_k1_IBM_s002
                                 + 0.0192060281328393 w_k1_IBM_s003
                                 + 0.0274296137444585 w_k1_IBM_s004
                                 + 0.0805230007207997 w_k1_IBM_s005
                                 + 0.0254848495477049 w_k1_IBM_s006
                                 + 0.0260483305021684 w_k1_IBM_s007
                                 + 0.00999489388141389 w_k1_IBM_s010
                                 + 0.0117490209381761 w_k1_IBM_s011
                                 + 0.0117528446843952 w_k1_IBM_s012
                                 + 0.00642148602616351 w_k1_IBM_s013
                                 + 0.0065392012711835 w_k1_IBM_s014
                                 + 0.00700860208743493 w_k1_IBM_s015
                                 + 0.00718955278236748 w_k1_IBM_s016
                                 + 0.0172131132153711 w_k1_IBM_s017
                                 + 0.0630289077243658 w_k1_IBM_s100
                                 + 0.058790304968789 w_k1_IBM_s101
                                 + 0.0549153067429368 w_k1_IBM_s102
                                 + 0.046988448661069 w_k1_IBM_s103
                                 + 0.0444012820128704 w_k1_IBM_s104
                                 + 0.0919772302674593 w_k1_IBM_s105
                                 + 0.0429178297627986 w_k1_IBM_s106
                                 + 0.0964305657735405 w_k1_IBM_s107
                                 + 0.0268094880500921 w_k1_IBM_s110
                                 + 0.0144243805011838 w_k1_IBM_s111
                                 + 0.0159708678747873 w_k1_IBM_s112
                                 + 0.0222885821413568 w_k1_IBM_s113
                                 + 0.0484521354962803 w_k1_IBM_s114
                                 + 0.0314843134431938 w_k1_IBM_s115
                                 + 0.0236921962866776 w_k1_IBM_s116
                                 + 0.0126557120569758 w_k1_IBM_s117
                                 + 0.018924321957922 w_k1_INTC_s000
                                 + 0.0173376344249902 w_k1_INTC_s001
                                 + 0.0490149187221304 w_k1_INTC_s002
                                 + 0.0134282459666577 w_k1_INTC_s003
                                 + 0.0261589039058958 w_k1_INTC_s004
                                 + 0.0823869384406655 w_k1_INTC_s005
                                 + 0.0245710015885787 w_k1_INTC_s006
                                 + 0.0246944061220285 w_k1_INTC_s007
                                 + 0.0097335507236883 w_k1_INTC_s010
                                 + 0.0111915583645886 w_k1_INTC_s011
                                 + 0.0138269055151717 w_k1_INTC_s012
                                 + 0.00656521866073513 w_k1_INTC_s013
                                 + 0.0051751317973196 w_k1_INTC_s014
                                 + 0.00618551069767511 w_k1_INTC_s015
                                 + 0.00659676144147352 w_k1_INTC_s016
                                 + 0.0173959402790055 w_k1_INTC_s017
                                 + 0.0682587535413229 w_k1_INTC_s100
                                 + 0.0549216464880553 w_k1_INTC_s101
                                 + 0.0569207609177619 w_k1_INTC_s102
                                 + 0.0455161314927308 w_k1_INTC_s103
                                 + 0.0430515896216349 w_k1_INTC_s104
                                 + 0.0894665042608676 w_k1_INTC_s105
                                 + 0.0417643195112289 w_k1_INTC_s106
                                 + 0.104404957905274 w_k1_INTC_s107
                                 + 0.0309744253567181 w_k1_INTC_s110
                                 + 0.0138848531207119 w_k1_INTC_s111
                                 + 0.0157386664911548 w_k1_INTC_s112
                                 + 0.024580134195709 w_k1_INTC_s113
                                 + 0.0445118181724716 w_k1_INTC_s114
                                 + 0.0286530999102167 w_k1_INTC_s115
                                 + 0.0233326329538901 w_k1_INTC_s116
                                 + 0.0146291881089968 w_k1_INTC_s117
                                 + 0.0208990992995715 w_k1_KO_s000
                                 + 0.0167572838251636 w_k1_KO_s001
                                 + 0.0500796765806655 w_k1_KO_s002
                                 + 0.0176678512749516 w_k1_KO_s003
                                 + 0.029931001322384 w_k1_KO_s004
                                 + 0.0750870748320143 w_k1_KO_s005
                                 + 0.0233673825220474 w_k1_KO_s006
                                 + 0.0250658845707942 w_k1_KO_s007
                                 + 0.0101388555829655 w_k1_KO_s010
                                 + 0.0126115569158338 w_k1_KO_s011
                                 + 0.0118509486383547 w_k1_KO_s012
                                 + 0.00668112142839169 w_k1_KO_s013
                                 + 0.00613086013509646 w_k1_KO_s014
                                 + 0.00683338372887196 w_k1_KO_s015
                                 + 0.00790962558811428 w_k1_KO_s016
                                 + 0.0171723349913092 w_k1_KO_s017
                                 + 0.058058791540796 w_k1_KO_s100
                                 + 0.0526348203628075 w_k1_KO_s101
                                 + 0.0546054318956832 w_k1_KO_s102
                                 + 0.0480630020600946 w_k1_KO_s103
                                 + 0.0412142548276336 w_k1_KO_s104
                                 + 0.0857932248080876 w_k1_KO_s105
                                 + 0.0400097435644578 w_k1_KO_s106
                                 + 0.0982438689198962 w_k1_KO_s107
                                 + 0.0287743490779853 w_k1_KO_s110
                                 + 0.0131576181947714 w_k1_KO_s111
                                 + 0.0158939932086934 w_k1_KO_s112
                                 + 0.0195064857468424 w_k1_KO_s113
                                 + 0.048626763188749 w_k1_KO_s114
                                 + 0.0315043647703222 w_k1_KO_s115
                                 + 0.024429534820323 w_k1_KO_s116
                                 + 0.0128375540387848 w_k1_KO_s117
                                 + 0.0221958601618152 w_k1_MAT_s000
                                 + 0.0187129922937857 w_k1_MAT_s001
                                 + 0.0348983567604152 w_k1_MAT_s002
                                 + 0.0176942801646311 w_k1_MAT_s003
                                 + 0.0316298618232594 w_k1_MAT_s004
                                 + 0.0884562506029321 w_k1_MAT_s005
                                 + 0.0228159355140957 w_k1_MAT_s006
                                 + 0.0269614609639184 w_k1_MAT_s007
                                 + 0.0119633287726711 w_k1_MAT_s010
                                 + 0.0116083056724028 w_k1_MAT_s011
                                 + 0.0117953917658508 w_k1_MAT_s012
                                 + 0.00618881923708154 w_k1_MAT_s013
                                 + 0.00563433065624642 w_k1_MAT_s014
                                 + 0.00778443104724008 w_k1_MAT_s015
                                 + 0.00885662982399607 w_k1_MAT_s016
                                 + 0.0153536280552806 w_k1_MAT_s017
                                 + 0.0563973666158359 w_k1_MAT_s100
                                 + 0.0623864016306505 w_k1_MAT_s101
                                 + 0.067679514604692 w_k1_MAT_s102
                                 + 0.0467381715334387 w_k1_MAT_s103
                                 + 0.0468982323645084 w_k1_MAT_s104
                                 + 0.0860974533983183 w_k1_MAT_s105
                                 + 0.03677314843819 w_k1_MAT_s106
                                 + 0.104762965424598 w_k1_MAT_s107
                                 + 0.0284231149291707 w_k1_MAT_s110
                                 + 0.0158921857073137 w_k1_MAT_s111
                                 + 0.0149574293139762 w_k1_MAT_s112
                                 + 0.020939589655964 w_k1_MAT_s113
                                 + 0.0467758389258963 w_k1_MAT_s114
                                 + 0.0363519880069906 w_k1_MAT_s115
                                 + 0.0207770532734697 w_k1_MAT_s116
                                 + 0.0111117098757031 w_k1_MAT_s117
                                 + 0.019447954450594 w_k1_MSFT_s000
                                 + 0.0165036524189064 w_k1_MSFT_s001
                                 + 0.0453547058430889 w_k1_MSFT_s002
                                 + 0.0177761461620119 w_k1_MSFT_s003
                                 + 0.0305385827794821 w_k1_MSFT_s004
                                 + 0.0724103730475119 w_k1_MSFT_s005
                                 + 0.0227308295295915 w_k1_MSFT_s006
                                 + 0.0231096048498997 w_k1_MSFT_s007
                                 + 0.0110015196425319 w_k1_MSFT_s010
                                 + 0.0117257125382679 w_k1_MSFT_s011
                                 + 0.0130353517077358 w_k1_MSFT_s012
                                 + 0.00578779389801231 w_k1_MSFT_s013
                                 + 0.00638520841471501 w_k1_MSFT_s014
                                 + 0.00669814253473804 w_k1_MSFT_s015
                                 + 0.00765147752538431 w_k1_MSFT_s016
                                 + 0.016629850270337 w_k1_MSFT_s017
                                 + 0.0617046429622985 w_k1_MSFT_s100
                                 + 0.0569043479051228 w_k1_MSFT_s101
                                 + 0.0573751027337401 w_k1_MSFT_s102
                                 + 0.0431740783347535 w_k1_MSFT_s103
                                 + 0.0438866968955361 w_k1_MSFT_s104
                                 + 0.0917170216033093 w_k1_MSFT_s105
                                 + 0.040901052425809 w_k1_MSFT_s106
                                 + 0.0981173659298919 w_k1_MSFT_s107
                                 + 0.0280356287193199 w_k1_MSFT_s110
                                 + 0.0149879994352739 w_k1_MSFT_s111
                                 + 0.0158631897081517 w_k1_MSFT_s112
                                 + 0.0195801629675506 w_k1_MSFT_s113
                                 + 0.0466498542049647 w_k1_MSFT_s114
                                 + 0.0274742444298885 w_k1_MSFT_s115
                                 + 0.0243682750422875 w_k1_MSFT_s116
                                 + 0.0132058034512383 w_k1_MSFT_s117
                                 + 0.0179949349953226 w_k1_PG_s000
                                 + 0.017222557260633 w_k1_PG_s001
                                 + 0.0500372340330441 w_k1_PG_s002
                                 + 0.0160075766698639 w_k1_PG_s003
                                 + 0.0281596053492554 w_k1_PG_s004
                                 + 0.0788569618132586 w_k1_PG_s005
                                 + 0.0257320345006333 w_k1_PG_s006
                                 + 0.0244530869348785 w_k1_PG_s007
                                 + 0.0105086921084751 w_k1_PG_s010
                                 + 0.0112126182131595 w_k1_PG_s011
                                 + 0.0117036257197626 w_k1_PG_s012
                                 + 0.00635432397831404 w_k1_PG_s013
                                 + 0.00606152769655051 w_k1_PG_s014
                                 + 0.00671274131931429 w_k1_PG_s015
                                 + 0.0072280697753278 w_k1_PG_s016
                                 + 0.0169038055311854 w_k1_PG_s017
                                 + 0.0587864300190802 w_k1_PG_s100
                                 + 0.0613742667207675 w_k1_PG_s101
                                 + 0.0531635933205552 w_k1_PG_s102
                                 + 0.0487941469169878 w_k1_PG_s103
                                 + 0.0403335515649764 w_k1_PG_s104
                                 + 0.086346178755935 w_k1_PG_s105
                                 + 0.0440208376677551 w_k1_PG_s106
                                 + 0.0936436535483093 w_k1_PG_s107
                                 + 0.0281589003029943 w_k1_PG_s110
                                 + 0.0136991033519423 w_k1_PG_s111
                                 + 0.013721823262628 w_k1_PG_s112
                                 + 0.0205975838386528 w_k1_PG_s113
                                 + 0.0454855975403226 w_k1_PG_s114
                                 + 0.0322248687163459 w_k1_PG_s115
                                 + 0.0205618747704563 w_k1_PG_s116
                                 + 0.0134538186185039 w_k1_PG_s117 >= 1.13
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
