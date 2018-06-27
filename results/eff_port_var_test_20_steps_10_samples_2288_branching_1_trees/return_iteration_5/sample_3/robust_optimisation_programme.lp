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
 w_k1_ADM_s0_return_constraint:  1.04197 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.90764 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.17849 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.914562 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.981923 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.15011 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.696336 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.06141 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.820505 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.992043 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.896881 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.34806 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.05586 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.26582 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.3428 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.1332 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.01247 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.23747 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.961173 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.09196 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.653105 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.921669 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.901157 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.24771 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.27402 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.77644 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.998624 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.989217 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.734555 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.00978 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.30293 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.97818 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.892243 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.886515 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.03496 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.15003 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.01664 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.966314 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.13035 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.820489 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.17162 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.841667 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.978289 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.23398 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.73538 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.19786 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.806921 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.02707 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.868985 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.797 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.904761 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.1914 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.18152 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.994624 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.830839 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.42422 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.01272 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.26546 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.525668 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.744671 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.12112 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.46239 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.33352 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.71587 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.05989 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.897361 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.564055 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.06433 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.64109 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.53081 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.911294 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.695477 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.18586 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.27874 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.12179 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.08495 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.22212 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.784125 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.23547 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.742069 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.930712 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.35488 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.681594 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.39352 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.665225 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.02712 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.831591 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.71946 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.899216 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.26233 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.61646 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.964665 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.833624 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.48248 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.800369 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.175 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.526035 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.823835 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.23282 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.85767 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.65909 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.751537 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.00238 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.921269 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.612944 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.52242 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.99529 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.52047 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.733386 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.706021 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.0902 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.43198 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.04063 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.949887 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.0919 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.86238 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.16542 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.903064 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.954547 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.14633 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.81287 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.22224 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.709841 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.971546 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.810578 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.42309 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.03233 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.24551 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.35526 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.999753 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.820327 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.26909 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.78348 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.01311 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.631699 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.970024 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.99988 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.44948 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.51623 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.726832 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.974622 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.995305 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.707773 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.26441 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.56124 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.20777 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.723869 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.795431 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.907727 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.31255 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.07336 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.987943 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.12475 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.854664 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.19017 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.83901 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.919499 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.5349 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.606911 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.16869 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.855507 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.949609 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.3471 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.53617 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.643269 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.729918 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.761922 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.808572 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.971615 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.19891 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.35279 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.67296 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.660149 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.511134 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.56427 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.56321 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.914816 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.24609 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.0558 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.703965 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.648057 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.12397 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.859229 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.43977 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.21215 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.781074 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.56049 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.991579 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.822301 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.15175 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.09224 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.846634 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.11314 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.938829 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.05795 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.1216 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.739684 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.17132 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.855439 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.06638 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.80464 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.40384 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.937653 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.07326 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.21536 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.889335 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.793342 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.08184 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.852624 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.94124 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.725366 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.958913 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.07955 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.17261 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.23805 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.681478 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.86197 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.953086 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.755164 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.20869 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.81123 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.53972 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.0499 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.855975 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.09957 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.1963 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.12649 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.0523 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.10435 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.798043 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.1719 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.880642 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.08776 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.39006 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.693742 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.15453 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.767933 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.06459 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.11307 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.62928 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.773076 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.00639 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.0972 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.851464 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.875432 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.19189 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.917849 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.31086 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.659676 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.756455 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.08285 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.34573 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.10538 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.815969 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.958666 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.800346 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.602783 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.15027 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.47426 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.35181 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.06417 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.833404 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.39065 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.21277 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.957036 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.953798 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.16584 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.855203 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.14015 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.823291 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.864976 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.46277 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.639443 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.0159 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.774062 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.15598 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.56819 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.91529 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.713807 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.886659 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.884935 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.705772 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.875076 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.36382 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.37641 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.64698 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.504563 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.533632 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.55017 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.91671 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.16109 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.2019 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.11017 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.761076 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.509699 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.28584 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.899622 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.91667 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.21048 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.758963 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 2.00455 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.03711 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.800865 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.16363 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.10314 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.857859 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.11582 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.862914 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.00742 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.18001 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.841271 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.22181 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.780454 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.979106 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.847203 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.39228 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.995638 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.07367 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.34163 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.991272 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.863499 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.28076 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.861434 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.08602 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.692721 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.0217 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.92624 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.28354 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.34744 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.72309 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.00901 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.01133 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.737655 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.24949 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.63614 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.29941 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.860338 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.874802 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.98185 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.27824 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.04117 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.946967 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.08589 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.92025 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.05639 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.918648 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.976004 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.11336 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.858266 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.17833 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.848361 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.06307 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.01039 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.29038 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.902645 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.06522 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.16112 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.976398 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.894296 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.19145 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.960133 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.11178 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.75198 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.866016 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.1078 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.24049 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.21133 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.877957 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.988699 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.91321 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.695071 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.14305 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.19361 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.31457 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.940276 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.83258 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.07657 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.17723 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.899097 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.973943 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.286999778669107 z_k1_s0000
                                 - 0.538998608378985 z_k1_s0001
                                 - 0.323999716769319 z_k1_s0002
                                 - 0.400998247681149 z_k1_s0003
                                 - 0.276000037395967 z_k1_s0004
                                 - 0.41400108209351 z_k1_s0005
                                 - 0.84099815860232 z_k1_s0006
                                 - 0.337999822753128 z_k1_s0007
                                 - 0.530001273476853 z_k1_s0010
                                 - 0.645999211408283 z_k1_s0011
                                 - 0.652998017473407 z_k1_s0012
                                 - 0.328999926845304 z_k1_s0013
                                 - 0.424999931505325 z_k1_s0014
                                 - 0.412999717022882 z_k1_s0015
                                 - 0.901998938635084 z_k1_s0016
                                 - 0.380000089745896 z_k1_s0017
                                 - 0.390000817693853 z_k1_s0020
                                 - 1.16200057159437 z_k1_s0021
                                 - 0.367000659441175 z_k1_s0022
                                 - 1.00600114791863 z_k1_s0023
                                 - 0.668002396621204 z_k1_s0024
                                 - 0.310000522558962 z_k1_s0025
                                 - 0.369000791037295 z_k1_s0026
                                 - 0.245000310993169 z_k1_s0027
                                 - 0.287000716579515 z_k1_s0030
                                 - 0.918003669509166 z_k1_s0031
                                 - 0.343001147673916 z_k1_s0032
                                 - 0.69400029792683 z_k1_s0033
                                 - 0.518000590263446 z_k1_s0034
                                 - 0.618002389577834 z_k1_s0035
                                 - 0.197000744401299 z_k1_s0036
                                 - 0.543001040092043 z_k1_s0037
                                 - 1.61500274161441 z_k1_s0040
                                 - 1.63599591770154 z_k1_s0041
                                 - 0.747999603906647 z_k1_s0042
                                 - 1.21899690019697 z_k1_s0043
                                 - 0.426000049468055 z_k1_s0044
                                 - 1.08799726577463 z_k1_s0045
                                 - 0.745999198737093 z_k1_s0046
                                 - 0.434999498832508 z_k1_s0047
                                 - 0.228000361344537 z_k1_s0050
                                 - 0.287000633829732 z_k1_s0051
                                 - 0.283000685932599 z_k1_s0052
                                 - 0.322000802531136 z_k1_s0053
                                 - 0.847001251047251 z_k1_s0054
                                 - 1.04700280991383 z_k1_s0055
                                 - 0.576003323612975 z_k1_s0056
                                 - 0.574000434857487 z_k1_s0057
                                 - 0.677002964273846 z_k1_s0060
                                 - 0.434002531858745 z_k1_s0061
                                 - 0.284000881064253 z_k1_s0062
                                 - 0.298001024866369 z_k1_s0063
                                 - 0.274001053492292 z_k1_s0064
                                 - 0.391002502896571 z_k1_s0065
                                 - 0.916005006465644 z_k1_s0066
                                 - 0.578002360103208 z_k1_s0067
                                 - 0.787996700538083 z_k1_s0070
                                 - 0.540998238301629 z_k1_s0071
                                 - 0.2249993637495 z_k1_s0072
                                 - 0.320998803262321 z_k1_s0073
                                 - 0.531996970508276 z_k1_s0074
                                 - 0.233999161948112 z_k1_s0075
                                 - 0.387998729735758 z_k1_s0076
                                 - 0.64499778418997 z_k1_s0077
                                 - 0.356999425825576 z_k1_s0100
                                 - 0.307000537125828 z_k1_s0101
                                 - 0.122000232631135 z_k1_s0102
                                 - 0.399000595336043 z_k1_s0103
                                 - 0.36400032174488 z_k1_s0104
                                 - 0.309000492673818 z_k1_s0105
                                 - 0.137000109541327 z_k1_s0106
                                 - 0.10800012319467 z_k1_s0107
                                 - 0.72400007751215 z_k1_s0110
                                 - 0.928999923569923 z_k1_s0111
                                 - 0.785003995495384 z_k1_s0112
                                 - 0.899001277221707 z_k1_s0113
                                 - 0.611997854336173 z_k1_s0114
                                 - 0.535000722308309 z_k1_s0115
                                 - 0.985001035157912 z_k1_s0116
                                 - 0.595001047008618 z_k1_s0117
                                 - 0.317000268849073 z_k1_s0120
                                 - 0.990002703978246 z_k1_s0121
                                 - 1.08099905431018 z_k1_s0122
                                 - 0.975003110736547 z_k1_s0123
                                 - 0.947001506016621 z_k1_s0124
                                 - 0.664000906182588 z_k1_s0125
                                 - 0.404000273653016 z_k1_s0126
                                 - 0.532000348651982 z_k1_s0127
                                 - 0.539002105790489 z_k1_s0130
                                 - 0.63100217184368 z_k1_s0131
                                 - 0.443000295143075 z_k1_s0132
                                 - 0.620001591011034 z_k1_s0133
                                 - 1.13399806408946 z_k1_s0134
                                 - 0.318000067892495 z_k1_s0135
                                 - 0.620997810358121 z_k1_s0136
                                 - 0.964999203209582 z_k1_s0137
                                 - 0.215999960190766 z_k1_s0140
                                 - 0.274000379101343 z_k1_s0141
                                 - 0.417001049946306 z_k1_s0142
                                 - 0.2529999905716 z_k1_s0143
                                 - 0.361998798082428 z_k1_s0144
                                 - 0.195000090061136 z_k1_s0145
                                 - 0.493001098339367 z_k1_s0146
                                 - 0.382001266396921 z_k1_s0147
                                 - 0.25400022858955 z_k1_s0150
                                 - 0.318000296907395 z_k1_s0151
                                 - 0.187000229399035 z_k1_s0152
                                 - 0.228000192326045 z_k1_s0153
                                 - 0.690999780313986 z_k1_s0154
                                 - 0.319998980088424 z_k1_s0155
                                 - 0.351000353407234 z_k1_s0156
                                 - 0.374001547998714 z_k1_s0157
                                 - 0.197000724937216 z_k1_s0160
                                 - 0.286000781654738 z_k1_s0161
                                 - 0.100000171265877 z_k1_s0162
                                 - 0.235000378939784 z_k1_s0163
                                 - 0.17599963617119 z_k1_s0164
                                 - 0.321999958393452 z_k1_s0165
                                 - 0.144000158758763 z_k1_s0166
                                 - 0.109000122350733 z_k1_s0167
                                 - 0.132000652475944 z_k1_s0170
                                 - 0.113000081731425 z_k1_s0171
                                 - 0.134000399543621 z_k1_s0172
                                 - 0.174000296899807 z_k1_s0173
                                 - 0.303000089773574 z_k1_s0174
                                 - 0.17200054983213 z_k1_s0175
                                 - 0.119000066334853 z_k1_s0176
                                 - 0.0920000117193606 z_k1_s0177
                                 - 0.893002579623355 z_k1_s1000
                                 - 0.215000476732198 z_k1_s1001
                                 - 0.325001095342766 z_k1_s1002
                                 - 0.28100069801811 z_k1_s1003
                                 - 0.635002157425146 z_k1_s1004
                                 - 0.648000537566103 z_k1_s1005
                                 - 0.571003214557592 z_k1_s1006
                                 - 0.179000696668129 z_k1_s1007
                                 - 0.349000156253537 z_k1_s1010
                                 - 0.365999947586838 z_k1_s1011
                                 - 0.115999919003489 z_k1_s1012
                                 - 0.339999508665438 z_k1_s1013
                                 - 0.159999996201347 z_k1_s1014
                                 - 0.19000037519529 z_k1_s1015
                                 - 0.101000005656474 z_k1_s1016
                                 - 0.218999802646249 z_k1_s1017
                                 - 0.681999985386855 z_k1_s1020
                                 - 0.157000052037118 z_k1_s1021
                                 - 0.604998334392531 z_k1_s1022
                                 - 0.416000394857435 z_k1_s1023
                                 - 0.429999530675265 z_k1_s1024
                                 - 0.2649999267137 z_k1_s1025
                                 - 0.417998441689623 z_k1_s1026
                                 - 0.228999861754902 z_k1_s1027
                                 - 0.266000271393707 z_k1_s1030
                                 - 0.204000306566669 z_k1_s1031
                                 - 0.639002356028884 z_k1_s1032
                                 - 0.580001914415738 z_k1_s1033
                                 - 0.375000130376674 z_k1_s1034
                                 - 0.726001620440558 z_k1_s1035
                                 - 0.435000413061117 z_k1_s1036
                                 - 0.865999825435946 z_k1_s1037
                                 - 0.679000530159411 z_k1_s1040
                                 - 1.43200283362445 z_k1_s1041
                                 - 0.587000794704137 z_k1_s1042
                                 - 0.646002362880611 z_k1_s1043
                                 - 0.290000730156156 z_k1_s1044
                                 - 0.687000258894071 z_k1_s1045
                                 - 0.422000822288741 z_k1_s1046
                                 - 0.967002184633908 z_k1_s1047
                                 - 1.12599720871306 z_k1_s1050
                                 - 1.36799470861742 z_k1_s1051
                                 - 0.402999481551702 z_k1_s1052
                                 - 0.787999959046288 z_k1_s1053
                                 - 0.368999598935796 z_k1_s1054
                                 - 1.15999640113018 z_k1_s1055
                                 - 0.763000958856521 z_k1_s1056
                                 - 0.924997516960887 z_k1_s1057
                                 - 0.23099988413271 z_k1_s1060
                                 - 0.465000477176308 z_k1_s1061
                                 - 0.830999441095578 z_k1_s1062
                                 - 0.246999916890391 z_k1_s1063
                                 - 0.556000302604332 z_k1_s1064
                                 - 0.221999884700266 z_k1_s1065
                                 - 0.201000088624585 z_k1_s1066
                                 - 0.286000015730991 z_k1_s1067
                                 - 0.0950000503004662 z_k1_s1070
                                 - 0.0999999414741933 z_k1_s1071
                                 - 0.46600051798961 z_k1_s1072
                                 - 0.211000420130458 z_k1_s1073
                                 - 0.160999919893449 z_k1_s1074
                                 - 0.120999908003777 z_k1_s1075
                                 - 0.307999509100567 z_k1_s1076
                                 - 0.302999264926898 z_k1_s1077
                                 - 0.0760001835719931 z_k1_s1100
                                 - 0.0579999608835839 z_k1_s1101
                                 - 0.0460002664254185 z_k1_s1102
                                 - 0.0890000417353281 z_k1_s1103
                                 - 0.0330000450614642 z_k1_s1104
                                 - 0.041000083500654 z_k1_s1105
                                 - 0.0719999827520886 z_k1_s1106
                                 - 0.0390000738908566 z_k1_s1107
                                 - 0.0650000878069993 z_k1_s1110
                                 - 0.0300000405263074 z_k1_s1111
                                 - 0.0410000553859534 z_k1_s1112
                                 - 0.111999922097813 z_k1_s1113
                                 - 0.0380000513333227 z_k1_s1114
                                 - 0.126999942360967 z_k1_s1115
                                 - 0.131999949115351 z_k1_s1116
                                 - 0.0280000378245536 z_k1_s1117
                                 - 0.0680000010628751 z_k1_s1120
                                 - 0.0429999972838781 z_k1_s1121
                                 - 0.0429999972838781 z_k1_s1122
                                 - 0.114000342864109 z_k1_s1123
                                 - 0.128000209812396 z_k1_s1124
                                 - 0.131000016729945 z_k1_s1125
                                 - 0.178000063770445 z_k1_s1126
                                 - 0.0629999388670979 z_k1_s1127
                                 - 0.187999302666164 z_k1_s1130
                                 - 0.128999993620813 z_k1_s1131
                                 - 0.142999886995966 z_k1_s1132
                                 - 0.146999353988994 z_k1_s1133
                                 - 0.143999753744223 z_k1_s1134
                                 - 0.0919999178365682 z_k1_s1135
                                 - 0.326999722669103 z_k1_s1136
                                 - 0.18299996892488 z_k1_s1137
                                 - 0.119000246861337 z_k1_s1140
                                 - 0.024000003840063 z_k1_s1141
                                 - 0.0709999325809535 z_k1_s1142
                                 - 0.0340000267317737 z_k1_s1143
                                 - 0.0800001831336862 z_k1_s1144
                                 - 0.054999998154302 z_k1_s1145
                                 - 0.0660003043854197 z_k1_s1146
                                 - 0.0620001930286497 z_k1_s1147
                                 - 0.0400001873895758 z_k1_s1150
                                 - 0.0980000959829194 z_k1_s1151
                                 - 0.093000410760658 z_k1_s1152
                                 - 0.14900087246691 z_k1_s1153
                                 - 0.0750002623550768 z_k1_s1154
                                 - 0.0410001956343303 z_k1_s1155
                                 - 0.122000293857027 z_k1_s1156
                                 - 0.0940000630039013 z_k1_s1157
                                 - 0.230000543608677 z_k1_s1160
                                 - 0.34000073933454 z_k1_s1161
                                 - 0.103000089215996 z_k1_s1162
                                 - 0.0920000696434096 z_k1_s1163
                                 - 0.140000060996474 z_k1_s1164
                                 - 0.332999450424182 z_k1_s1165
                                 - 0.122000082714083 z_k1_s1166
                                 - 0.118000021851302 z_k1_s1167
                                 - 0.0270000595496185 z_k1_s1170
                                 - 0.043000098149399 z_k1_s1171
                                 - 0.0710002774492326 z_k1_s1172
                                 - 0.0689999485486281 z_k1_s1173
                                 - 0.0380000804994567 z_k1_s1174
                                 - 0.105000057748178 z_k1_s1175
                                 - 0.0370000501494159 z_k1_s1176
                                 - 0.0570002101493594 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.785867 w_k1_ADM_s000
                                 + 0.598839 w_k1_AXP_s000
                                 + 0.545442 w_k1_DIS_s000
                                 + 0.667274 w_k1_FDX_s000
                                 + 0.843004 w_k1_IBM_s000
                                 + 1.04918 w_k1_INTC_s000
                                 + 0.759467 w_k1_KO_s000
                                 + 0.625916 w_k1_MAT_s000
                                 + 0.756421 w_k1_MSFT_s000
                                 + 0.697137 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.14418 w_k1_ADM_s000
                                 + 0.891434 w_k1_AXP_s000
                                 + 0.880911 w_k1_DIS_s000
                                 + 0.943208 w_k1_FDX_s000
                                 + 0.851854 w_k1_IBM_s000
                                 + 1.03303 w_k1_INTC_s000
                                 + 0.904174 w_k1_KO_s000
                                 + 0.892354 w_k1_MAT_s000
                                 + 0.937872 w_k1_MSFT_s000
                                 + 0.87413 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.951415 w_k1_ADM_s000
                                 + 0.738533 w_k1_AXP_s000
                                 + 0.795779 w_k1_DIS_s000
                                 + 0.803007 w_k1_FDX_s000
                                 + 1.0556 w_k1_IBM_s000
                                 + 1.03617 w_k1_INTC_s000
                                 + 1.08914 w_k1_KO_s000 + 1.0393 w_k1_MAT_s000
                                 + 0.882108 w_k1_MSFT_s000
                                 + 0.89464 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.825373 w_k1_ADM_s000
                                 + 0.742528 w_k1_AXP_s000
                                 + 0.729957 w_k1_DIS_s000
                                 + 0.875559 w_k1_FDX_s000
                                 + 0.671517 w_k1_IBM_s000
                                 + 1.04911 w_k1_INTC_s000
                                 + 0.760964 w_k1_KO_s000
                                 + 0.627009 w_k1_MAT_s000
                                 + 0.895076 w_k1_MSFT_s000
                                 + 0.750616 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.19485 w_k1_ADM_s000
                                 + 0.835637 w_k1_AXP_s000
                                 + 1.1779 w_k1_DIS_s000 + 1.06232 w_k1_FDX_s000
                                 + 0.877672 w_k1_IBM_s000
                                 + 1.41038 w_k1_INTC_s000
                                 + 0.955938 w_k1_KO_s000
                                 + 0.543646 w_k1_MAT_s000
                                 + 1.0356 w_k1_MSFT_s000
                                 + 0.894229 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.28741 w_k1_ADM_s000
                                 + 1.35833 w_k1_AXP_s000
                                 + 0.967794 w_k1_DIS_s000
                                 + 1.09145 w_k1_FDX_s000
                                 + 1.04259 w_k1_IBM_s000
                                 + 1.3721 w_k1_INTC_s000 + 1.04867 w_k1_KO_s000
                                 + 1.11893 w_k1_MAT_s000
                                 + 0.932903 w_k1_MSFT_s000
                                 + 1.12487 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.34145 w_k1_ADM_s000
                                 + 1.20427 w_k1_AXP_s000
                                 + 1.43503 w_k1_DIS_s000
                                 + 1.25691 w_k1_FDX_s000 + 1.1895 w_k1_IBM_s000
                                 + 1.32418 w_k1_INTC_s000
                                 + 1.33082 w_k1_KO_s000
                                 + 0.977438 w_k1_MAT_s000
                                 + 1.22818 w_k1_MSFT_s000
                                 + 1.04005 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.21057 w_k1_ADM_s000
                                 + 1.28118 w_k1_AXP_s000 + 1.2326 w_k1_DIS_s000
                                 + 1.0224 w_k1_FDX_s000 + 1.58491 w_k1_IBM_s000
                                 + 1.23655 w_k1_INTC_s000
                                 + 1.34729 w_k1_KO_s000 + 1.51271 w_k1_MAT_s000
                                 + 1.07751 w_k1_MSFT_s000
                                 + 1.13912 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.15852 w_k1_ADM_s001
                                 + 0.887782 w_k1_AXP_s001
                                 + 1.2042 w_k1_DIS_s001 + 1.06034 w_k1_FDX_s001
                                 + 0.894906 w_k1_IBM_s001
                                 + 1.10551 w_k1_INTC_s001
                                 + 1.18095 w_k1_KO_s001
                                 + 0.959999 w_k1_MAT_s001
                                 + 1.17724 w_k1_MSFT_s001
                                 + 0.913678 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.29106 w_k1_ADM_s001
                                 + 1.2834 w_k1_AXP_s001 + 1.11577 w_k1_DIS_s001
                                 + 1.25769 w_k1_FDX_s001
                                 + 0.857749 w_k1_IBM_s001
                                 + 1.02895 w_k1_INTC_s001
                                 + 1.02082 w_k1_KO_s001 + 1.21248 w_k1_MAT_s001
                                 + 1.00575 w_k1_MSFT_s001 + 1.0505 w_k1_PG_s001
                                 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.18827 w_k1_ADM_s001
                                 + 1.05373 w_k1_AXP_s001
                                 + 0.773785 w_k1_DIS_s001
                                 + 0.915131 w_k1_FDX_s001
                                 + 1.01429 w_k1_IBM_s001
                                 + 0.846086 w_k1_INTC_s001
                                 + 1.02392 w_k1_KO_s001 + 1.36366 w_k1_MAT_s001
                                 + 0.846322 w_k1_MSFT_s001
                                 + 0.974042 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.02597 w_k1_ADM_s001
                                 + 1.06718 w_k1_AXP_s001
                                 + 1.12429 w_k1_DIS_s001
                                 + 0.931065 w_k1_FDX_s001
                                 + 1.67041 w_k1_IBM_s001
                                 + 1.07302 w_k1_INTC_s001
                                 + 1.31803 w_k1_KO_s001 + 1.90249 w_k1_MAT_s001
                                 + 0.981945 w_k1_MSFT_s001
                                 + 1.08395 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.923597 w_k1_ADM_s001
                                 + 0.728509 w_k1_AXP_s001
                                 + 0.595739 w_k1_DIS_s001
                                 + 0.73318 w_k1_FDX_s001
                                 + 0.786566 w_k1_IBM_s001
                                 + 0.730491 w_k1_INTC_s001
                                 + 0.831735 w_k1_KO_s001
                                 + 0.778755 w_k1_MAT_s001
                                 + 0.726141 w_k1_MSFT_s001
                                 + 0.789336 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.865269 w_k1_ADM_s001
                                 + 0.78845 w_k1_AXP_s001
                                 + 0.856176 w_k1_DIS_s001
                                 + 0.901118 w_k1_FDX_s001
                                 + 0.663748 w_k1_IBM_s001
                                 + 0.852888 w_k1_INTC_s001
                                 + 0.911775 w_k1_KO_s001
                                 + 0.907393 w_k1_MAT_s001
                                 + 0.883997 w_k1_MSFT_s001
                                 + 0.820014 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.19478 w_k1_ADM_s001
                                 + 1.17692 w_k1_AXP_s001
                                 + 1.42736 w_k1_DIS_s001
                                 + 1.14579 w_k1_FDX_s001
                                 + 1.15683 w_k1_IBM_s001
                                 + 1.12364 w_k1_INTC_s001
                                 + 1.40127 w_k1_KO_s001 + 1.44283 w_k1_MAT_s001
                                 + 1.17845 w_k1_MSFT_s001
                                 + 1.06831 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.996147 w_k1_ADM_s001
                                 + 0.766774 w_k1_AXP_s001
                                 + 0.71217 w_k1_DIS_s001
                                 + 0.74756 w_k1_FDX_s001
                                 + 1.22807 w_k1_IBM_s001
                                 + 0.893663 w_k1_INTC_s001
                                 + 0.972284 w_k1_KO_s001
                                 + 1.05615 w_k1_MAT_s001
                                 + 0.845331 w_k1_MSFT_s001
                                 + 0.853222 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.06939 w_k1_ADM_s002
                                 + 0.874819 w_k1_AXP_s002
                                 + 1.29767 w_k1_DIS_s002
                                 + 1.14802 w_k1_FDX_s002
                                 + 0.947471 w_k1_IBM_s002
                                 + 0.881754 w_k1_INTC_s002
                                 + 1.25347 w_k1_KO_s002 + 1.06594 w_k1_MAT_s002
                                 + 1.10418 w_k1_MSFT_s002
                                 + 1.11179 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.863221 w_k1_ADM_s002
                                 + 0.758362 w_k1_AXP_s002
                                 + 0.844381 w_k1_DIS_s002
                                 + 0.88426 w_k1_FDX_s002
                                 + 1.04445 w_k1_IBM_s002
                                 + 0.803311 w_k1_INTC_s002
                                 + 0.999673 w_k1_KO_s002
                                 + 1.15835 w_k1_MAT_s002
                                 + 0.821731 w_k1_MSFT_s002
                                 + 0.935634 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.27453 w_k1_ADM_s002
                                 + 1.35215 w_k1_AXP_s002
                                 + 1.52695 w_k1_DIS_s002
                                 + 1.47779 w_k1_FDX_s002
                                 + 1.05339 w_k1_IBM_s002
                                 + 1.40763 w_k1_INTC_s002
                                 + 1.48772 w_k1_KO_s002 + 1.84737 w_k1_MAT_s002
                                 + 1.44464 w_k1_MSFT_s002
                                 + 1.14629 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.23438 w_k1_ADM_s002
                                 + 1.16981 w_k1_AXP_s002
                                 + 1.32228 w_k1_DIS_s002
                                 + 1.10312 w_k1_FDX_s002 + 1.3683 w_k1_IBM_s002
                                 + 1.01848 w_k1_INTC_s002
                                 + 1.32269 w_k1_KO_s002 + 1.46043 w_k1_MAT_s002
                                 + 1.04024 w_k1_MSFT_s002
                                 + 1.10976 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.827615 w_k1_ADM_s002
                                 + 0.589364 w_k1_AXP_s002
                                 + 0.668843 w_k1_DIS_s002
                                 + 0.763137 w_k1_FDX_s002
                                 + 0.76552 w_k1_IBM_s002
                                 + 0.733829 w_k1_INTC_s002
                                 + 0.866748 w_k1_KO_s002
                                 + 0.888367 w_k1_MAT_s002
                                 + 0.807253 w_k1_MSFT_s002
                                 + 0.860687 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.921587 w_k1_ADM_s002
                                 + 1.17281 w_k1_AXP_s002
                                 + 0.872539 w_k1_DIS_s002
                                 + 1.01128 w_k1_FDX_s002
                                 + 1.02504 w_k1_IBM_s002
                                 + 0.84421 w_k1_INTC_s002 + 1.0455 w_k1_KO_s002
                                 + 1.41039 w_k1_MAT_s002
                                 + 0.879646 w_k1_MSFT_s002
                                 + 1.0565 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.685384 w_k1_ADM_s002
                                 + 0.642335 w_k1_AXP_s002
                                 + 0.938857 w_k1_DIS_s002
                                 + 1.01072 w_k1_FDX_s002
                                 + 0.745728 w_k1_IBM_s002
                                 + 0.81444 w_k1_INTC_s002
                                 + 0.962912 w_k1_KO_s002
                                 + 0.846181 w_k1_MAT_s002
                                 + 1.04646 w_k1_MSFT_s002
                                 + 0.952569 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.996383 w_k1_ADM_s002
                                 + 1.00239 w_k1_AXP_s002
                                 + 1.06213 w_k1_DIS_s002
                                 + 1.24275 w_k1_FDX_s002
                                 + 0.785165 w_k1_IBM_s002
                                 + 1.11163 w_k1_INTC_s002
                                 + 1.05893 w_k1_KO_s002
                                 + 0.893633 w_k1_MAT_s002
                                 + 1.09763 w_k1_MSFT_s002
                                 + 1.09465 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.15341 w_k1_ADM_s003
                                 + 1.14324 w_k1_AXP_s003
                                 + 1.27078 w_k1_DIS_s003
                                 + 1.08962 w_k1_FDX_s003
                                 + 1.08051 w_k1_IBM_s003
                                 + 1.00332 w_k1_INTC_s003
                                 + 1.03921 w_k1_KO_s003
                                 + 0.986382 w_k1_MAT_s003
                                 + 1.19191 w_k1_MSFT_s003
                                 + 1.04316 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.921216 w_k1_ADM_s003
                                 + 0.967195 w_k1_AXP_s003
                                 + 0.967146 w_k1_DIS_s003
                                 + 1.01676 w_k1_FDX_s003 + 1.0861 w_k1_IBM_s003
                                 + 1.04779 w_k1_INTC_s003
                                 + 0.912357 w_k1_KO_s003
                                 + 1.01113 w_k1_MAT_s003
                                 + 1.06839 w_k1_MSFT_s003
                                 + 0.983573 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.964111 w_k1_ADM_s003
                                 + 0.878676 w_k1_AXP_s003
                                 + 0.839893 w_k1_DIS_s003
                                 + 0.97554 w_k1_FDX_s003
                                 + 0.678386 w_k1_IBM_s003
                                 + 0.856206 w_k1_INTC_s003
                                 + 0.799194 w_k1_KO_s003
                                 + 0.816917 w_k1_MAT_s003
                                 + 0.921597 w_k1_MSFT_s003
                                 + 0.927467 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.07802 w_k1_ADM_s003
                                 + 0.995225 w_k1_AXP_s003
                                 + 0.804386 w_k1_DIS_s003
                                 + 0.967868 w_k1_FDX_s003
                                 + 0.916339 w_k1_IBM_s003
                                 + 0.843465 w_k1_INTC_s003
                                 + 0.82629 w_k1_KO_s003
                                 + 0.765459 w_k1_MAT_s003
                                 + 0.925596 w_k1_MSFT_s003
                                 + 0.89012 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.914137 w_k1_ADM_s003
                                 + 0.754541 w_k1_AXP_s003
                                 + 0.600312 w_k1_DIS_s003
                                 + 0.764098 w_k1_FDX_s003
                                 + 0.909244 w_k1_IBM_s003
                                 + 0.698678 w_k1_INTC_s003
                                 + 0.758892 w_k1_KO_s003
                                 + 0.727284 w_k1_MAT_s003
                                 + 0.820128 w_k1_MSFT_s003
                                 + 0.803057 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.22536 w_k1_ADM_s003
                                 + 1.56422 w_k1_AXP_s003
                                 + 1.27389 w_k1_DIS_s003
                                 + 1.20747 w_k1_FDX_s003
                                 + 1.30232 w_k1_IBM_s003
                                 + 1.12879 w_k1_INTC_s003
                                 + 1.33369 w_k1_KO_s003 + 1.34415 w_k1_MAT_s003
                                 + 1.25739 w_k1_MSFT_s003
                                 + 1.19195 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.974742 w_k1_ADM_s003
                                 + 1.18705 w_k1_AXP_s003
                                 + 1.26321 w_k1_DIS_s003
                                 + 1.31446 w_k1_FDX_s003
                                 + 0.720062 w_k1_IBM_s003
                                 + 1.14819 w_k1_INTC_s003
                                 + 0.834508 w_k1_KO_s003
                                 + 0.78601 w_k1_MAT_s003
                                 + 1.30242 w_k1_MSFT_s003
                                 + 1.06276 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.02329 w_k1_ADM_s003
                                 + 1.06051 w_k1_AXP_s003
                                 + 1.39632 w_k1_DIS_s003
                                 + 1.24583 w_k1_FDX_s003 + 1.1693 w_k1_IBM_s003
                                 + 1.17033 w_k1_INTC_s003 + 1.3586 w_k1_KO_s003
                                 + 1.04005 w_k1_MAT_s003
                                 + 1.38748 w_k1_MSFT_s003
                                 + 1.12078 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.0711 w_k1_ADM_s004
                                 + 0.927388 w_k1_AXP_s004
                                 + 0.857565 w_k1_DIS_s004
                                 + 0.861718 w_k1_FDX_s004
                                 + 1.15161 w_k1_IBM_s004
                                 + 1.11147 w_k1_INTC_s004
                                 + 1.01713 w_k1_KO_s004 + 1.01337 w_k1_MAT_s004
                                 + 0.916698 w_k1_MSFT_s004
                                 + 0.990128 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.839861 w_k1_ADM_s004
                                 + 1.59207 w_k1_AXP_s004 + 1.2955 w_k1_DIS_s004
                                 + 1.10179 w_k1_FDX_s004
                                 + 1.42344 w_k1_IBM_s004
                                 + 1.28672 w_k1_INTC_s004
                                 + 1.21722 w_k1_KO_s004 + 1.49979 w_k1_MAT_s004
                                 + 1.1968 w_k1_MSFT_s004 + 1.1587 w_k1_PG_s004
                                 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.802378 w_k1_ADM_s004
                                 + 0.863574 w_k1_AXP_s004
                                 + 0.726646 w_k1_DIS_s004
                                 + 0.917695 w_k1_FDX_s004
                                 + 0.66752 w_k1_IBM_s004
                                 + 0.967886 w_k1_INTC_s004
                                 + 0.720216 w_k1_KO_s004
                                 + 0.478221 w_k1_MAT_s004
                                 + 0.913035 w_k1_MSFT_s004
                                 + 0.823853 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.999652 w_k1_ADM_s004
                                 + 1.02691 w_k1_AXP_s004
                                 + 0.917063 w_k1_DIS_s004
                                 + 1.00942 w_k1_FDX_s004
                                 + 0.832518 w_k1_IBM_s004
                                 + 1.24295 w_k1_INTC_s004
                                 + 0.960835 w_k1_KO_s004
                                 + 0.77031 w_k1_MAT_s004
                                 + 1.09128 w_k1_MSFT_s004
                                 + 0.995066 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.903208 w_k1_ADM_s004
                                 + 0.935141 w_k1_AXP_s004
                                 + 0.808751 w_k1_DIS_s004
                                 + 1.00467 w_k1_FDX_s004
                                 + 0.552709 w_k1_IBM_s004
                                 + 1.03679 w_k1_INTC_s004 + 0.6332 w_k1_KO_s004
                                 + 0.511758 w_k1_MAT_s004
                                 + 0.955656 w_k1_MSFT_s004
                                 + 0.79068 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.2641 w_k1_ADM_s004
                                 + 1.41946 w_k1_AXP_s004
                                 + 1.47672 w_k1_DIS_s004
                                 + 1.29215 w_k1_FDX_s004
                                 + 1.07996 w_k1_IBM_s004
                                 + 1.44358 w_k1_INTC_s004
                                 + 1.11785 w_k1_KO_s004 + 0.7347 w_k1_MAT_s004
                                 + 1.34785 w_k1_MSFT_s004
                                 + 1.09306 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.541295 w_k1_ADM_s004
                                 + 0.723351 w_k1_AXP_s004
                                 + 0.662329 w_k1_DIS_s004
                                 + 0.754344 w_k1_FDX_s004
                                 + 0.861046 w_k1_IBM_s004
                                 + 0.948803 w_k1_INTC_s004
                                 + 0.712336 w_k1_KO_s004
                                 + 0.777998 w_k1_MAT_s004
                                 + 0.815938 w_k1_MSFT_s004
                                 + 0.86271 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.0737 w_k1_ADM_s004
                                 + 1.21991 w_k1_AXP_s004
                                 + 1.13595 w_k1_DIS_s004
                                 + 1.27368 w_k1_FDX_s004
                                 + 0.72861 w_k1_IBM_s004
                                 + 1.22708 w_k1_INTC_s004
                                 + 0.889828 w_k1_KO_s004
                                 + 0.91055 w_k1_MAT_s004
                                 + 1.18012 w_k1_MSFT_s004
                                 + 0.996686 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.944795 w_k1_ADM_s005
                                 + 1.16307 w_k1_AXP_s005
                                 + 1.22373 w_k1_DIS_s005
                                 + 0.984399 w_k1_FDX_s005
                                 + 2.13398 w_k1_IBM_s005
                                 + 0.915568 w_k1_INTC_s005
                                 + 1.39777 w_k1_KO_s005 + 1.63638 w_k1_MAT_s005
                                 + 0.784708 w_k1_MSFT_s005
                                 + 1.1614 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.03662 w_k1_ADM_s005
                                 + 1.25751 w_k1_AXP_s005
                                 + 1.74604 w_k1_DIS_s005
                                 + 1.27449 w_k1_FDX_s005
                                 + 1.59661 w_k1_IBM_s005
                                 + 1.28401 w_k1_INTC_s005
                                 + 1.36403 w_k1_KO_s005 + 1.77561 w_k1_MAT_s005
                                 + 1.27096 w_k1_MSFT_s005
                                 + 1.35622 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.789687 w_k1_ADM_s005
                                 + 0.794094 w_k1_AXP_s005
                                 + 0.668534 w_k1_DIS_s005
                                 + 0.866984 w_k1_FDX_s005
                                 + 0.527857 w_k1_IBM_s005
                                 + 0.748426 w_k1_INTC_s005
                                 + 0.540474 w_k1_KO_s005
                                 + 0.603459 w_k1_MAT_s005
                                 + 0.77353 w_k1_MSFT_s005
                                 + 0.852582 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.51976 w_k1_ADM_s005
                                 + 1.37876 w_k1_AXP_s005
                                 + 1.78857 w_k1_DIS_s005
                                 + 1.45203 w_k1_FDX_s005
                                 + 0.763935 w_k1_IBM_s005
                                 + 1.33655 w_k1_INTC_s005
                                 + 1.05418 w_k1_KO_s005
                                 + 0.700698 w_k1_MAT_s005
                                 + 1.35644 w_k1_MSFT_s005 + 1.3431 w_k1_PG_s005
                                 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.802732 w_k1_ADM_s005
                                 + 0.66023 w_k1_AXP_s005
                                 + 0.631126 w_k1_DIS_s005
                                 + 0.757168 w_k1_FDX_s005
                                 + 0.944828 w_k1_IBM_s005
                                 + 0.760149 w_k1_INTC_s005
                                 + 0.831778 w_k1_KO_s005
                                 + 0.952165 w_k1_MAT_s005
                                 + 0.757132 w_k1_MSFT_s005
                                 + 0.890356 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.89219 w_k1_ADM_s005
                                 + 1.01287 w_k1_AXP_s005
                                 + 0.840539 w_k1_DIS_s005
                                 + 0.880132 w_k1_FDX_s005
                                 + 1.33931 w_k1_IBM_s005
                                 + 0.919817 w_k1_INTC_s005
                                 + 1.06476 w_k1_KO_s005 + 1.43507 w_k1_MAT_s005
                                 + 0.907144 w_k1_MSFT_s005
                                 + 1.06322 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.938721 w_k1_ADM_s005
                                 + 1.22539 w_k1_AXP_s005
                                 + 1.21068 w_k1_DIS_s005
                                 + 1.14911 w_k1_FDX_s005
                                 + 0.798209 w_k1_IBM_s005
                                 + 1.24155 w_k1_INTC_s005
                                 + 0.919645 w_k1_KO_s005
                                 + 0.730156 w_k1_MAT_s005
                                 + 1.22577 w_k1_MSFT_s005
                                 + 1.15482 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.999748 w_k1_ADM_s005
                                 + 0.78634 w_k1_AXP_s005
                                 + 1.01265 w_k1_DIS_s005
                                 + 0.93446 w_k1_FDX_s005
                                 + 0.769414 w_k1_IBM_s005
                                 + 1.09929 w_k1_INTC_s005
                                 + 0.941079 w_k1_KO_s005
                                 + 0.590495 w_k1_MAT_s005
                                 + 0.97823 w_k1_MSFT_s005
                                 + 0.911228 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.825419 w_k1_ADM_s006
                                 + 1.23932 w_k1_AXP_s006
                                 + 0.854786 w_k1_DIS_s006
                                 + 0.933246 w_k1_FDX_s006
                                 + 1.25017 w_k1_IBM_s006
                                 + 1.19282 w_k1_INTC_s006
                                 + 1.06404 w_k1_KO_s006 + 1.2527 w_k1_MAT_s006
                                 + 0.995613 w_k1_MSFT_s006
                                 + 1.10464 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.981448 w_k1_ADM_s006
                                 + 1.44826 w_k1_AXP_s006
                                 + 1.65107 w_k1_DIS_s006
                                 + 1.35022 w_k1_FDX_s006
                                 + 1.24568 w_k1_IBM_s006
                                 + 1.38351 w_k1_INTC_s006
                                 + 1.39642 w_k1_KO_s006 + 1.09534 w_k1_MAT_s006
                                 + 1.20707 w_k1_MSFT_s006
                                 + 1.24315 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.917836 w_k1_ADM_s006
                                 + 0.769677 w_k1_AXP_s006
                                 + 0.701139 w_k1_DIS_s006
                                 + 0.878473 w_k1_FDX_s006
                                 + 0.653904 w_k1_IBM_s006
                                 + 0.976587 w_k1_INTC_s006
                                 + 0.78672 w_k1_KO_s006
                                 + 0.487376 w_k1_MAT_s006
                                 + 0.858436 w_k1_MSFT_s006
                                 + 0.837537 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.625633 w_k1_ADM_s006
                                 + 0.871813 w_k1_AXP_s006
                                 + 0.566983 w_k1_DIS_s006
                                 + 0.696489 w_k1_FDX_s006
                                 + 0.889943 w_k1_IBM_s006
                                 + 0.929468 w_k1_INTC_s006
                                 + 0.846194 w_k1_KO_s006
                                 + 1.00832 w_k1_MAT_s006
                                 + 0.800308 w_k1_MSFT_s006
                                 + 0.869757 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.939998 w_k1_ADM_s006
                                 + 0.892913 w_k1_AXP_s006
                                 + 1.34882 w_k1_DIS_s006
                                 + 1.00317 w_k1_FDX_s006 + 1.0548 w_k1_IBM_s006
                                 + 1.09718 w_k1_INTC_s006
                                 + 1.29723 w_k1_KO_s006
                                 + 0.902496 w_k1_MAT_s006
                                 + 1.20149 w_k1_MSFT_s006
                                 + 1.10169 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.23113 w_k1_ADM_s006
                                 + 1.52666 w_k1_AXP_s006
                                 + 1.18894 w_k1_DIS_s006
                                 + 1.09536 w_k1_FDX_s006
                                 + 1.44915 w_k1_IBM_s006
                                 + 1.12741 w_k1_INTC_s006
                                 + 1.34877 w_k1_KO_s006 + 1.82433 w_k1_MAT_s006
                                 + 1.12881 w_k1_MSFT_s006
                                 + 1.16274 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.812112 w_k1_ADM_s006
                                 + 0.831073 w_k1_AXP_s006
                                 + 0.858848 w_k1_DIS_s006
                                 + 0.90496 w_k1_FDX_s006
                                 + 0.906931 w_k1_IBM_s006
                                 + 1.00689 w_k1_INTC_s006
                                 + 0.914535 w_k1_KO_s006
                                 + 0.728933 w_k1_MAT_s006
                                 + 0.862106 w_k1_MSFT_s006
                                 + 1.04346 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.07462 w_k1_ADM_s006
                                 + 1.31186 w_k1_AXP_s006
                                 + 1.05354 w_k1_DIS_s006
                                 + 1.22338 w_k1_FDX_s006
                                 + 0.722802 w_k1_IBM_s006
                                 + 1.15537 w_k1_INTC_s006
                                 + 0.987064 w_k1_KO_s006
                                 + 0.975706 w_k1_MAT_s006
                                 + 1.16319 w_k1_MSFT_s006
                                 + 1.05832 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.964837 w_k1_ADM_s007
                                 + 1.07108 w_k1_AXP_s007
                                 + 0.84248 w_k1_DIS_s007
                                 + 0.975315 w_k1_FDX_s007
                                 + 0.929927 w_k1_IBM_s007
                                 + 1.09997 w_k1_INTC_s007
                                 + 1.04125 w_k1_KO_s007 + 0.99553 w_k1_MAT_s007
                                 + 1.11143 w_k1_MSFT_s007
                                 + 0.991786 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.698512 w_k1_ADM_s007
                                 + 0.972865 w_k1_AXP_s007
                                 + 0.822825 w_k1_DIS_s007
                                 + 0.868085 w_k1_FDX_s007
                                 + 1.31906 w_k1_IBM_s007
                                 + 1.14714 w_k1_INTC_s007
                                 + 1.17549 w_k1_KO_s007 + 1.14165 w_k1_MAT_s007
                                 + 1.09695 w_k1_MSFT_s007
                                 + 0.892004 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.663788 w_k1_ADM_s007
                                 + 0.667426 w_k1_AXP_s007
                                 + 0.60415 w_k1_DIS_s007
                                 + 0.738291 w_k1_FDX_s007
                                 + 0.668192 w_k1_IBM_s007
                                 + 0.815555 w_k1_INTC_s007
                                 + 0.663823 w_k1_KO_s007
                                 + 0.709552 w_k1_MAT_s007
                                 + 0.860387 w_k1_MSFT_s007
                                 + 0.784814 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.11315 w_k1_ADM_s007
                                 + 1.38725 w_k1_AXP_s007 + 1.1524 w_k1_DIS_s007
                                 + 1.33728 w_k1_FDX_s007
                                 + 0.809179 w_k1_IBM_s007
                                 + 0.986783 w_k1_INTC_s007
                                 + 1.10287 w_k1_KO_s007 + 1.03423 w_k1_MAT_s007
                                 + 1.26306 w_k1_MSFT_s007
                                 + 1.25623 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.08947 w_k1_ADM_s007
                                 + 1.40082 w_k1_AXP_s007
                                 + 1.52439 w_k1_DIS_s007
                                 + 1.37396 w_k1_FDX_s007
                                 + 1.14203 w_k1_IBM_s007
                                 + 1.43652 w_k1_INTC_s007
                                 + 1.35499 w_k1_KO_s007 + 1.30345 w_k1_MAT_s007
                                 + 1.46698 w_k1_MSFT_s007
                                 + 1.21922 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.681912 w_k1_ADM_s007
                                 + 0.602407 w_k1_AXP_s007
                                 + 0.677265 w_k1_DIS_s007
                                 + 0.738659 w_k1_FDX_s007
                                 + 0.912831 w_k1_IBM_s007
                                 + 0.86068 w_k1_INTC_s007
                                 + 0.869651 w_k1_KO_s007
                                 + 0.703743 w_k1_MAT_s007
                                 + 0.92822 w_k1_MSFT_s007
                                 + 0.937587 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.08623 w_k1_ADM_s007
                                 + 1.3034 w_k1_AXP_s007 + 1.19436 w_k1_DIS_s007
                                 + 1.08654 w_k1_FDX_s007
                                 + 1.53821 w_k1_IBM_s007
                                 + 1.15063 w_k1_INTC_s007
                                 + 1.34508 w_k1_KO_s007 + 2.13823 w_k1_MAT_s007
                                 + 1.2533 w_k1_MSFT_s007 + 1.09833 w_k1_PG_s007
                                 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.947664 w_k1_ADM_s007
                                 + 0.905778 w_k1_AXP_s007
                                 + 1.10558 w_k1_DIS_s007
                                 + 1.01754 w_k1_FDX_s007
                                 + 0.965164 w_k1_IBM_s007
                                 + 1.25515 w_k1_INTC_s007
                                 + 1.10326 w_k1_KO_s007
                                 + 0.717662 w_k1_MAT_s007
                                 + 1.20209 w_k1_MSFT_s007
                                 + 1.02122 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.15575 w_k1_ADM_s010
                                 + 1.08639 w_k1_AXP_s010
                                 + 0.935429 w_k1_DIS_s010
                                 + 1.11217 w_k1_FDX_s010
                                 + 0.80661 w_k1_IBM_s010
                                 + 0.984957 w_k1_INTC_s010
                                 + 0.962413 w_k1_KO_s010
                                 + 1.25644 w_k1_MAT_s010
                                 + 1.03296 w_k1_MSFT_s010
                                 + 0.928606 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.16409 w_k1_ADM_s010
                                 + 1.32819 w_k1_AXP_s010
                                 + 1.33137 w_k1_DIS_s010
                                 + 1.35398 w_k1_FDX_s010
                                 + 0.873545 w_k1_IBM_s010
                                 + 1.25257 w_k1_INTC_s010
                                 + 0.914697 w_k1_KO_s010
                                 + 0.771178 w_k1_MAT_s010
                                 + 1.18707 w_k1_MSFT_s010
                                 + 0.930155 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.55694 w_k1_ADM_s010
                                 + 1.62276 w_k1_AXP_s010
                                 + 1.80808 w_k1_DIS_s010
                                 + 1.58724 w_k1_FDX_s010 + 1.4581 w_k1_IBM_s010
                                 + 1.05436 w_k1_INTC_s010
                                 + 1.56047 w_k1_KO_s010 + 1.6465 w_k1_MAT_s010
                                 + 1.55234 w_k1_MSFT_s010
                                 + 1.32334 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.896342 w_k1_ADM_s010
                                 + 0.773745 w_k1_AXP_s010
                                 + 0.684429 w_k1_DIS_s010
                                 + 0.809287 w_k1_FDX_s010
                                 + 1.00536 w_k1_IBM_s010
                                 + 0.795314 w_k1_INTC_s010
                                 + 0.929017 w_k1_KO_s010
                                 + 0.725284 w_k1_MAT_s010
                                 + 0.858008 w_k1_MSFT_s010
                                 + 0.846114 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.905762 w_k1_ADM_s010
                                 + 0.831569 w_k1_AXP_s010
                                 + 0.913642 w_k1_DIS_s010
                                 + 0.936914 w_k1_FDX_s010
                                 + 0.922352 w_k1_IBM_s010
                                 + 0.824363 w_k1_INTC_s010
                                 + 0.890333 w_k1_KO_s010
                                 + 0.728441 w_k1_MAT_s010
                                 + 0.869792 w_k1_MSFT_s010
                                 + 0.936301 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.05084 w_k1_ADM_s010
                                 + 1.24309 w_k1_AXP_s010
                                 + 1.25859 w_k1_DIS_s010
                                 + 1.17852 w_k1_FDX_s010
                                 + 1.39636 w_k1_IBM_s010
                                 + 1.12546 w_k1_INTC_s010
                                 + 1.15911 w_k1_KO_s010 + 1.01341 w_k1_MAT_s010
                                 + 1.15635 w_k1_MSFT_s010
                                 + 1.01192 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.973398 w_k1_ADM_s010
                                 + 1.16143 w_k1_AXP_s010
                                 + 0.824437 w_k1_DIS_s010
                                 + 1.04833 w_k1_FDX_s010
                                 + 0.983057 w_k1_IBM_s010
                                 + 1.01663 w_k1_INTC_s010
                                 + 0.995252 w_k1_KO_s010
                                 + 1.19262 w_k1_MAT_s010
                                 + 1.05109 w_k1_MSFT_s010
                                 + 0.910038 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.8463 w_k1_ADM_s010
                                 + 0.883663 w_k1_AXP_s010
                                 + 1.15776 w_k1_DIS_s010
                                 + 0.895912 w_k1_FDX_s010
                                 + 1.58277 w_k1_IBM_s010
                                 + 0.946609 w_k1_INTC_s010
                                 + 1.32315 w_k1_KO_s010 + 1.45117 w_k1_MAT_s010
                                 + 1.00654 w_k1_MSFT_s010
                                 + 1.00395 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.11216 w_k1_ADM_s011
                                 + 1.58211 w_k1_AXP_s011
                                 + 1.57358 w_k1_DIS_s011
                                 + 1.38844 w_k1_FDX_s011
                                 + 1.24164 w_k1_IBM_s011
                                 + 1.48263 w_k1_INTC_s011
                                 + 1.30144 w_k1_KO_s011 + 1.96764 w_k1_MAT_s011
                                 + 1.34034 w_k1_MSFT_s011
                                 + 1.25552 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.84186 w_k1_ADM_s011
                                 + 0.970743 w_k1_AXP_s011
                                 + 0.962447 w_k1_DIS_s011
                                 + 0.958235 w_k1_FDX_s011
                                 + 1.20612 w_k1_IBM_s011
                                 + 1.08336 w_k1_INTC_s011
                                 + 1.04879 w_k1_KO_s011 + 1.58332 w_k1_MAT_s011
                                 + 1.07921 w_k1_MSFT_s011
                                 + 0.932298 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.0336 w_k1_ADM_s011
                                 + 0.844913 w_k1_AXP_s011
                                 + 1.06803 w_k1_DIS_s011
                                 + 1.00646 w_k1_FDX_s011
                                 + 1.07238 w_k1_IBM_s011
                                 + 0.9251 w_k1_INTC_s011 + 1.05234 w_k1_KO_s011
                                 + 0.816294 w_k1_MAT_s011
                                 + 1.00925 w_k1_MSFT_s011
                                 + 0.88739 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.789064 w_k1_ADM_s011
                                 + 0.775209 w_k1_AXP_s011
                                 + 0.780777 w_k1_DIS_s011
                                 + 0.862861 w_k1_FDX_s011
                                 + 0.655621 w_k1_IBM_s011
                                 + 0.963255 w_k1_INTC_s011
                                 + 0.81863 w_k1_KO_s011
                                 + 0.612743 w_k1_MAT_s011
                                 + 0.980326 w_k1_MSFT_s011
                                 + 0.797627 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.966639 w_k1_ADM_s011
                                 + 1.37511 w_k1_AXP_s011
                                 + 1.04015 w_k1_DIS_s011
                                 + 1.04392 w_k1_FDX_s011
                                 + 0.988907 w_k1_IBM_s011
                                 + 0.950698 w_k1_INTC_s011
                                 + 0.958379 w_k1_KO_s011
                                 + 1.63175 w_k1_MAT_s011
                                 + 0.957472 w_k1_MSFT_s011
                                 + 0.903017 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.894963 w_k1_ADM_s011
                                 + 0.979751 w_k1_AXP_s011
                                 + 0.806357 w_k1_DIS_s011
                                 + 0.991213 w_k1_FDX_s011
                                 + 0.677772 w_k1_IBM_s011
                                 + 1.16266 w_k1_INTC_s011
                                 + 0.773468 w_k1_KO_s011
                                 + 0.946971 w_k1_MAT_s011
                                 + 0.982444 w_k1_MSFT_s011
                                 + 0.862488 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.03977 w_k1_ADM_s011
                                 + 1.16232 w_k1_AXP_s011
                                 + 1.40845 w_k1_DIS_s011
                                 + 1.22034 w_k1_FDX_s011
                                 + 0.873431 w_k1_IBM_s011
                                 + 1.33148 w_k1_INTC_s011
                                 + 1.07455 w_k1_KO_s011 + 1.02557 w_k1_MAT_s011
                                 + 1.42627 w_k1_MSFT_s011
                                 + 1.03548 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.801872 w_k1_ADM_s011
                                 + 0.639927 w_k1_AXP_s011
                                 + 0.647595 w_k1_DIS_s011
                                 + 0.74978 w_k1_FDX_s011
                                 + 0.928907 w_k1_IBM_s011
                                 + 0.961911 w_k1_INTC_s011
                                 + 0.867689 w_k1_KO_s011
                                 + 1.00634 w_k1_MAT_s011
                                 + 0.813133 w_k1_MSFT_s011
                                 + 0.788984 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.886513 w_k1_ADM_s012
                                 + 0.862844 w_k1_AXP_s012
                                 + 0.66335 w_k1_DIS_s012
                                 + 0.769944 w_k1_FDX_s012
                                 + 1.38501 w_k1_IBM_s012
                                 + 0.950206 w_k1_INTC_s012
                                 + 0.999752 w_k1_KO_s012
                                 + 1.64788 w_k1_MAT_s012
                                 + 0.880952 w_k1_MSFT_s012
                                 + 0.850845 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.628127 w_k1_ADM_s012
                                 + 0.64366 w_k1_AXP_s012
                                 + 0.62329 w_k1_DIS_s012
                                 + 0.750308 w_k1_FDX_s012
                                 + 0.822808 w_k1_IBM_s012
                                 + 0.821055 w_k1_INTC_s012
                                 + 0.711949 w_k1_KO_s012
                                 + 0.758248 w_k1_MAT_s012
                                 + 0.773935 w_k1_MSFT_s012
                                 + 0.786435 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.902575 w_k1_ADM_s012
                                 + 0.857553 w_k1_AXP_s012
                                 + 0.929821 w_k1_DIS_s012
                                 + 0.987091 w_k1_FDX_s012
                                 + 0.905383 w_k1_IBM_s012
                                 + 1.02856 w_k1_INTC_s012
                                 + 1.02564 w_k1_KO_s012 + 1.00384 w_k1_MAT_s012
                                 + 1.00914 w_k1_MSFT_s012
                                 + 0.954056 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.21651 w_k1_ADM_s012
                                 + 1.1837 w_k1_AXP_s012 + 1.31484 w_k1_DIS_s012
                                 + 1.23767 w_k1_FDX_s012
                                 + 1.00887 w_k1_IBM_s012
                                 + 1.13556 w_k1_INTC_s012
                                 + 1.07553 w_k1_KO_s012 + 0.95027 w_k1_MAT_s012
                                 + 1.11674 w_k1_MSFT_s012 + 1.0487 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.979703 w_k1_ADM_s012
                                 + 1.28564 w_k1_AXP_s012
                                 + 1.31925 w_k1_DIS_s012
                                 + 1.18762 w_k1_FDX_s012
                                 + 1.55726 w_k1_IBM_s012
                                 + 1.26881 w_k1_INTC_s012
                                 + 1.31369 w_k1_KO_s012 + 1.46591 w_k1_MAT_s012
                                 + 1.26372 w_k1_MSFT_s012
                                 + 1.11576 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.791381 w_k1_ADM_s012
                                 + 0.988344 w_k1_AXP_s012
                                 + 1.10457 w_k1_DIS_s012
                                 + 0.92992 w_k1_FDX_s012 + 1.1068 w_k1_IBM_s012
                                 + 0.986853 w_k1_INTC_s012
                                 + 1.11025 w_k1_KO_s012 + 1.20267 w_k1_MAT_s012
                                 + 1.06388 w_k1_MSFT_s012
                                 + 0.992377 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.780197 w_k1_ADM_s012
                                 + 0.824521 w_k1_AXP_s012
                                 + 0.700028 w_k1_DIS_s012
                                 + 0.814719 w_k1_FDX_s012
                                 + 0.981466 w_k1_IBM_s012
                                 + 0.935128 w_k1_INTC_s012
                                 + 0.832151 w_k1_KO_s012
                                 + 0.943655 w_k1_MAT_s012
                                 + 0.88871 w_k1_MSFT_s012
                                 + 0.890909 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.796069 w_k1_ADM_s012
                                 + 0.887739 w_k1_AXP_s012
                                 + 0.873565 w_k1_DIS_s012
                                 + 0.951771 w_k1_FDX_s012
                                 + 0.662813 w_k1_IBM_s012
                                 + 0.906579 w_k1_INTC_s012
                                 + 0.755882 w_k1_KO_s012
                                 + 0.845764 w_k1_MAT_s012
                                 + 0.921522 w_k1_MSFT_s012
                                 + 0.913876 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.29961 w_k1_ADM_s013
                                 + 1.29687 w_k1_AXP_s013
                                 + 1.47297 w_k1_DIS_s013
                                 + 1.22663 w_k1_FDX_s013
                                 + 1.66198 w_k1_IBM_s013
                                 + 1.17037 w_k1_INTC_s013
                                 + 1.34166 w_k1_KO_s013 + 1.67869 w_k1_MAT_s013
                                 + 1.05969 w_k1_MSFT_s013
                                 + 1.14626 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.3249 w_k1_ADM_s013
                                 + 1.23689 w_k1_AXP_s013
                                 + 0.99251 w_k1_DIS_s013 + 1.169 w_k1_FDX_s013
                                 + 0.94864 w_k1_IBM_s013
                                 + 1.20702 w_k1_INTC_s013
                                 + 1.00375 w_k1_KO_s013 + 1.30562 w_k1_MAT_s013
                                 + 1.07399 w_k1_MSFT_s013
                                 + 1.02948 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.867225 w_k1_ADM_s013
                                 + 0.710182 w_k1_AXP_s013
                                 + 0.716103 w_k1_DIS_s013
                                 + 0.866019 w_k1_FDX_s013
                                 + 0.651166 w_k1_IBM_s013
                                 + 0.904826 w_k1_INTC_s013
                                 + 0.745315 w_k1_KO_s013
                                 + 0.45964 w_k1_MAT_s013
                                 + 0.85855 w_k1_MSFT_s013
                                 + 0.764214 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.23855 w_k1_ADM_s013
                                 + 0.913478 w_k1_AXP_s013
                                 + 1.00056 w_k1_DIS_s013
                                 + 1.10122 w_k1_FDX_s013
                                 + 0.773057 w_k1_IBM_s013
                                 + 1.10969 w_k1_INTC_s013
                                 + 0.837523 w_k1_KO_s013
                                 + 0.676592 w_k1_MAT_s013
                                 + 0.944303 w_k1_MSFT_s013
                                 + 0.915164 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.898737 w_k1_ADM_s013
                                 + 0.710576 w_k1_AXP_s013
                                 + 0.880633 w_k1_DIS_s013
                                 + 0.83123 w_k1_FDX_s013
                                 + 1.17305 w_k1_IBM_s013
                                 + 0.977329 w_k1_INTC_s013
                                 + 0.951977 w_k1_KO_s013
                                 + 0.951553 w_k1_MAT_s013
                                 + 0.94708 w_k1_MSFT_s013
                                 + 0.887724 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.20723 w_k1_ADM_s013
                                 + 0.943302 w_k1_AXP_s013
                                 + 1.06731 w_k1_DIS_s013
                                 + 0.989798 w_k1_FDX_s013
                                 + 1.13808 w_k1_IBM_s013
                                 + 1.15099 w_k1_INTC_s013
                                 + 1.19007 w_k1_KO_s013 + 1.17323 w_k1_MAT_s013
                                 + 1.10023 w_k1_MSFT_s013
                                 + 0.982247 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.48135 w_k1_ADM_s013
                                 + 1.17342 w_k1_AXP_s013
                                 + 1.51157 w_k1_DIS_s013
                                 + 1.25214 w_k1_FDX_s013
                                 + 1.08614 w_k1_IBM_s013
                                 + 1.06079 w_k1_INTC_s013
                                 + 1.22369 w_k1_KO_s013 + 1.08426 w_k1_MAT_s013
                                 + 1.23412 w_k1_MSFT_s013
                                 + 1.10838 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.853671 w_k1_ADM_s013
                                 + 0.790117 w_k1_AXP_s013
                                 + 0.63003 w_k1_DIS_s013
                                 + 0.803535 w_k1_FDX_s013
                                 + 0.79695 w_k1_IBM_s013
                                 + 1.02281 w_k1_INTC_s013
                                 + 0.729026 w_k1_KO_s013
                                 + 0.643774 w_k1_MAT_s013
                                 + 0.772646 w_k1_MSFT_s013
                                 + 0.780989 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.796191 w_k1_ADM_s014
                                 + 0.734254 w_k1_AXP_s014
                                 + 0.638635 w_k1_DIS_s014
                                 + 0.764837 w_k1_FDX_s014
                                 + 0.995105 w_k1_IBM_s014
                                 + 0.714478 w_k1_INTC_s014
                                 + 0.923316 w_k1_KO_s014
                                 + 0.97886 w_k1_MAT_s014
                                 + 0.764132 w_k1_MSFT_s014
                                 + 0.805085 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.12346 w_k1_ADM_s014
                                 + 1.04948 w_k1_AXP_s014
                                 + 1.29687 w_k1_DIS_s014
                                 + 1.12698 w_k1_FDX_s014
                                 + 1.31278 w_k1_IBM_s014
                                 + 1.1432 w_k1_INTC_s014 + 1.33391 w_k1_KO_s014
                                 + 1.00507 w_k1_MAT_s014
                                 + 1.22688 w_k1_MSFT_s014
                                 + 1.02568 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.662085 w_k1_ADM_s014
                                 + 0.714473 w_k1_AXP_s014
                                 + 0.757705 w_k1_DIS_s014
                                 + 0.891531 w_k1_FDX_s014
                                 + 0.716772 w_k1_IBM_s014
                                 + 0.861061 w_k1_INTC_s014
                                 + 0.764972 w_k1_KO_s014
                                 + 0.622407 w_k1_MAT_s014
                                 + 0.876403 w_k1_MSFT_s014
                                 + 0.809345 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.961642 w_k1_ADM_s014
                                 + 1.43296 w_k1_AXP_s014 + 1.3691 w_k1_DIS_s014
                                 + 1.24949 w_k1_FDX_s014
                                 + 1.16644 w_k1_IBM_s014
                                 + 1.20439 w_k1_INTC_s014
                                 + 1.25469 w_k1_KO_s014 + 1.56201 w_k1_MAT_s014
                                 + 1.18531 w_k1_MSFT_s014
                                 + 1.16183 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.895345 w_k1_ADM_s014
                                 + 0.870917 w_k1_AXP_s014
                                 + 0.907476 w_k1_DIS_s014
                                 + 0.82292 w_k1_FDX_s014
                                 + 1.46171 w_k1_IBM_s014
                                 + 0.98192 w_k1_INTC_s014
                                 + 1.23887 w_k1_KO_s014 + 1.26879 w_k1_MAT_s014
                                 + 0.882255 w_k1_MSFT_s014
                                 + 1.05411 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.80022 w_k1_ADM_s014
                                 + 0.973309 w_k1_AXP_s014
                                 + 1.02973 w_k1_DIS_s014
                                 + 0.96499 w_k1_FDX_s014
                                 + 1.06586 w_k1_IBM_s014
                                 + 0.984646 w_k1_INTC_s014
                                 + 1.21683 w_k1_KO_s014 + 1.12167 w_k1_MAT_s014
                                 + 1.02491 w_k1_MSFT_s014
                                 + 0.930347 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.40157 w_k1_ADM_s014
                                 + 1.30964 w_k1_AXP_s014
                                 + 1.33268 w_k1_DIS_s014
                                 + 1.28429 w_k1_FDX_s014
                                 + 0.92612 w_k1_IBM_s014
                                 + 1.0534 w_k1_INTC_s014 + 1.20595 w_k1_KO_s014
                                 + 0.937596 w_k1_MAT_s014
                                 + 1.23745 w_k1_MSFT_s014 + 1.0876 w_k1_PG_s014
                                 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.15208 w_k1_ADM_s014
                                 + 0.803551 w_k1_AXP_s014
                                 + 0.922402 w_k1_DIS_s014
                                 + 0.99911 w_k1_FDX_s014
                                 + 0.826065 w_k1_IBM_s014
                                 + 0.971267 w_k1_INTC_s014
                                 + 1.09607 w_k1_KO_s014
                                 + 0.928409 w_k1_MAT_s014
                                 + 0.992601 w_k1_MSFT_s014
                                 + 0.883505 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.28079 w_k1_ADM_s015
                                 + 1.20574 w_k1_AXP_s015
                                 + 0.877201 w_k1_DIS_s015
                                 + 1.03212 w_k1_FDX_s015
                                 + 0.845989 w_k1_IBM_s015
                                 + 1.08867 w_k1_INTC_s015
                                 + 0.830169 w_k1_KO_s015
                                 + 1.07347 w_k1_MAT_s015
                                 + 0.895764 w_k1_MSFT_s015
                                 + 0.9996 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.11484 w_k1_ADM_s015
                                 + 0.795264 w_k1_AXP_s015
                                 + 0.698911 w_k1_DIS_s015
                                 + 0.773117 w_k1_FDX_s015
                                 + 0.936399 w_k1_IBM_s015
                                 + 0.89406 w_k1_INTC_s015
                                 + 0.845294 w_k1_KO_s015
                                 + 0.761688 w_k1_MAT_s015
                                 + 0.737736 w_k1_MSFT_s015
                                 + 0.876417 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.754148 w_k1_ADM_s015
                                 + 0.69536 w_k1_AXP_s015
                                 + 0.679192 w_k1_DIS_s015
                                 + 0.768168 w_k1_FDX_s015
                                 + 0.674136 w_k1_IBM_s015
                                 + 0.91757 w_k1_INTC_s015
                                 + 0.722191 w_k1_KO_s015
                                 + 0.789407 w_k1_MAT_s015
                                 + 0.769047 w_k1_MSFT_s015
                                 + 0.822318 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.976277 w_k1_ADM_s015
                                 + 0.755684 w_k1_AXP_s015
                                 + 0.741884 w_k1_DIS_s015
                                 + 0.93428 w_k1_FDX_s015
                                 + 0.689889 w_k1_IBM_s015
                                 + 1.11415 w_k1_INTC_s015
                                 + 0.849472 w_k1_KO_s015
                                 + 0.671916 w_k1_MAT_s015
                                 + 0.91495 w_k1_MSFT_s015
                                 + 0.905793 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.16173 w_k1_ADM_s015
                                 + 1.09445 w_k1_AXP_s015
                                 + 1.27775 w_k1_DIS_s015
                                 + 1.07919 w_k1_FDX_s015
                                 + 1.05291 w_k1_IBM_s015
                                 + 1.15095 w_k1_INTC_s015
                                 + 1.14106 w_k1_KO_s015 + 1.16784 w_k1_MAT_s015
                                 + 1.09009 w_k1_MSFT_s015 + 1.1877 w_k1_PG_s015
                                 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.38172 w_k1_ADM_s015
                                 + 1.41301 w_k1_AXP_s015
                                 + 1.60932 w_k1_DIS_s015
                                 + 1.35798 w_k1_FDX_s015
                                 + 1.46104 w_k1_IBM_s015
                                 + 1.17958 w_k1_INTC_s015
                                 + 1.30807 w_k1_KO_s015 + 1.64597 w_k1_MAT_s015
                                 + 1.0483 w_k1_MSFT_s015 + 1.31765 w_k1_PG_s015
                                 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.962006 w_k1_ADM_s015
                                 + 0.714465 w_k1_AXP_s015
                                 + 1.14821 w_k1_DIS_s015
                                 + 1.00848 w_k1_FDX_s015
                                 + 0.828523 w_k1_IBM_s015
                                 + 1.12157 w_k1_INTC_s015
                                 + 0.936112 w_k1_KO_s015
                                 + 0.623329 w_k1_MAT_s015
                                 + 1.06511 w_k1_MSFT_s015
                                 + 1.05024 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.00913 w_k1_ADM_s015
                                 + 0.973877 w_k1_AXP_s015
                                 + 0.991363 w_k1_DIS_s015
                                 + 0.816197 w_k1_FDX_s015
                                 + 1.45453 w_k1_IBM_s015
                                 + 0.932195 w_k1_INTC_s015
                                 + 1.0938 w_k1_KO_s015 + 1.3298 w_k1_MAT_s015
                                 + 0.929781 w_k1_MSFT_s015
                                 + 1.05477 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.927894 w_k1_ADM_s016
                                 + 0.966903 w_k1_AXP_s016
                                 + 1.33651 w_k1_DIS_s016 + 1.0767 w_k1_FDX_s016
                                 + 1.09116 w_k1_IBM_s016
                                 + 1.06104 w_k1_INTC_s016
                                 + 1.12171 w_k1_KO_s016
                                 + 0.967508 w_k1_MAT_s016
                                 + 1.14781 w_k1_MSFT_s016
                                 + 0.979864 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.957954 w_k1_ADM_s016
                                 + 1.21183 w_k1_AXP_s016
                                 + 0.921973 w_k1_DIS_s016
                                 + 0.792383 w_k1_FDX_s016
                                 + 1.45155 w_k1_IBM_s016
                                 + 0.880805 w_k1_INTC_s016
                                 + 1.0003 w_k1_KO_s016 + 1.6016 w_k1_MAT_s016
                                 + 0.954152 w_k1_MSFT_s016
                                 + 0.971851 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.23729 w_k1_ADM_s016
                                 + 1.58665 w_k1_AXP_s016
                                 + 2.07656 w_k1_DIS_s016
                                 + 1.48038 w_k1_FDX_s016
                                 + 1.07277 w_k1_IBM_s016
                                 + 1.44499 w_k1_INTC_s016
                                 + 1.24575 w_k1_KO_s016 + 1.29525 w_k1_MAT_s016
                                 + 1.64044 w_k1_MSFT_s016
                                 + 1.15718 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.10781 w_k1_ADM_s016
                                 + 1.27091 w_k1_AXP_s016
                                 + 1.21572 w_k1_DIS_s016
                                 + 1.21755 w_k1_FDX_s016
                                 + 0.752623 w_k1_IBM_s016
                                 + 1.02574 w_k1_INTC_s016
                                 + 0.924344 w_k1_KO_s016
                                 + 0.977464 w_k1_MAT_s016
                                 + 1.14658 w_k1_MSFT_s016
                                 + 1.09171 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.46631 w_k1_ADM_s016
                                 + 1.7393 w_k1_AXP_s016 + 1.28632 w_k1_DIS_s016
                                 + 1.2539 w_k1_FDX_s016 + 1.31299 w_k1_IBM_s016
                                 + 1.00516 w_k1_INTC_s016
                                 + 1.17519 w_k1_KO_s016 + 2.01889 w_k1_MAT_s016
                                 + 1.11688 w_k1_MSFT_s016
                                 + 1.10905 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.17359 w_k1_ADM_s016
                                 + 0.813217 w_k1_AXP_s016
                                 + 0.828845 w_k1_DIS_s016
                                 + 0.847048 w_k1_FDX_s016
                                 + 0.830296 w_k1_IBM_s016
                                 + 0.738452 w_k1_INTC_s016
                                 + 0.879827 w_k1_KO_s016
                                 + 0.672387 w_k1_MAT_s016
                                 + 0.861842 w_k1_MSFT_s016
                                 + 0.933999 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.32886 w_k1_ADM_s016
                                 + 1.18739 w_k1_AXP_s016
                                 + 0.744438 w_k1_DIS_s016
                                 + 0.982068 w_k1_FDX_s016
                                 + 0.680747 w_k1_IBM_s016
                                 + 0.865619 w_k1_INTC_s016
                                 + 0.693572 w_k1_KO_s016
                                 + 0.821095 w_k1_MAT_s016
                                 + 0.864404 w_k1_MSFT_s016
                                 + 0.822323 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.07401 w_k1_ADM_s016
                                 + 1.2017 w_k1_AXP_s016
                                 + 0.904566 w_k1_DIS_s016
                                 + 1.00895 w_k1_FDX_s016
                                 + 1.03973 w_k1_IBM_s016
                                 + 0.894046 w_k1_INTC_s016
                                 + 0.992208 w_k1_KO_s016
                                 + 1.38367 w_k1_MAT_s016
                                 + 0.95299 w_k1_MSFT_s016
                                 + 0.962464 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.949362 w_k1_ADM_s017
                                 + 1.16041 w_k1_AXP_s017
                                 + 0.830487 w_k1_DIS_s017
                                 + 0.937018 w_k1_FDX_s017
                                 + 0.885852 w_k1_IBM_s017
                                 + 0.828549 w_k1_INTC_s017
                                 + 0.797756 w_k1_KO_s017
                                 + 1.16889 w_k1_MAT_s017
                                 + 0.898777 w_k1_MSFT_s017
                                 + 0.93976 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.06296 w_k1_ADM_s017
                                 + 1.04306 w_k1_AXP_s017
                                 + 1.29514 w_k1_DIS_s017
                                 + 1.04206 w_k1_FDX_s017
                                 + 1.33774 w_k1_IBM_s017
                                 + 1.05519 w_k1_INTC_s017
                                 + 1.17427 w_k1_KO_s017 + 1.05194 w_k1_MAT_s017
                                 + 1.02245 w_k1_MSFT_s017 + 1.0092 w_k1_PG_s017
                                 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.16767 w_k1_ADM_s017
                                 + 1.8102 w_k1_AXP_s017 + 1.42574 w_k1_DIS_s017
                                 + 1.229 w_k1_FDX_s017 + 1.14948 w_k1_IBM_s017
                                 + 1.08751 w_k1_INTC_s017
                                 + 1.13251 w_k1_KO_s017 + 1.28763 w_k1_MAT_s017
                                 + 1.08572 w_k1_MSFT_s017
                                 + 1.05784 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.18441 w_k1_ADM_s017
                                 + 1.23736 w_k1_AXP_s017
                                 + 1.14661 w_k1_DIS_s017
                                 + 1.17957 w_k1_FDX_s017
                                 + 0.786573 w_k1_IBM_s017
                                 + 0.856253 w_k1_INTC_s017
                                 + 0.803806 w_k1_KO_s017
                                 + 0.613874 w_k1_MAT_s017
                                 + 0.987545 w_k1_MSFT_s017
                                 + 1.09291 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.09568 w_k1_ADM_s017
                                 + 0.995439 w_k1_AXP_s017
                                 + 0.88379 w_k1_DIS_s017
                                 + 0.869672 w_k1_FDX_s017
                                 + 0.981447 w_k1_IBM_s017
                                 + 0.845098 w_k1_INTC_s017
                                 + 0.81421 w_k1_KO_s017
                                 + 0.902582 w_k1_MAT_s017
                                 + 0.807208 w_k1_MSFT_s017
                                 + 0.880518 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.884927 w_k1_ADM_s017
                                 + 0.748428 w_k1_AXP_s017
                                 + 0.645681 w_k1_DIS_s017
                                 + 0.740058 w_k1_FDX_s017
                                 + 0.803308 w_k1_IBM_s017
                                 + 0.70348 w_k1_INTC_s017
                                 + 0.62612 w_k1_KO_s017
                                 + 0.777165 w_k1_MAT_s017
                                 + 0.734427 w_k1_MSFT_s017
                                 + 0.823239 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.954729 w_k1_ADM_s017
                                 + 0.829363 w_k1_AXP_s017
                                 + 1.02911 w_k1_DIS_s017
                                 + 1.05833 w_k1_FDX_s017
                                 + 0.826186 w_k1_IBM_s017
                                 + 0.963656 w_k1_INTC_s017
                                 + 0.861457 w_k1_KO_s017
                                 + 0.483378 w_k1_MAT_s017
                                 + 1.03925 w_k1_MSFT_s017
                                 + 0.985931 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.68718 w_k1_ADM_s017
                                 + 1.17008 w_k1_AXP_s017
                                 + 1.57021 w_k1_DIS_s017
                                 + 1.49044 w_k1_FDX_s017
                                 + 0.99751 w_k1_IBM_s017
                                 + 1.12564 w_k1_INTC_s017
                                 + 1.03552 w_k1_KO_s017
                                 + 0.964971 w_k1_MAT_s017
                                 + 1.30982 w_k1_MSFT_s017
                                 + 1.27965 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.890866 w_k1_ADM_s100
                                 + 0.832856 w_k1_AXP_s100
                                 + 0.77287 w_k1_DIS_s100
                                 + 0.865505 w_k1_FDX_s100
                                 + 0.765913 w_k1_IBM_s100
                                 + 0.832669 w_k1_INTC_s100
                                 + 0.907368 w_k1_KO_s100
                                 + 0.759862 w_k1_MAT_s100
                                 + 0.936017 w_k1_MSFT_s100
                                 + 0.845163 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.18922 w_k1_ADM_s100
                                 + 1.00406 w_k1_AXP_s100
                                 + 1.17999 w_k1_DIS_s100
                                 + 1.04715 w_k1_FDX_s100
                                 + 1.42513 w_k1_IBM_s100
                                 + 0.955173 w_k1_INTC_s100
                                 + 1.47024 w_k1_KO_s100 + 1.34211 w_k1_MAT_s100
                                 + 1.18275 w_k1_MSFT_s100
                                 + 1.03341 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.97225 w_k1_ADM_s100
                                 + 0.850626 w_k1_AXP_s100
                                 + 0.823143 w_k1_DIS_s100
                                 + 0.839356 w_k1_FDX_s100
                                 + 1.22205 w_k1_IBM_s100
                                 + 0.800166 w_k1_INTC_s100
                                 + 1.03404 w_k1_KO_s100 + 1.38116 w_k1_MAT_s100
                                 + 0.965807 w_k1_MSFT_s100
                                 + 0.958561 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.42776 w_k1_ADM_s100
                                 + 1.11787 w_k1_AXP_s100
                                 + 1.42011 w_k1_DIS_s100
                                 + 1.14938 w_k1_FDX_s100
                                 + 1.79021 w_k1_IBM_s100
                                 + 0.940085 w_k1_INTC_s100
                                 + 1.73356 w_k1_KO_s100 + 1.8322 w_k1_MAT_s100
                                 + 1.2265 w_k1_MSFT_s100 + 1.21816 w_k1_PG_s100
                                 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.38131 w_k1_ADM_s100
                                 + 1.51854 w_k1_AXP_s100
                                 + 1.34216 w_k1_DIS_s100
                                 + 1.36506 w_k1_FDX_s100
                                 + 1.07705 w_k1_IBM_s100
                                 + 1.15106 w_k1_INTC_s100
                                 + 1.23682 w_k1_KO_s100 + 1.28085 w_k1_MAT_s100
                                 + 1.37636 w_k1_MSFT_s100
                                 + 1.02766 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.01004 w_k1_ADM_s100
                                 + 1.03762 w_k1_AXP_s100
                                 + 0.973702 w_k1_DIS_s100
                                 + 1.01926 w_k1_FDX_s100
                                 + 1.07573 w_k1_IBM_s100
                                 + 0.954326 w_k1_INTC_s100
                                 + 1.0988 w_k1_KO_s100 + 1.03186 w_k1_MAT_s100
                                 + 1.08741 w_k1_MSFT_s100
                                 + 0.956051 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.38089 w_k1_ADM_s100
                                 + 1.08378 w_k1_AXP_s100
                                 + 1.02443 w_k1_DIS_s100
                                 + 1.12671 w_k1_FDX_s100
                                 + 0.733739 w_k1_IBM_s100
                                 + 0.717305 w_k1_INTC_s100
                                 + 0.961304 w_k1_KO_s100
                                 + 0.835452 w_k1_MAT_s100
                                 + 1.05144 w_k1_MSFT_s100
                                 + 0.988387 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.23574 w_k1_ADM_s100
                                 + 0.99095 w_k1_AXP_s100
                                 + 1.19189 w_k1_DIS_s100
                                 + 1.14289 w_k1_FDX_s100
                                 + 0.77479 w_k1_IBM_s100
                                 + 1.14994 w_k1_INTC_s100
                                 + 1.14633 w_k1_KO_s100
                                 + 0.824094 w_k1_MAT_s100
                                 + 1.43788 w_k1_MSFT_s100
                                 + 1.10948 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.39407 w_k1_ADM_s101
                                 + 1.62871 w_k1_AXP_s101
                                 + 1.32559 w_k1_DIS_s101 + 1.2561 w_k1_FDX_s101
                                 + 0.936038 w_k1_IBM_s101
                                 + 1.00506 w_k1_INTC_s101
                                 + 1.15847 w_k1_KO_s101
                                 + 0.827169 w_k1_MAT_s101
                                 + 1.17357 w_k1_MSFT_s101
                                 + 0.978804 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.884398 w_k1_ADM_s101
                                 + 1.18016 w_k1_AXP_s101
                                 + 0.702152 w_k1_DIS_s101
                                 + 0.860209 w_k1_FDX_s101
                                 + 1.01176 w_k1_IBM_s101
                                 + 1.03254 w_k1_INTC_s101
                                 + 0.935576 w_k1_KO_s101
                                 + 0.96009 w_k1_MAT_s101
                                 + 0.90111 w_k1_MSFT_s101
                                 + 0.987999 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.51645 w_k1_ADM_s101
                                 + 1.60194 w_k1_AXP_s101
                                 + 1.62475 w_k1_DIS_s101
                                 + 1.36307 w_k1_FDX_s101
                                 + 1.72878 w_k1_IBM_s101
                                 + 1.63383 w_k1_INTC_s101 + 1.7921 w_k1_KO_s101
                                 + 1.77158 w_k1_MAT_s101
                                 + 1.54189 w_k1_MSFT_s101
                                 + 1.47097 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.16156 w_k1_ADM_s101
                                 + 1.07209 w_k1_AXP_s101
                                 + 0.749849 w_k1_DIS_s101
                                 + 0.988394 w_k1_FDX_s101
                                 + 0.750901 w_k1_IBM_s101
                                 + 1.00152 w_k1_INTC_s101
                                 + 0.777213 w_k1_KO_s101
                                 + 0.628893 w_k1_MAT_s101
                                 + 0.913978 w_k1_MSFT_s101
                                 + 0.91497 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.11677 w_k1_ADM_s101
                                 + 0.947891 w_k1_AXP_s101
                                 + 1.0355 w_k1_DIS_s101
                                 + 0.969161 w_k1_FDX_s101
                                 + 0.845973 w_k1_IBM_s101
                                 + 1.1066 w_k1_INTC_s101
                                 + 0.904128 w_k1_KO_s101
                                 + 0.596191 w_k1_MAT_s101
                                 + 1.09447 w_k1_MSFT_s101
                                 + 0.921641 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.764287 w_k1_ADM_s101
                                 + 0.744136 w_k1_AXP_s101
                                 + 0.606181 w_k1_DIS_s101
                                 + 0.752602 w_k1_FDX_s101
                                 + 0.74595 w_k1_IBM_s101
                                 + 0.845345 w_k1_INTC_s101
                                 + 0.740885 w_k1_KO_s101
                                 + 0.623324 w_k1_MAT_s101
                                 + 0.810958 w_k1_MSFT_s101
                                 + 0.831905 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.41602 w_k1_ADM_s101
                                 + 1.54633 w_k1_AXP_s101
                                 + 1.15834 w_k1_DIS_s101
                                 + 1.15205 w_k1_FDX_s101
                                 + 1.32028 w_k1_IBM_s101
                                 + 1.02889 w_k1_INTC_s101
                                 + 1.33095 w_k1_KO_s101 + 1.52719 w_k1_MAT_s101
                                 + 1.18154 w_k1_MSFT_s101
                                 + 1.16444 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.32039 w_k1_ADM_s101
                                 + 1.17869 w_k1_AXP_s101
                                 + 0.979527 w_k1_DIS_s101
                                 + 0.95164 w_k1_FDX_s101
                                 + 1.38162 w_k1_IBM_s101
                                 + 1.1457 w_k1_INTC_s101 + 1.12983 w_k1_KO_s101
                                 + 1.1398 w_k1_MAT_s101
                                 + 1.07868 w_k1_MSFT_s101
                                 + 1.15907 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.06184 w_k1_ADM_s102
                                 + 1.05168 w_k1_AXP_s102
                                 + 0.936549 w_k1_DIS_s102
                                 + 1.05472 w_k1_FDX_s102
                                 + 0.88745 w_k1_IBM_s102
                                 + 1.05722 w_k1_INTC_s102
                                 + 0.949966 w_k1_KO_s102
                                 + 0.818294 w_k1_MAT_s102
                                 + 0.954652 w_k1_MSFT_s102
                                 + 0.938782 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.94386 w_k1_ADM_s102
                                 + 0.809597 w_k1_AXP_s102
                                 + 0.753655 w_k1_DIS_s102
                                 + 0.91812 w_k1_FDX_s102
                                 + 0.623282 w_k1_IBM_s102
                                 + 0.886539 w_k1_INTC_s102
                                 + 0.76468 w_k1_KO_s102
                                 + 0.547093 w_k1_MAT_s102
                                 + 1.05832 w_k1_MSFT_s102
                                 + 0.863797 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.889414 w_k1_ADM_s102
                                 + 0.798472 w_k1_AXP_s102
                                 + 0.633789 w_k1_DIS_s102
                                 + 0.769292 w_k1_FDX_s102
                                 + 0.898531 w_k1_IBM_s102
                                 + 0.717971 w_k1_INTC_s102
                                 + 0.801893 w_k1_KO_s102
                                 + 0.784972 w_k1_MAT_s102
                                 + 0.722709 w_k1_MSFT_s102
                                 + 0.771355 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.96358 w_k1_ADM_s102
                                 + 1.5842 w_k1_AXP_s102
                                 + 0.994968 w_k1_DIS_s102
                                 + 1.0087 w_k1_FDX_s102 + 1.327 w_k1_IBM_s102
                                 + 1.04152 w_k1_INTC_s102
                                 + 1.09406 w_k1_KO_s102 + 1.27569 w_k1_MAT_s102
                                 + 0.960713 w_k1_MSFT_s102
                                 + 1.09087 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.19291 w_k1_ADM_s102
                                 + 1.49636 w_k1_AXP_s102
                                 + 1.57179 w_k1_DIS_s102
                                 + 1.26128 w_k1_FDX_s102
                                 + 1.46466 w_k1_IBM_s102
                                 + 1.38306 w_k1_INTC_s102
                                 + 1.49152 w_k1_KO_s102 + 1.81286 w_k1_MAT_s102
                                 + 1.31064 w_k1_MSFT_s102
                                 + 1.12705 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.28863 w_k1_ADM_s102
                                 + 1.54237 w_k1_AXP_s102
                                 + 1.40529 w_k1_DIS_s102
                                 + 1.34223 w_k1_FDX_s102
                                 + 0.831067 w_k1_IBM_s102
                                 + 1.52559 w_k1_INTC_s102
                                 + 1.24463 w_k1_KO_s102
                                 + 0.866728 w_k1_MAT_s102
                                 + 1.42493 w_k1_MSFT_s102
                                 + 1.13625 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.901254 w_k1_ADM_s102
                                 + 0.888948 w_k1_AXP_s102
                                 + 0.966725 w_k1_DIS_s102
                                 + 0.869166 w_k1_FDX_s102
                                 + 1.2252 w_k1_IBM_s102
                                 + 0.88972 w_k1_INTC_s102
                                 + 1.02752 w_k1_KO_s102
                                 + 0.905484 w_k1_MAT_s102
                                 + 0.870032 w_k1_MSFT_s102
                                 + 0.990369 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.03284 w_k1_ADM_s102
                                 + 1.18352 w_k1_AXP_s102
                                 + 1.43956 w_k1_DIS_s102
                                 + 1.15724 w_k1_FDX_s102
                                 + 0.954674 w_k1_IBM_s102
                                 + 1.21912 w_k1_INTC_s102
                                 + 0.936597 w_k1_KO_s102
                                 + 1.10914 w_k1_MAT_s102
                                 + 1.08396 w_k1_MSFT_s102
                                 + 0.968629 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.20191 w_k1_ADM_s103
                                 + 1.55735 w_k1_AXP_s103
                                 + 1.45249 w_k1_DIS_s103
                                 + 1.21652 w_k1_FDX_s103
                                 + 1.43238 w_k1_IBM_s103
                                 + 1.25502 w_k1_INTC_s103
                                 + 1.33215 w_k1_KO_s103 + 1.6102 w_k1_MAT_s103
                                 + 1.27441 w_k1_MSFT_s103
                                 + 1.28715 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.51396 w_k1_ADM_s103
                                 + 1.67765 w_k1_AXP_s103
                                 + 1.49465 w_k1_DIS_s103
                                 + 1.39008 w_k1_FDX_s103 + 1.7944 w_k1_IBM_s103
                                 + 1.36326 w_k1_INTC_s103
                                 + 1.48319 w_k1_KO_s103 + 1.67003 w_k1_MAT_s103
                                 + 1.29934 w_k1_MSFT_s103
                                 + 1.30625 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.08494 w_k1_ADM_s103
                                 + 1.41865 w_k1_AXP_s103
                                 + 1.04681 w_k1_DIS_s103
                                 + 1.16481 w_k1_FDX_s103
                                 + 1.11596 w_k1_IBM_s103
                                 + 1.12617 w_k1_INTC_s103
                                 + 1.09718 w_k1_KO_s103 + 1.06242 w_k1_MAT_s103
                                 + 1.16403 w_k1_MSFT_s103
                                 + 1.04034 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.684651 w_k1_ADM_s103
                                 + 0.703135 w_k1_AXP_s103
                                 + 0.680469 w_k1_DIS_s103
                                 + 0.828851 w_k1_FDX_s103
                                 + 0.633659 w_k1_IBM_s103
                                 + 0.894568 w_k1_INTC_s103
                                 + 0.799468 w_k1_KO_s103
                                 + 0.703633 w_k1_MAT_s103
                                 + 0.892932 w_k1_MSFT_s103
                                 + 0.82327 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.10536 w_k1_ADM_s103
                                 + 1.22214 w_k1_AXP_s103
                                 + 1.09654 w_k1_DIS_s103
                                 + 1.01969 w_k1_FDX_s103
                                 + 1.38899 w_k1_IBM_s103
                                 + 1.05907 w_k1_INTC_s103
                                 + 1.19859 w_k1_KO_s103 + 1.07903 w_k1_MAT_s103
                                 + 1.11995 w_k1_MSFT_s103
                                 + 1.02148 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.962523 w_k1_ADM_s103
                                 + 1.13727 w_k1_AXP_s103
                                 + 0.924213 w_k1_DIS_s103
                                 + 1.00185 w_k1_FDX_s103
                                 + 0.745843 w_k1_IBM_s103
                                 + 1.20334 w_k1_INTC_s103
                                 + 0.99418 w_k1_KO_s103
                                 + 0.802037 w_k1_MAT_s103
                                 + 1.07151 w_k1_MSFT_s103
                                 + 0.968041 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.869422 w_k1_ADM_s103
                                 + 0.963712 w_k1_AXP_s103
                                 + 0.703439 w_k1_DIS_s103
                                 + 0.715195 w_k1_FDX_s103
                                 + 1.09561 w_k1_IBM_s103
                                 + 0.915701 w_k1_INTC_s103
                                 + 0.952329 w_k1_KO_s103
                                 + 0.991108 w_k1_MAT_s103
                                 + 0.786181 w_k1_MSFT_s103
                                 + 0.849773 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.16444 w_k1_ADM_s103
                                 + 1.17209 w_k1_AXP_s103 + 1.0034 w_k1_DIS_s103
                                 + 1.06243 w_k1_FDX_s103
                                 + 0.900287 w_k1_IBM_s103
                                 + 1.25507 w_k1_INTC_s103
                                 + 1.16631 w_k1_KO_s103
                                 + 0.889018 w_k1_MAT_s103
                                 + 1.12103 w_k1_MSFT_s103
                                 + 1.04693 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.05005 w_k1_ADM_s104
                                 + 0.900718 w_k1_AXP_s104
                                 + 1.25648 w_k1_DIS_s104
                                 + 1.14227 w_k1_FDX_s104
                                 + 0.977301 w_k1_IBM_s104
                                 + 1.16439 w_k1_INTC_s104
                                 + 1.12098 w_k1_KO_s104
                                 + 0.936584 w_k1_MAT_s104
                                 + 1.06038 w_k1_MSFT_s104
                                 + 1.01703 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.909576 w_k1_ADM_s104
                                 + 1.24183 w_k1_AXP_s104
                                 + 1.39752 w_k1_DIS_s104
                                 + 1.21291 w_k1_FDX_s104
                                 + 1.33016 w_k1_IBM_s104
                                 + 1.54052 w_k1_INTC_s104
                                 + 1.27674 w_k1_KO_s104 + 1.41702 w_k1_MAT_s104
                                 + 1.16535 w_k1_MSFT_s104
                                 + 1.08794 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.841158 w_k1_ADM_s104
                                 + 0.754623 w_k1_AXP_s104
                                 + 0.86151 w_k1_DIS_s104
                                 + 0.923164 w_k1_FDX_s104
                                 + 0.697351 w_k1_IBM_s104
                                 + 0.956474 w_k1_INTC_s104
                                 + 0.802215 w_k1_KO_s104
                                 + 0.588314 w_k1_MAT_s104
                                 + 0.886108 w_k1_MSFT_s104
                                 + 0.88739 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.18251 w_k1_ADM_s104
                                 + 1.11185 w_k1_AXP_s104
                                 + 0.99391 w_k1_DIS_s104
                                 + 1.09701 w_k1_FDX_s104
                                 + 0.789311 w_k1_IBM_s104
                                 + 1.03431 w_k1_INTC_s104
                                 + 0.897176 w_k1_KO_s104
                                 + 0.790635 w_k1_MAT_s104
                                 + 0.950971 w_k1_MSFT_s104
                                 + 1.02294 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.152 w_k1_ADM_s104
                                 + 1.06439 w_k1_AXP_s104
                                 + 1.24494 w_k1_DIS_s104 + 1.3884 w_k1_FDX_s104
                                 + 0.761118 w_k1_IBM_s104
                                 + 1.20878 w_k1_INTC_s104
                                 + 0.911105 w_k1_KO_s104
                                 + 0.636145 w_k1_MAT_s104
                                 + 1.26277 w_k1_MSFT_s104
                                 + 1.09738 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.915899 w_k1_ADM_s104
                                 + 0.996151 w_k1_AXP_s104
                                 + 0.784321 w_k1_DIS_s104
                                 + 0.996614 w_k1_FDX_s104
                                 + 1.04895 w_k1_IBM_s104
                                 + 1.23669 w_k1_INTC_s104
                                 + 0.89618 w_k1_KO_s104 + 0.94438 w_k1_MAT_s104
                                 + 0.94007 w_k1_MSFT_s104
                                 + 0.98952 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.828355 w_k1_ADM_s104
                                 + 0.834648 w_k1_AXP_s104
                                 + 0.861883 w_k1_DIS_s104
                                 + 0.929924 w_k1_FDX_s104
                                 + 1.27106 w_k1_IBM_s104
                                 + 1.12902 w_k1_INTC_s104 + 0.9496 w_k1_KO_s104
                                 + 1.05592 w_k1_MAT_s104
                                 + 0.898909 w_k1_MSFT_s104
                                 + 0.910344 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.870325 w_k1_ADM_s104
                                 + 0.700695 w_k1_AXP_s104
                                 + 0.731307 w_k1_DIS_s104
                                 + 0.740779 w_k1_FDX_s104
                                 + 0.873559 w_k1_IBM_s104
                                 + 0.831758 w_k1_INTC_s104
                                 + 0.769105 w_k1_KO_s104
                                 + 0.683376 w_k1_MAT_s104
                                 + 0.789665 w_k1_MSFT_s104
                                 + 0.836816 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.08447 w_k1_ADM_s105
                                 + 1.04045 w_k1_AXP_s105
                                 + 1.02737 w_k1_DIS_s105
                                 + 1.13283 w_k1_FDX_s105
                                 + 0.777718 w_k1_IBM_s105
                                 + 1.11082 w_k1_INTC_s105
                                 + 0.845612 w_k1_KO_s105
                                 + 0.677743 w_k1_MAT_s105
                                 + 1.10851 w_k1_MSFT_s105
                                 + 1.04123 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.07708 w_k1_ADM_s105
                                 + 1.42013 w_k1_AXP_s105
                                 + 1.46627 w_k1_DIS_s105
                                 + 1.39525 w_k1_FDX_s105
                                 + 1.24283 w_k1_IBM_s105
                                 + 1.28864 w_k1_INTC_s105
                                 + 1.15621 w_k1_KO_s105 + 1.08917 w_k1_MAT_s105
                                 + 1.19123 w_k1_MSFT_s105
                                 + 1.13916 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.779949 w_k1_ADM_s105
                                 + 0.732689 w_k1_AXP_s105
                                 + 0.594278 w_k1_DIS_s105
                                 + 0.6677 w_k1_FDX_s105 + 1.16919 w_k1_IBM_s105
                                 + 0.612193 w_k1_INTC_s105
                                 + 0.768264 w_k1_KO_s105
                                 + 0.884995 w_k1_MAT_s105
                                 + 0.706567 w_k1_MSFT_s105
                                 + 0.825513 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.807071 w_k1_ADM_s105
                                 + 0.790746 w_k1_AXP_s105
                                 + 0.922961 w_k1_DIS_s105
                                 + 0.902166 w_k1_FDX_s105
                                 + 1.11749 w_k1_IBM_s105
                                 + 0.90699 w_k1_INTC_s105
                                 + 0.966238 w_k1_KO_s105
                                 + 1.04146 w_k1_MAT_s105
                                 + 0.961516 w_k1_MSFT_s105
                                 + 0.994008 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.640503 w_k1_ADM_s105
                                 + 0.485683 w_k1_AXP_s105
                                 + 0.699509 w_k1_DIS_s105
                                 + 0.754288 w_k1_FDX_s105
                                 + 0.801775 w_k1_IBM_s105
                                 + 0.858313 w_k1_INTC_s105
                                 + 0.828941 w_k1_KO_s105
                                 + 0.845995 w_k1_MAT_s105
                                 + 0.847294 w_k1_MSFT_s105
                                 + 0.868259 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.965736 w_k1_ADM_s105
                                 + 1.19816 w_k1_AXP_s105
                                 + 0.905549 w_k1_DIS_s105
                                 + 1.02977 w_k1_FDX_s105
                                 + 1.15024 w_k1_IBM_s105
                                 + 0.969184 w_k1_INTC_s105
                                 + 1.05775 w_k1_KO_s105 + 1.3058 w_k1_MAT_s105
                                 + 0.95926 w_k1_MSFT_s105
                                 + 1.09438 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.854826 w_k1_ADM_s105
                                 + 0.989541 w_k1_AXP_s105
                                 + 1.17982 w_k1_DIS_s105
                                 + 0.90841 w_k1_FDX_s105
                                 + 1.46026 w_k1_IBM_s105
                                 + 1.03356 w_k1_INTC_s105
                                 + 1.26282 w_k1_KO_s105 + 1.50543 w_k1_MAT_s105
                                 + 1.0033 w_k1_MSFT_s105 + 1.08387 w_k1_PG_s105
                                 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.988492 w_k1_ADM_s105
                                 + 0.89424 w_k1_AXP_s105 + 0.6766 w_k1_DIS_s105
                                 + 0.876645 w_k1_FDX_s105
                                 + 0.7626 w_k1_IBM_s105
                                 + 0.875898 w_k1_INTC_s105
                                 + 0.735469 w_k1_KO_s105
                                 + 0.689042 w_k1_MAT_s105
                                 + 0.788122 w_k1_MSFT_s105
                                 + 0.900558 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.989635 w_k1_ADM_s106
                                 + 1.11826 w_k1_AXP_s106
                                 + 1.04515 w_k1_DIS_s106
                                 + 0.885051 w_k1_FDX_s106
                                 + 1.71179 w_k1_IBM_s106
                                 + 0.940369 w_k1_INTC_s106
                                 + 1.17538 w_k1_KO_s106 + 1.46846 w_k1_MAT_s106
                                 + 0.891034 w_k1_MSFT_s106
                                 + 1.0508 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.26189 w_k1_ADM_s106
                                 + 1.14397 w_k1_AXP_s106
                                 + 0.847528 w_k1_DIS_s106
                                 + 1.07732 w_k1_FDX_s106
                                 + 0.752409 w_k1_IBM_s106
                                 + 0.97421 w_k1_INTC_s106
                                 + 0.830637 w_k1_KO_s106
                                 + 0.895428 w_k1_MAT_s106
                                 + 0.966878 w_k1_MSFT_s106
                                 + 1.08171 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.924286 w_k1_ADM_s106
                                 + 1.20361 w_k1_AXP_s106
                                 + 1.04543 w_k1_DIS_s106
                                 + 1.05249 w_k1_FDX_s106
                                 + 1.18471 w_k1_IBM_s106
                                 + 1.0596 w_k1_INTC_s106 + 1.19857 w_k1_KO_s106
                                 + 1.45998 w_k1_MAT_s106
                                 + 1.15313 w_k1_MSFT_s106
                                 + 1.17747 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.12262 w_k1_ADM_s106
                                 + 1.36377 w_k1_AXP_s106
                                 + 1.25511 w_k1_DIS_s106 + 1.4061 w_k1_FDX_s106
                                 + 0.85736 w_k1_IBM_s106
                                 + 1.43062 w_k1_INTC_s106
                                 + 1.18159 w_k1_KO_s106 + 1.12193 w_k1_MAT_s106
                                 + 1.34246 w_k1_MSFT_s106
                                 + 1.21868 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.784243 w_k1_ADM_s106
                                 + 0.77425 w_k1_AXP_s106
                                 + 0.727903 w_k1_DIS_s106
                                 + 0.82159 w_k1_FDX_s106
                                 + 0.867679 w_k1_IBM_s106
                                 + 0.751463 w_k1_INTC_s106
                                 + 0.965735 w_k1_KO_s106
                                 + 1.05572 w_k1_MAT_s106
                                 + 0.875934 w_k1_MSFT_s106
                                 + 0.866431 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.781533 w_k1_ADM_s106
                                 + 0.596741 w_k1_AXP_s106
                                 + 0.454851 w_k1_DIS_s106
                                 + 0.769278 w_k1_FDX_s106
                                 + 0.566311 w_k1_IBM_s106
                                 + 0.787242 w_k1_INTC_s106
                                 + 0.575951 w_k1_KO_s106
                                 + 0.441675 w_k1_MAT_s106
                                 + 0.719885 w_k1_MSFT_s106
                                 + 0.812021 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.17924 w_k1_ADM_s106
                                 + 1.49621 w_k1_AXP_s106
                                 + 1.47605 w_k1_DIS_s106
                                 + 1.44569 w_k1_FDX_s106
                                 + 1.32318 w_k1_IBM_s106
                                 + 1.11739 w_k1_INTC_s106
                                 + 1.15261 w_k1_KO_s106 + 1.51362 w_k1_MAT_s106
                                 + 1.19621 w_k1_MSFT_s106
                                 + 1.30357 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.759491 w_k1_ADM_s106
                                 + 0.775476 w_k1_AXP_s106
                                 + 0.589918 w_k1_DIS_s106
                                 + 0.736661 w_k1_FDX_s106
                                 + 1.15487 w_k1_IBM_s106
                                 + 0.774178 w_k1_INTC_s106
                                 + 0.906016 w_k1_KO_s106
                                 + 1.10349 w_k1_MAT_s106
                                 + 0.794158 w_k1_MSFT_s106
                                 + 0.911939 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.957127 w_k1_ADM_s107
                                 + 1.17496 w_k1_AXP_s107
                                 + 1.16544 w_k1_DIS_s107
                                 + 1.12827 w_k1_FDX_s107
                                 + 0.889988 w_k1_IBM_s107
                                 + 1.2731 w_k1_INTC_s107
                                 + 0.916069 w_k1_KO_s107
                                 + 0.81999 w_k1_MAT_s107
                                 + 1.12261 w_k1_MSFT_s107
                                 + 1.00594 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.32104 w_k1_ADM_s107
                                 + 1.39751 w_k1_AXP_s107
                                 + 0.868881 w_k1_DIS_s107
                                 + 1.0352 w_k1_FDX_s107 + 1.20918 w_k1_IBM_s107
                                 + 0.936798 w_k1_INTC_s107
                                 + 0.924036 w_k1_KO_s107
                                 + 1.26677 w_k1_MAT_s107
                                 + 0.828843 w_k1_MSFT_s107
                                 + 0.960718 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.989089 w_k1_ADM_s107
                                 + 0.976512 w_k1_AXP_s107
                                 + 0.766156 w_k1_DIS_s107
                                 + 0.888956 w_k1_FDX_s107
                                 + 0.775209 w_k1_IBM_s107
                                 + 0.752031 w_k1_INTC_s107
                                 + 0.842249 w_k1_KO_s107
                                 + 0.718983 w_k1_MAT_s107
                                 + 0.860794 w_k1_MSFT_s107
                                 + 0.779272 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.06624 w_k1_ADM_s107
                                 + 0.823855 w_k1_AXP_s107
                                 + 0.928747 w_k1_DIS_s107
                                 + 0.981011 w_k1_FDX_s107
                                 + 1.12575 w_k1_IBM_s107
                                 + 0.884516 w_k1_INTC_s107
                                 + 1.1213 w_k1_KO_s107 + 1.02771 w_k1_MAT_s107
                                 + 0.961726 w_k1_MSFT_s107
                                 + 0.997474 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.20067 w_k1_ADM_s107
                                 + 0.979369 w_k1_AXP_s107
                                 + 1.10906 w_k1_DIS_s107
                                 + 1.07066 w_k1_FDX_s107
                                 + 0.733934 w_k1_IBM_s107
                                 + 0.98197 w_k1_INTC_s107
                                 + 0.884925 w_k1_KO_s107
                                 + 0.640971 w_k1_MAT_s107
                                 + 1.12209 w_k1_MSFT_s107
                                 + 0.935957 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.25892 w_k1_ADM_s107
                                 + 1.37244 w_k1_AXP_s107
                                 + 1.79791 w_k1_DIS_s107
                                 + 1.48512 w_k1_FDX_s107
                                 + 1.01614 w_k1_IBM_s107
                                 + 1.2155 w_k1_INTC_s107 + 1.20604 w_k1_KO_s107
                                 + 0.922571 w_k1_MAT_s107
                                 + 1.22482 w_k1_MSFT_s107
                                 + 1.18286 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.21603 w_k1_ADM_s107
                                 + 1.16338 w_k1_AXP_s107
                                 + 1.42246 w_k1_DIS_s107
                                 + 1.13727 w_k1_FDX_s107
                                 + 1.30321 w_k1_IBM_s107
                                 + 1.06191 w_k1_INTC_s107
                                 + 1.48495 w_k1_KO_s107 + 1.30167 w_k1_MAT_s107
                                 + 1.26952 w_k1_MSFT_s107
                                 + 1.22671 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.08629 w_k1_ADM_s107
                                 + 1.01697 w_k1_AXP_s107
                                 + 1.08113 w_k1_DIS_s107
                                 + 0.833562 w_k1_FDX_s107
                                 + 1.15679 w_k1_IBM_s107
                                 + 0.952699 w_k1_INTC_s107
                                 + 1.02767 w_k1_KO_s107
                                 + 0.671682 w_k1_MAT_s107
                                 + 0.947298 w_k1_MSFT_s107
                                 + 0.918962 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.20858 w_k1_ADM_s110
                                 + 1.00984 w_k1_AXP_s110
                                 + 0.973639 w_k1_DIS_s110
                                 + 1.04402 w_k1_FDX_s110
                                 + 1.15559 w_k1_IBM_s110
                                 + 0.801178 w_k1_INTC_s110
                                 + 0.840787 w_k1_KO_s110
                                 + 1.54494 w_k1_MAT_s110
                                 + 0.837017 w_k1_MSFT_s110
                                 + 0.916104 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.08969 w_k1_ADM_s110
                                 + 0.818979 w_k1_AXP_s110
                                 + 1.06027 w_k1_DIS_s110 + 1.016 w_k1_FDX_s110
                                 + 0.937048 w_k1_IBM_s110
                                 + 0.874386 w_k1_INTC_s110
                                 + 0.932464 w_k1_KO_s110 + 1.1168 w_k1_MAT_s110
                                 + 0.953905 w_k1_MSFT_s110
                                 + 1.12781 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.10255 w_k1_ADM_s110
                                 + 1.3634 w_k1_AXP_s110 + 1.0994 w_k1_DIS_s110
                                 + 1.33838 w_k1_FDX_s110
                                 + 0.881244 w_k1_IBM_s110
                                 + 0.882939 w_k1_INTC_s110
                                 + 0.894728 w_k1_KO_s110
                                 + 1.79309 w_k1_MAT_s110
                                 + 0.906865 w_k1_MSFT_s110
                                 + 1.17115 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.817092 w_k1_ADM_s110
                                 + 0.592547 w_k1_AXP_s110
                                 + 0.677595 w_k1_DIS_s110
                                 + 0.790841 w_k1_FDX_s110
                                 + 0.636118 w_k1_IBM_s110
                                 + 0.6605 w_k1_INTC_s110
                                 + 0.569816 w_k1_KO_s110
                                 + 0.520011 w_k1_MAT_s110
                                 + 0.791274 w_k1_MSFT_s110
                                 + 0.792007 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.28495 w_k1_ADM_s110
                                 + 1.34975 w_k1_AXP_s110
                                 + 1.70053 w_k1_DIS_s110
                                 + 1.37846 w_k1_FDX_s110
                                 + 1.02701 w_k1_IBM_s110
                                 + 1.02033 w_k1_INTC_s110
                                 + 1.02608 w_k1_KO_s110 + 1.22609 w_k1_MAT_s110
                                 + 1.13959 w_k1_MSFT_s110
                                 + 1.17044 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.880048 w_k1_ADM_s110
                                 + 0.840782 w_k1_AXP_s110
                                 + 0.921718 w_k1_DIS_s110
                                 + 0.893474 w_k1_FDX_s110
                                 + 1.39179 w_k1_IBM_s110
                                 + 0.88651 w_k1_INTC_s110
                                 + 0.879015 w_k1_KO_s110
                                 + 1.11745 w_k1_MAT_s110
                                 + 0.817267 w_k1_MSFT_s110
                                 + 1.02276 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.801611 w_k1_ADM_s110
                                 + 0.744644 w_k1_AXP_s110
                                 + 0.774422 w_k1_DIS_s110
                                 + 0.887763 w_k1_FDX_s110
                                 + 1.02532 w_k1_IBM_s110
                                 + 0.837145 w_k1_INTC_s110
                                 + 0.727737 w_k1_KO_s110 + 0.7617 w_k1_MAT_s110
                                 + 0.723176 w_k1_MSFT_s110
                                 + 0.833956 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.08415 w_k1_ADM_s110
                                 + 1.3076 w_k1_AXP_s110 + 1.28211 w_k1_DIS_s110
                                 + 1.31033 w_k1_FDX_s110
                                 + 1.54669 w_k1_IBM_s110
                                 + 1.03861 w_k1_INTC_s110
                                 + 1.16925 w_k1_KO_s110 + 1.75107 w_k1_MAT_s110
                                 + 1.31587 w_k1_MSFT_s110
                                 + 1.11589 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.04096 w_k1_ADM_s111
                                 + 1.56071 w_k1_AXP_s111
                                 + 1.28146 w_k1_DIS_s111
                                 + 1.28141 w_k1_FDX_s111
                                 + 1.29155 w_k1_IBM_s111
                                 + 1.00567 w_k1_INTC_s111
                                 + 1.36633 w_k1_KO_s111 + 1.25856 w_k1_MAT_s111
                                 + 1.22955 w_k1_MSFT_s111
                                 + 1.10729 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.75753 w_k1_ADM_s111
                                 + 1.23762 w_k1_AXP_s111
                                 + 1.25674 w_k1_DIS_s111
                                 + 1.23519 w_k1_FDX_s111
                                 + 0.759138 w_k1_IBM_s111
                                 + 0.895132 w_k1_INTC_s111
                                 + 1.00604 w_k1_KO_s111
                                 + 0.637087 w_k1_MAT_s111
                                 + 1.00711 w_k1_MSFT_s111
                                 + 0.92647 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.34571 w_k1_ADM_s111
                                 + 1.22878 w_k1_AXP_s111
                                 + 1.01145 w_k1_DIS_s111
                                 + 1.13409 w_k1_FDX_s111
                                 + 1.15797 w_k1_IBM_s111
                                 + 0.889736 w_k1_INTC_s111
                                 + 1.21728 w_k1_KO_s111 + 1.2101 w_k1_MAT_s111
                                 + 1.0164 w_k1_MSFT_s111 + 1.03219 w_k1_PG_s111
                                 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.32588 w_k1_ADM_s111
                                 + 1.07915 w_k1_AXP_s111
                                 + 0.972138 w_k1_DIS_s111
                                 + 1.09679 w_k1_FDX_s111
                                 + 0.862193 w_k1_IBM_s111
                                 + 0.858701 w_k1_INTC_s111
                                 + 1.02016 w_k1_KO_s111
                                 + 0.917354 w_k1_MAT_s111
                                 + 0.991078 w_k1_MSFT_s111
                                 + 0.961062 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.88317 w_k1_ADM_s111
                                 + 1.56369 w_k1_AXP_s111
                                 + 1.71117 w_k1_DIS_s111
                                 + 1.37261 w_k1_FDX_s111
                                 + 1.66519 w_k1_IBM_s111
                                 + 1.15541 w_k1_INTC_s111 + 1.8629 w_k1_KO_s111
                                 + 1.23603 w_k1_MAT_s111
                                 + 1.30419 w_k1_MSFT_s111 + 1.2445 w_k1_PG_s111
                                 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.767763 w_k1_ADM_s111
                                 + 0.689079 w_k1_AXP_s111
                                 + 0.726107 w_k1_DIS_s111
                                 + 0.751788 w_k1_FDX_s111
                                 + 0.789304 w_k1_IBM_s111
                                 + 0.798643 w_k1_INTC_s111
                                 + 0.816497 w_k1_KO_s111
                                 + 0.749958 w_k1_MAT_s111
                                 + 0.819449 w_k1_MSFT_s111
                                 + 0.859543 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.11943 w_k1_ADM_s111
                                 + 1.02242 w_k1_AXP_s111
                                 + 1.01849 w_k1_DIS_s111
                                 + 0.886936 w_k1_FDX_s111
                                 + 1.03934 w_k1_IBM_s111
                                 + 0.872205 w_k1_INTC_s111
                                 + 1.10147 w_k1_KO_s111
                                 + 0.775257 w_k1_MAT_s111
                                 + 0.930409 w_k1_MSFT_s111
                                 + 1.00065 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.995939 w_k1_ADM_s111
                                 + 0.88012 w_k1_AXP_s111
                                 + 1.00289 w_k1_DIS_s111
                                 + 0.945877 w_k1_FDX_s111
                                 + 1.35115 w_k1_IBM_s111
                                 + 0.871444 w_k1_INTC_s111
                                 + 1.21683 w_k1_KO_s111 + 1.35214 w_k1_MAT_s111
                                 + 1.0101 w_k1_MSFT_s111
                                 + 0.948724 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.995699 w_k1_ADM_s112
                                 + 1.21533 w_k1_AXP_s112
                                 + 1.63011 w_k1_DIS_s112
                                 + 1.30829 w_k1_FDX_s112 + 1.3052 w_k1_IBM_s112
                                 + 0.98993 w_k1_INTC_s112
                                 + 1.20053 w_k1_KO_s112 + 1.29438 w_k1_MAT_s112
                                 + 1.11035 w_k1_MSFT_s112
                                 + 1.10423 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.688974 w_k1_ADM_s112
                                 + 1.28631 w_k1_AXP_s112
                                 + 1.40192 w_k1_DIS_s112
                                 + 0.96992 w_k1_FDX_s112
                                 + 1.74265 w_k1_IBM_s112
                                 + 0.764836 w_k1_INTC_s112
                                 + 1.20468 w_k1_KO_s112 + 1.60944 w_k1_MAT_s112
                                 + 0.814961 w_k1_MSFT_s112
                                 + 1.04866 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.831695 w_k1_ADM_s112
                                 + 0.658579 w_k1_AXP_s112
                                 + 0.536405 w_k1_DIS_s112
                                 + 0.666234 w_k1_FDX_s112
                                 + 1.06068 w_k1_IBM_s112
                                 + 0.532841 w_k1_INTC_s112
                                 + 0.689541 w_k1_KO_s112
                                 + 1.00299 w_k1_MAT_s112
                                 + 0.661352 w_k1_MSFT_s112
                                 + 0.757684 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.03884 w_k1_ADM_s112
                                 + 1.05321 w_k1_AXP_s112 + 1.0436 w_k1_DIS_s112
                                 + 1.07561 w_k1_FDX_s112
                                 + 0.987538 w_k1_IBM_s112
                                 + 0.902566 w_k1_INTC_s112
                                 + 0.887274 w_k1_KO_s112
                                 + 0.943291 w_k1_MAT_s112
                                 + 0.982907 w_k1_MSFT_s112
                                 + 1.02114 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.992183 w_k1_ADM_s112
                                 + 1.03878 w_k1_AXP_s112
                                 + 0.950369 w_k1_DIS_s112
                                 + 1.10741 w_k1_FDX_s112
                                 + 0.808096 w_k1_IBM_s112
                                 + 0.86861 w_k1_INTC_s112
                                 + 0.880867 w_k1_KO_s112
                                 + 0.770299 w_k1_MAT_s112
                                 + 0.851721 w_k1_MSFT_s112
                                 + 1.02651 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.678889 w_k1_ADM_s112
                                 + 0.898734 w_k1_AXP_s112
                                 + 1.05485 w_k1_DIS_s112
                                 + 0.98431 w_k1_FDX_s112
                                 + 1.18855 w_k1_IBM_s112
                                 + 0.752879 w_k1_INTC_s112
                                 + 0.915086 w_k1_KO_s112
                                 + 1.11219 w_k1_MAT_s112
                                 + 0.836522 w_k1_MSFT_s112
                                 + 0.918515 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.622537 w_k1_ADM_s112
                                 + 0.741805 w_k1_AXP_s112
                                 + 0.699799 w_k1_DIS_s112
                                 + 0.804837 w_k1_FDX_s112
                                 + 0.833802 w_k1_IBM_s112
                                 + 0.802981 w_k1_INTC_s112
                                 + 0.723312 w_k1_KO_s112
                                 + 0.684051 w_k1_MAT_s112
                                 + 0.828687 w_k1_MSFT_s112
                                 + 0.802124 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.21442 w_k1_ADM_s112
                                 + 1.2056 w_k1_AXP_s112 + 1.31632 w_k1_DIS_s112
                                 + 1.34829 w_k1_FDX_s112
                                 + 1.11298 w_k1_IBM_s112
                                 + 0.86262 w_k1_INTC_s112
                                 + 0.991588 w_k1_KO_s112
                                 + 0.815587 w_k1_MAT_s112
                                 + 0.9851 w_k1_MSFT_s112 + 1.02031 w_k1_PG_s112
                                 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.84622 w_k1_ADM_s113
                                 + 0.821625 w_k1_AXP_s113
                                 + 0.7893 w_k1_DIS_s113
                                 + 0.872711 w_k1_FDX_s113
                                 + 0.753435 w_k1_IBM_s113
                                 + 0.818958 w_k1_INTC_s113
                                 + 0.79424 w_k1_KO_s113
                                 + 0.796452 w_k1_MAT_s113
                                 + 0.792047 w_k1_MSFT_s113
                                 + 0.953979 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.786339 w_k1_ADM_s113
                                 + 1.07425 w_k1_AXP_s113
                                 + 0.631199 w_k1_DIS_s113
                                 + 0.853226 w_k1_FDX_s113
                                 + 0.736454 w_k1_IBM_s113
                                 + 0.963859 w_k1_INTC_s113
                                 + 0.738099 w_k1_KO_s113
                                 + 0.665548 w_k1_MAT_s113
                                 + 0.753885 w_k1_MSFT_s113
                                 + 0.858411 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.79948 w_k1_ADM_s113
                                 + 1.19159 w_k1_AXP_s113
                                 + 1.17944 w_k1_DIS_s113
                                 + 0.952932 w_k1_FDX_s113
                                 + 1.57927 w_k1_IBM_s113
                                 + 1.1331 w_k1_INTC_s113 + 1.4008 w_k1_KO_s113
                                 + 1.43837 w_k1_MAT_s113
                                 + 0.936399 w_k1_MSFT_s113
                                 + 1.07725 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.987769 w_k1_ADM_s113
                                 + 1.49776 w_k1_AXP_s113 + 1.1739 w_k1_DIS_s113
                                 + 1.16639 w_k1_FDX_s113
                                 + 1.13746 w_k1_IBM_s113
                                 + 1.1991 w_k1_INTC_s113 + 1.02137 w_k1_KO_s113
                                 + 1.1253 w_k1_MAT_s113
                                 + 1.01166 w_k1_MSFT_s113
                                 + 1.19261 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.34834 w_k1_ADM_s113
                                 + 1.65787 w_k1_AXP_s113
                                 + 1.22486 w_k1_DIS_s113
                                 + 1.22322 w_k1_FDX_s113
                                 + 0.956019 w_k1_IBM_s113
                                 + 1.24396 w_k1_INTC_s113
                                 + 1.00781 w_k1_KO_s113
                                 + 0.950743 w_k1_MAT_s113
                                 + 1.00445 w_k1_MSFT_s113
                                 + 1.12529 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.05723 w_k1_ADM_s113
                                 + 1.40794 w_k1_AXP_s113
                                 + 1.65364 w_k1_DIS_s113
                                 + 1.26076 w_k1_FDX_s113
                                 + 1.24435 w_k1_IBM_s113
                                 + 1.3195 w_k1_INTC_s113 + 1.56298 w_k1_KO_s113
                                 + 1.51163 w_k1_MAT_s113
                                 + 1.38539 w_k1_MSFT_s113
                                 + 1.12702 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.928067 w_k1_ADM_s113
                                 + 1.11175 w_k1_AXP_s113
                                 + 0.759162 w_k1_DIS_s113
                                 + 0.785228 w_k1_FDX_s113
                                 + 1.12709 w_k1_IBM_s113
                                 + 0.951067 w_k1_INTC_s113
                                 + 0.843222 w_k1_KO_s113
                                 + 0.977463 w_k1_MAT_s113
                                 + 0.791468 w_k1_MSFT_s113
                                 + 0.921681 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.965363 w_k1_ADM_s113
                                 + 0.990954 w_k1_AXP_s113
                                 + 1.27212 w_k1_DIS_s113
                                 + 1.05825 w_k1_FDX_s113
                                 + 0.780332 w_k1_IBM_s113
                                 + 0.960189 w_k1_INTC_s113
                                 + 0.963927 w_k1_KO_s113
                                 + 0.669903 w_k1_MAT_s113
                                 + 1.09272 w_k1_MSFT_s113
                                 + 0.990412 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.932255 w_k1_ADM_s114
                                 + 1.03953 w_k1_AXP_s114 + 0.7916 w_k1_DIS_s114
                                 + 0.983886 w_k1_FDX_s114
                                 + 1.0093 w_k1_IBM_s114
                                 + 1.00755 w_k1_INTC_s114
                                 + 0.82433 w_k1_KO_s114
                                 + 0.760325 w_k1_MAT_s114
                                 + 0.910037 w_k1_MSFT_s114
                                 + 0.928388 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.93075 w_k1_ADM_s114
                                 + 0.838353 w_k1_AXP_s114
                                 + 0.814876 w_k1_DIS_s114
                                 + 0.825621 w_k1_FDX_s114
                                 + 1.03268 w_k1_IBM_s114
                                 + 0.752093 w_k1_INTC_s114
                                 + 0.925006 w_k1_KO_s114
                                 + 0.684351 w_k1_MAT_s114
                                 + 0.791838 w_k1_MSFT_s114
                                 + 0.8935 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.17773 w_k1_ADM_s114
                                 + 1.12872 w_k1_AXP_s114
                                 + 1.07761 w_k1_DIS_s114
                                 + 1.19371 w_k1_FDX_s114
                                 + 0.840417 w_k1_IBM_s114
                                 + 0.976568 w_k1_INTC_s114
                                 + 0.909602 w_k1_KO_s114
                                 + 0.657124 w_k1_MAT_s114
                                 + 1.06226 w_k1_MSFT_s114
                                 + 0.891354 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.736252 w_k1_ADM_s114
                                 + 0.820858 w_k1_AXP_s114
                                 + 1.0088 w_k1_DIS_s114 + 1.01528 w_k1_FDX_s114
                                 + 0.740412 w_k1_IBM_s114
                                 + 0.895092 w_k1_INTC_s114
                                 + 0.781061 w_k1_KO_s114
                                 + 0.48173 w_k1_MAT_s114
                                 + 0.933568 w_k1_MSFT_s114
                                 + 0.938461 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.24831 w_k1_ADM_s114
                                 + 1.72586 w_k1_AXP_s114
                                 + 1.40505 w_k1_DIS_s114
                                 + 1.51956 w_k1_FDX_s114
                                 + 1.43713 w_k1_IBM_s114
                                 + 1.24638 w_k1_INTC_s114
                                 + 1.09565 w_k1_KO_s114 + 1.34957 w_k1_MAT_s114
                                 + 1.14263 w_k1_MSFT_s114
                                 + 1.32975 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.620579 w_k1_ADM_s114
                                 + 0.833242 w_k1_AXP_s114
                                 + 0.843986 w_k1_DIS_s114
                                 + 0.843474 w_k1_FDX_s114
                                 + 1.48358 w_k1_IBM_s114
                                 + 0.956499 w_k1_INTC_s114
                                 + 1.0716 w_k1_KO_s114 + 1.60013 w_k1_MAT_s114
                                 + 0.966578 w_k1_MSFT_s114
                                 + 1.02002 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.684198 w_k1_ADM_s114
                                 + 0.829071 w_k1_AXP_s114
                                 + 1.03967 w_k1_DIS_s114
                                 + 1.06539 w_k1_FDX_s114
                                 + 1.14205 w_k1_IBM_s114
                                 + 1.05684 w_k1_INTC_s114
                                 + 0.880552 w_k1_KO_s114
                                 + 0.841157 w_k1_MAT_s114
                                 + 1.05063 w_k1_MSFT_s114
                                 + 0.949767 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.689572 w_k1_ADM_s114
                                 + 0.698303 w_k1_AXP_s114
                                 + 0.605258 w_k1_DIS_s114
                                 + 0.94478 w_k1_FDX_s114
                                 + 0.623612 w_k1_IBM_s114
                                 + 0.932653 w_k1_INTC_s114
                                 + 0.545207 w_k1_KO_s114
                                 + 0.366639 w_k1_MAT_s114
                                 + 0.784721 w_k1_MSFT_s114
                                 + 0.797162 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.19686 w_k1_ADM_s115
                                 + 1.55347 w_k1_AXP_s115
                                 + 1.54618 w_k1_DIS_s115
                                 + 1.36046 w_k1_FDX_s115
                                 + 1.62341 w_k1_IBM_s115
                                 + 1.46611 w_k1_INTC_s115
                                 + 1.41445 w_k1_KO_s115 + 1.89318 w_k1_MAT_s115
                                 + 1.53494 w_k1_MSFT_s115 + 1.1072 w_k1_PG_s115
                                 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.30817 w_k1_ADM_s115
                                 + 1.45022 w_k1_AXP_s115
                                 + 1.10774 w_k1_DIS_s115
                                 + 1.21091 w_k1_FDX_s115
                                 + 1.01103 w_k1_IBM_s115
                                 + 1.06826 w_k1_INTC_s115
                                 + 1.02533 w_k1_KO_s115 + 1.56607 w_k1_MAT_s115
                                 + 0.981969 w_k1_MSFT_s115
                                 + 1.05805 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.788792 w_k1_ADM_s115
                                 + 0.879032 w_k1_AXP_s115
                                 + 0.786114 w_k1_DIS_s115
                                 + 0.934635 w_k1_FDX_s115
                                 + 0.693624 w_k1_IBM_s115
                                 + 0.961287 w_k1_INTC_s115
                                 + 0.631361 w_k1_KO_s115
                                 + 0.796484 w_k1_MAT_s115
                                 + 0.817182 w_k1_MSFT_s115
                                 + 0.809098 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.733774 w_k1_ADM_s115
                                 + 0.846874 w_k1_AXP_s115
                                 + 0.797027 w_k1_DIS_s115
                                 + 0.751278 w_k1_FDX_s115
                                 + 1.00517 w_k1_IBM_s115
                                 + 0.753836 w_k1_INTC_s115
                                 + 0.900474 w_k1_KO_s115
                                 + 1.40133 w_k1_MAT_s115
                                 + 0.766708 w_k1_MSFT_s115
                                 + 0.837514 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.21225 w_k1_ADM_s115
                                 + 1.32994 w_k1_AXP_s115
                                 + 1.09896 w_k1_DIS_s115
                                 + 1.01674 w_k1_FDX_s115 + 1.4172 w_k1_IBM_s115
                                 + 1.17536 w_k1_INTC_s115
                                 + 1.10411 w_k1_KO_s115 + 1.37928 w_k1_MAT_s115
                                 + 1.0741 w_k1_MSFT_s115 + 1.00346 w_k1_PG_s115
                                 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.07811 w_k1_ADM_s115
                                 + 1.29791 w_k1_AXP_s115 + 1.1084 w_k1_DIS_s115
                                 + 1.32212 w_k1_FDX_s115
                                 + 0.84255 w_k1_IBM_s115
                                 + 1.29895 w_k1_INTC_s115
                                 + 0.923016 w_k1_KO_s115 + 1.0701 w_k1_MAT_s115
                                 + 1.13169 w_k1_MSFT_s115
                                 + 1.11624 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.834936 w_k1_ADM_s115
                                 + 1.03907 w_k1_AXP_s115
                                 + 1.00834 w_k1_DIS_s115
                                 + 1.00037 w_k1_FDX_s115
                                 + 1.01213 w_k1_IBM_s115
                                 + 0.899435 w_k1_INTC_s115
                                 + 0.861595 w_k1_KO_s115
                                 + 1.23397 w_k1_MAT_s115
                                 + 0.911033 w_k1_MSFT_s115
                                 + 0.90742 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.733677 w_k1_ADM_s115
                                 + 1.07103 w_k1_AXP_s115
                                 + 1.18527 w_k1_DIS_s115
                                 + 0.974394 w_k1_FDX_s115
                                 + 1.07892 w_k1_IBM_s115
                                 + 1.19622 w_k1_INTC_s115
                                 + 1.09371 w_k1_KO_s115 + 1.44722 w_k1_MAT_s115
                                 + 1.10255 w_k1_MSFT_s115
                                 + 1.04138 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.01023 w_k1_ADM_s116
                                 + 1.14278 w_k1_AXP_s116 + 1.2566 w_k1_DIS_s116
                                 + 1.08642 w_k1_FDX_s116
                                 + 1.32034 w_k1_IBM_s116
                                 + 1.09372 w_k1_INTC_s116
                                 + 1.14881 w_k1_KO_s116 + 1.29756 w_k1_MAT_s116
                                 + 1.12187 w_k1_MSFT_s116 + 1.2606 w_k1_PG_s116
                                 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.18222 w_k1_ADM_s116
                                 + 0.789175 w_k1_AXP_s116
                                 + 0.842026 w_k1_DIS_s116
                                 + 1.00692 w_k1_FDX_s116
                                 + 0.662042 w_k1_IBM_s116
                                 + 0.926216 w_k1_INTC_s116
                                 + 0.814354 w_k1_KO_s116
                                 + 0.848335 w_k1_MAT_s116
                                 + 0.946705 w_k1_MSFT_s116
                                 + 1.02774 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.06814 w_k1_ADM_s116
                                 + 0.862765 w_k1_AXP_s116
                                 + 0.865488 w_k1_DIS_s116
                                 + 0.817843 w_k1_FDX_s116
                                 + 1.43808 w_k1_IBM_s116
                                 + 0.880117 w_k1_INTC_s116
                                 + 1.05771 w_k1_KO_s116 + 1.26155 w_k1_MAT_s116
                                 + 0.804567 w_k1_MSFT_s116
                                 + 1.03025 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.15884 w_k1_ADM_s116
                                 + 0.929569 w_k1_AXP_s116
                                 + 0.859748 w_k1_DIS_s116
                                 + 0.940446 w_k1_FDX_s116
                                 + 1.29007 w_k1_IBM_s116
                                 + 0.929428 w_k1_INTC_s116
                                 + 1.00503 w_k1_KO_s116 + 1.60926 w_k1_MAT_s116
                                 + 0.903784 w_k1_MSFT_s116
                                 + 1.1575 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.13706 w_k1_ADM_s116
                                 + 0.817275 w_k1_AXP_s116
                                 + 1.19563 w_k1_DIS_s116
                                 + 1.04362 w_k1_FDX_s116
                                 + 1.00854 w_k1_IBM_s116
                                 + 0.970773 w_k1_INTC_s116
                                 + 1.1197 w_k1_KO_s116 + 0.670217 w_k1_MAT_s116
                                 + 1.06919 w_k1_MSFT_s116
                                 + 1.15999 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.709101 w_k1_ADM_s116
                                 + 0.79659 w_k1_AXP_s116
                                 + 0.680191 w_k1_DIS_s116
                                 + 0.820263 w_k1_FDX_s116
                                 + 0.920916 w_k1_IBM_s116
                                 + 0.851546 w_k1_INTC_s116
                                 + 0.834265 w_k1_KO_s116
                                 + 0.935176 w_k1_MAT_s116
                                 + 0.800084 w_k1_MSFT_s116
                                 + 1.06473 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.07506 w_k1_ADM_s116
                                 + 1.22696 w_k1_AXP_s116
                                 + 1.46692 w_k1_DIS_s116
                                 + 1.21517 w_k1_FDX_s116
                                 + 1.23214 w_k1_IBM_s116
                                 + 1.12259 w_k1_INTC_s116 + 1.4499 w_k1_KO_s116
                                 + 1.4226 w_k1_MAT_s116
                                 + 1.15369 w_k1_MSFT_s116
                                 + 1.31261 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.46303 w_k1_ADM_s116
                                 + 1.61425 w_k1_AXP_s116
                                 + 1.43344 w_k1_DIS_s116
                                 + 1.35918 w_k1_FDX_s116
                                 + 0.935114 w_k1_IBM_s116
                                 + 1.27743 w_k1_INTC_s116
                                 + 1.23684 w_k1_KO_s116 + 1.1055 w_k1_MAT_s116
                                 + 1.32116 w_k1_MSFT_s116 + 1.3156 w_k1_PG_s116
                                 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.826499 w_k1_ADM_s117
                                 + 0.798554 w_k1_AXP_s117
                                 + 0.995912 w_k1_DIS_s117
                                 + 0.917416 w_k1_FDX_s117
                                 + 0.716388 w_k1_IBM_s117
                                 + 0.88935 w_k1_INTC_s117
                                 + 1.05188 w_k1_KO_s117
                                 + 0.541399 w_k1_MAT_s117
                                 + 1.01288 w_k1_MSFT_s117
                                 + 0.909466 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.903601 w_k1_ADM_s117
                                 + 0.864747 w_k1_AXP_s117
                                 + 0.83832 w_k1_DIS_s117
                                 + 0.799798 w_k1_FDX_s117
                                 + 1.10076 w_k1_IBM_s117
                                 + 0.929435 w_k1_INTC_s117
                                 + 1.17163 w_k1_KO_s117 + 1.36569 w_k1_MAT_s117
                                 + 0.982411 w_k1_MSFT_s117
                                 + 0.996895 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.95327 w_k1_ADM_s117
                                 + 1.19474 w_k1_AXP_s117
                                 + 1.39333 w_k1_DIS_s117
                                 + 1.11727 w_k1_FDX_s117
                                 + 1.51326 w_k1_IBM_s117
                                 + 0.937759 w_k1_INTC_s117
                                 + 1.35235 w_k1_KO_s117 + 1.97505 w_k1_MAT_s117
                                 + 1.06107 w_k1_MSFT_s117
                                 + 1.08293 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.888487 w_k1_ADM_s117
                                 + 0.964773 w_k1_AXP_s117
                                 + 1.2479 w_k1_DIS_s117 + 1.09402 w_k1_FDX_s117
                                 + 0.904119 w_k1_IBM_s117
                                 + 1.01936 w_k1_INTC_s117 + 1.0794 w_k1_KO_s117
                                 + 1.43751 w_k1_MAT_s117
                                 + 1.17051 w_k1_MSFT_s117
                                 + 1.07423 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.721309 w_k1_ADM_s117
                                 + 0.905899 w_k1_AXP_s117
                                 + 0.631722 w_k1_DIS_s117
                                 + 0.721922 w_k1_FDX_s117
                                 + 0.740185 w_k1_IBM_s117
                                 + 0.835844 w_k1_INTC_s117
                                 + 0.764568 w_k1_KO_s117
                                 + 0.877982 w_k1_MAT_s117
                                 + 0.829925 w_k1_MSFT_s117
                                 + 0.873997 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.81653 w_k1_ADM_s117
                                 + 1.14531 w_k1_AXP_s117
                                 + 1.09827 w_k1_DIS_s117
                                 + 0.881625 w_k1_FDX_s117
                                 + 1.0531 w_k1_IBM_s117
                                 + 0.908845 w_k1_INTC_s117
                                 + 1.16041 w_k1_KO_s117 + 1.50499 w_k1_MAT_s117
                                 + 0.921581 w_k1_MSFT_s117
                                 + 1.15725 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.25095 w_k1_ADM_s117
                                 + 1.68054 w_k1_AXP_s117
                                 + 1.71063 w_k1_DIS_s117
                                 + 1.35541 w_k1_FDX_s117
                                 + 1.15203 w_k1_IBM_s117
                                 + 1.19082 w_k1_INTC_s117
                                 + 1.31989 w_k1_KO_s117 + 1.60233 w_k1_MAT_s117
                                 + 1.46146 w_k1_MSFT_s117
                                 + 1.02562 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.773677 w_k1_ADM_s117
                                 + 1.12554 w_k1_AXP_s117
                                 + 0.972143 w_k1_DIS_s117
                                 + 0.965436 w_k1_FDX_s117
                                 + 0.780354 w_k1_IBM_s117
                                 + 1.05765 w_k1_INTC_s117
                                 + 1.00538 w_k1_KO_s117
                                 + 0.948317 w_k1_MAT_s117
                                 + 1.07105 w_k1_MSFT_s117
                                 + 0.853944 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0388158360307975 w_k1_ADM_s000
                                 + 0.0476763358158751 w_k1_ADM_s001
                                 + 0.0446522995538707 w_k1_ADM_s002
                                 + 0.0423401789316641 w_k1_ADM_s003
                                 + 0.0755355831396156 w_k1_ADM_s004
                                 + 0.0395437274868975 w_k1_ADM_s005
                                 + 0.0353583444221501 w_k1_ADM_s006
                                 + 0.0341671244584511 w_k1_ADM_s007
                                 + 0.0219673024951073 w_k1_ADM_s010
                                 + 0.056797175266661 w_k1_ADM_s011
                                 + 0.0525662274810505 w_k1_ADM_s012
                                 + 0.058353633869488 w_k1_ADM_s013
                                 + 0.026104196016243 w_k1_ADM_s014
                                 + 0.0300344239020754 w_k1_ADM_s015
                                 + 0.0178522559521225 w_k1_ADM_s016
                                 + 0.0136101937152819 w_k1_ADM_s017
                                 + 0.0430974461747485 w_k1_ADM_s100
                                 + 0.0213713936587296 w_k1_ADM_s101
                                 + 0.0327898516314144 w_k1_ADM_s102
                                 + 0.0421884196485664 w_k1_ADM_s103
                                 + 0.0542764094344159 w_k1_ADM_s104
                                 + 0.0656802222902514 w_k1_ADM_s105
                                 + 0.0292453458225163 w_k1_ADM_s106
                                 + 0.0195824280267831 w_k1_ADM_s107
                                 + 0.00456976452754589 w_k1_ADM_s110
                                 + 0.00668778549999657 w_k1_ADM_s111
                                 + 0.00654778526097444 w_k1_ADM_s112
                                 + 0.0129161820214926 w_k1_ADM_s113
                                 + 0.00463835662821 w_k1_ADM_s114
                                 + 0.00664716266570914 w_k1_ADM_s115
                                 + 0.0155005469277022 w_k1_ADM_s116
                                 + 0.0039368890721599 w_k1_ADM_s117
                                 + 0.0342819865406353 w_k1_AXP_s000
                                 + 0.0432698740303517 w_k1_AXP_s001
                                 + 0.0413534360951167 w_k1_AXP_s002
                                 + 0.0437532472916403 w_k1_AXP_s003
                                 + 0.0901312551612326 w_k1_AXP_s004
                                 + 0.040716591072333 w_k1_AXP_s005
                                 + 0.0430707412271278 w_k1_AXP_s006
                                 + 0.0394193884897787 w_k1_AXP_s007
                                 + 0.0224365601884871 w_k1_AXP_s010
                                 + 0.0629881739775088 w_k1_AXP_s011
                                 + 0.0567102057957711 w_k1_AXP_s012
                                 + 0.0495738002073145 w_k1_AXP_s013
                                 + 0.0256430931820751 w_k1_AXP_s014
                                 + 0.0268491644206879 w_k1_AXP_s015
                                 + 0.0186433632540884 w_k1_AXP_s016
                                 + 0.0136559903261834 w_k1_AXP_s017
                                 + 0.0398306966530529 w_k1_AXP_s100
                                 + 0.0225805441785286 w_k1_AXP_s101
                                 + 0.0368122140116688 w_k1_AXP_s102
                                 + 0.0478904435112381 w_k1_AXP_s103
                                 + 0.0557393971642702 w_k1_AXP_s104
                                 + 0.0718392385569204 w_k1_AXP_s105
                                 + 0.0321279899854761 w_k1_AXP_s106
                                 + 0.0187046610189724 w_k1_AXP_s107
                                 + 0.00423327041121066 w_k1_AXP_s110
                                 + 0.00616355776002657 w_k1_AXP_s111
                                 + 0.00745031254705197 w_k1_AXP_s112
                                 + 0.0159675988691844 w_k1_AXP_s113
                                 + 0.00533784323359658 w_k1_AXP_s114
                                 + 0.00792600237891094 w_k1_AXP_s115
                                 + 0.0142539872822488 w_k1_AXP_s116
                                 + 0.00491159302142339 w_k1_AXP_s117
                                 + 0.0353113956123305 w_k1_DIS_s000
                                 + 0.0439907682121876 w_k1_DIS_s001
                                 + 0.0470180784944623 w_k1_DIS_s002
                                 + 0.0420417229117551 w_k1_DIS_s003
                                 + 0.0810526028428178 w_k1_DIS_s004
                                 + 0.042384704754334 w_k1_DIS_s005
                                 + 0.0389346111594758 w_k1_DIS_s006
                                 + 0.0376082783957047 w_k1_DIS_s007
                                 + 0.0219580977406852 w_k1_DIS_s010
                                 + 0.0641432812557778 w_k1_DIS_s011
                                 + 0.058447652313917 w_k1_DIS_s012
                                 + 0.0524249340743443 w_k1_DIS_s013
                                 + 0.02694308173752 w_k1_DIS_s014
                                 + 0.029129210548669 w_k1_DIS_s015
                                 + 0.0171940635625366 w_k1_DIS_s016
                                 + 0.012923053948249 w_k1_DIS_s017
                                 + 0.0389198347261889 w_k1_DIS_s100
                                 + 0.0177540105204185 w_k1_DIS_s101
                                 + 0.0333641794262079 w_k1_DIS_s102
                                 + 0.0401198243042288 w_k1_DIS_s103
                                 + 0.059729314897292 w_k1_DIS_s104
                                 + 0.0696406154257177 w_k1_DIS_s105
                                 + 0.0278538816063115 w_k1_DIS_s106
                                 + 0.0191240410058767 w_k1_DIS_s107
                                 + 0.00446039874418116 w_k1_DIS_s110
                                 + 0.00591107830836933 w_k1_DIS_s111
                                 + 0.00780491354483673 w_k1_DIS_s112
                                 + 0.0138059281136612 w_k1_DIS_s113
                                 + 0.00489535221906951 w_k1_DIS_s114
                                 + 0.00724572895489778 w_k1_DIS_s115
                                 + 0.0148555217749889 w_k1_DIS_s116
                                 + 0.00505999029317649 w_k1_DIS_s117
                                 + 0.034588588101179 w_k1_FDX_s000
                                 + 0.0427968450390565 w_k1_FDX_s001
                                 + 0.0462803454438223 w_k1_FDX_s002
                                 + 0.0432988229639144 w_k1_FDX_s003
                                 + 0.0806175154008169 w_k1_FDX_s004
                                 + 0.0406422920341059 w_k1_FDX_s005
                                 + 0.039140733051212 w_k1_FDX_s006
                                 + 0.0381523307226673 w_k1_FDX_s007
                                 + 0.022748464719975 w_k1_FDX_s010
                                 + 0.0627855169108481 w_k1_FDX_s011
                                 + 0.0583829079543857 w_k1_FDX_s012
                                 + 0.052755583915895 w_k1_FDX_s013
                                 + 0.0266277488375323 w_k1_FDX_s014
                                 + 0.0270418180915294 w_k1_FDX_s015
                                 + 0.0161772478873653 w_k1_FDX_s016
                                 + 0.0126523478845637 w_k1_FDX_s017
                                 + 0.0396903435958123 w_k1_FDX_s100
                                 + 0.0187021147654578 w_k1_FDX_s101
                                 + 0.0327486238258761 w_k1_FDX_s102
                                 + 0.041731258187388 w_k1_FDX_s103
                                 + 0.0595913624002576 w_k1_FDX_s104
                                 + 0.0714112117279485 w_k1_FDX_s105
                                 + 0.0305617896905261 w_k1_FDX_s106
                                 + 0.0178687616838265 w_k1_FDX_s107
                                 + 0.00467368028725452 w_k1_FDX_s110
                                 + 0.0058088204171902 w_k1_FDX_s111
                                 + 0.00780834811450351 w_k1_FDX_s112
                                 + 0.0132442520333368 w_k1_FDX_s113
                                 + 0.00553019214099911 w_k1_FDX_s114
                                 + 0.00716051944294522 w_k1_FDX_s115
                                 + 0.0149087783657057 w_k1_FDX_s116
                                 + 0.00439159545713683 w_k1_FDX_s117
                                 + 0.0352231882500743 w_k1_IBM_s000
                                 + 0.0435884499886748 w_k1_IBM_s001
                                 + 0.0464294417847367 w_k1_IBM_s002
                                 + 0.0422838731217635 w_k1_IBM_s003
                                 + 0.0807246909272915 w_k1_IBM_s004
                                 + 0.0444409470813024 w_k1_IBM_s005
                                 + 0.0394208249463858 w_k1_IBM_s006
                                 + 0.0389698716611953 w_k1_IBM_s007
                                 + 0.022079964931815 w_k1_IBM_s010
                                 + 0.0583150809353207 w_k1_IBM_s011
                                 + 0.0617477329022541 w_k1_IBM_s012
                                 + 0.053978605432963 w_k1_IBM_s013
                                 + 0.026777649560946 w_k1_IBM_s014
                                 + 0.0282591519671319 w_k1_IBM_s015
                                 + 0.0162404703259806 w_k1_IBM_s016
                                 + 0.0118462755342095 w_k1_IBM_s017
                                 + 0.0382924173168356 w_k1_IBM_s100
                                 + 0.0186583507772456 w_k1_IBM_s101
                                 + 0.0337952618731337 w_k1_IBM_s102
                                 + 0.0414628996624519 w_k1_IBM_s103
                                 + 0.0581009867242148 w_k1_IBM_s104
                                 + 0.073773699994339 w_k1_IBM_s105
                                 + 0.0314595793099822 w_k1_IBM_s106
                                 + 0.0179725907416642 w_k1_IBM_s107
                                 + 0.00464424715909965 w_k1_IBM_s110
                                 + 0.00589311363345953 w_k1_IBM_s111
                                 + 0.00799547462929577 w_k1_IBM_s112
                                 + 0.0139319458013969 w_k1_IBM_s113
                                 + 0.00540342123702142 w_k1_IBM_s114
                                 + 0.0074403005718174 w_k1_IBM_s115
                                 + 0.0150410742311642 w_k1_IBM_s116
                                 + 0.00462309511197849 w_k1_IBM_s117
                                 + 0.0410322865831386 w_k1_INTC_s000
                                 + 0.0417195381376349 w_k1_INTC_s001
                                 + 0.0414330958137562 w_k1_INTC_s002
                                 + 0.0405006845147041 w_k1_INTC_s003
                                 + 0.0939309545317539 w_k1_INTC_s004
                                 + 0.04172464764727 w_k1_INTC_s005
                                 + 0.0429389497912734 w_k1_INTC_s006
                                 + 0.0420926360196048 w_k1_INTC_s007
                                 + 0.0207148007270607 w_k1_INTC_s010
                                 + 0.0675973656998536 w_k1_INTC_s011
                                 + 0.0605003772640525 w_k1_INTC_s012
                                 + 0.056013750810016 w_k1_INTC_s013
                                 + 0.0256915095214984 w_k1_INTC_s014
                                 + 0.0290153042257405 w_k1_INTC_s015
                                 + 0.0148327480822514 w_k1_INTC_s016
                                 + 0.0111861835335864 w_k1_INTC_s017
                                 + 0.034379104492704 w_k1_INTC_s100
                                 + 0.0195121397118715 w_k1_INTC_s101
                                 + 0.033779309581713 w_k1_INTC_s102
                                 + 0.0460641644656805 w_k1_INTC_s103
                                 + 0.0670718073886939 w_k1_INTC_s104
                                 + 0.0701483231610146 w_k1_INTC_s105
                                 + 0.0294271497163251 w_k1_INTC_s106
                                 + 0.0167261172728896 w_k1_INTC_s107
                                 + 0.00381802197981081 w_k1_INTC_s110
                                 + 0.0050974103982361 w_k1_INTC_s111
                                 + 0.00633093282297772 w_k1_INTC_s112
                                 + 0.0140383846274234 w_k1_INTC_s113
                                 + 0.0051761284690553 w_k1_INTC_s114
                                 + 0.0072864216697699 w_k1_INTC_s115
                                 + 0.0144979535958036 w_k1_INTC_s116
                                 + 0.00432432824669512 w_k1_INTC_s117
                                 + 0.0363593205984496 w_k1_KO_s000
                                 + 0.0475106561755239 w_k1_KO_s001
                                 + 0.0504494555838783 w_k1_KO_s002
                                 + 0.041028210751472 w_k1_KO_s003
                                 + 0.0774844865299899 w_k1_KO_s004
                                 + 0.0409179290863707 w_k1_KO_s005
                                 + 0.0409306269464536 w_k1_KO_s006
                                 + 0.0411765705340741 w_k1_KO_s007
                                 + 0.0214694645335515 w_k1_KO_s010
                                 + 0.0605364956703994 w_k1_KO_s011
                                 + 0.0589870618741876 w_k1_KO_s012
                                 + 0.0512736708825657 w_k1_KO_s013
                                 + 0.0290034496345188 w_k1_KO_s014
                                 + 0.027531083320266 w_k1_KO_s015
                                 + 0.0154702321001721 w_k1_KO_s016
                                 + 0.0108179617326404 w_k1_KO_s017
                                 + 0.0420108787006579 w_k1_KO_s100
                                 + 0.0188614988004747 w_k1_KO_s101
                                 + 0.0332336702037193 w_k1_KO_s102
                                 + 0.0441725251378769 w_k1_KO_s103
                                 + 0.0566427517386601 w_k1_KO_s104
                                 + 0.0678155637133041 w_k1_KO_s105
                                 + 0.0310122818731536 w_k1_KO_s106
                                 + 0.0186566409715416 w_k1_KO_s107
                                 + 0.00377752478509971 w_k1_KO_s110
                                 + 0.00637109836236829 w_k1_KO_s111
                                 + 0.00688084086978919 w_k1_KO_s112
                                 + 0.013360366989243 w_k1_KO_s113
                                 + 0.00449943234542268 w_k1_KO_s114
                                 + 0.00678524975775605 w_k1_KO_s115
                                 + 0.0149991786270109 w_k1_KO_s116
                                 + 0.00506316427336239 w_k1_KO_s117
                                 + 0.0319538155070742 w_k1_MAT_s000
                                 + 0.0521694026102049 w_k1_MAT_s001
                                 + 0.0547073584177119 w_k1_MAT_s002
                                 + 0.0394975987387185 w_k1_MAT_s003
                                 + 0.0738079546921126 w_k1_MAT_s004
                                 + 0.0434762499415912 w_k1_MAT_s005
                                 + 0.039546266524216 w_k1_MAT_s006
                                 + 0.0404436798200501 w_k1_MAT_s007
                                 + 0.020739751066891 w_k1_MAT_s010
                                 + 0.0720134499570194 w_k1_MAT_s011
                                 + 0.0630268883867023 w_k1_MAT_s012
                                 + 0.0509848623513486 w_k1_MAT_s013
                                 + 0.0263647164739317 w_k1_MAT_s014
                                 + 0.0286551676536667 w_k1_MAT_s015
                                 + 0.0184877876265593 w_k1_MAT_s016
                                 + 0.0110597533308454 w_k1_MAT_s017
                                 + 0.0403738853349676 w_k1_MAT_s100
                                 + 0.0167708597273628 w_k1_MAT_s101
                                 + 0.0329126003507414 w_k1_MAT_s102
                                 + 0.040439360136432 w_k1_MAT_s103
                                 + 0.0546090756373011 w_k1_MAT_s104
                                 + 0.0704334027540447 w_k1_MAT_s105
                                 + 0.0355081606794148 w_k1_MAT_s106
                                 + 0.0157572986836168 w_k1_MAT_s107
                                 + 0.0052036451315763 w_k1_MAT_s110
                                 + 0.00535684571805905 w_k1_MAT_s111
                                 + 0.00725326216688908 w_k1_MAT_s112
                                 + 0.0132489218063895 w_k1_MAT_s113
                                 + 0.00444159182903751 w_k1_MAT_s114
                                 + 0.00945979441857185 w_k1_MAT_s115
                                 + 0.0157411568394946 w_k1_MAT_s116
                                 + 0.0061748780552763 w_k1_MAT_s117
                                 + 0.0343647480978215 w_k1_MSFT_s000
                                 + 0.042072457185838 w_k1_MSFT_s001
                                 + 0.0442915137883487 w_k1_MSFT_s002
                                 + 0.0449321529837081 w_k1_MSFT_s003
                                 + 0.0844725377386766 w_k1_MSFT_s004
                                 + 0.0405799042987706 w_k1_MSFT_s005
                                 + 0.0391279061135399 w_k1_MSFT_s006
                                 + 0.0432753427868532 w_k1_MSFT_s007
                                 + 0.021915519751251 w_k1_MSFT_s010
                                 + 0.0664684335194226 w_k1_MSFT_s011
                                 + 0.0597761525883854 w_k1_MSFT_s012
                                 + 0.0515052112313254 w_k1_MSFT_s013
                                 + 0.0267503429676654 w_k1_MSFT_s014
                                 + 0.0262484567025546 w_k1_MSFT_s015
                                 + 0.0163493253696313 w_k1_MSFT_s016
                                 + 0.0116657633955734 w_k1_MSFT_s017
                                 + 0.0418508407402328 w_k1_MSFT_s100
                                 + 0.0191375728609967 w_k1_MSFT_s101
                                 + 0.0320720331454095 w_k1_MSFT_s102
                                 + 0.0437647958251545 w_k1_MSFT_s103
                                 + 0.0567823937024885 w_k1_MSFT_s104
                                 + 0.0684011525489259 w_k1_MSFT_s105
                                 + 0.0305966684863418 w_k1_MSFT_s106
                                 + 0.0180048893978633 w_k1_MSFT_s107
                                 + 0.0040558157232691 w_k1_MSFT_s110
                                 + 0.0056753335544389 w_k1_MSFT_s111
                                 + 0.00679209432494158 w_k1_MSFT_s112
                                 + 0.0125964762287435 w_k1_MSFT_s113
                                 + 0.00497027618773152 w_k1_MSFT_s114
                                 + 0.00689612722848987 w_k1_MSFT_s115
                                 + 0.014586911553183 w_k1_MSFT_s116
                                 + 0.00469120427308351 w_k1_MSFT_s117
                                 + 0.0323429895314028 w_k1_PG_s000
                                 + 0.041175118593679 w_k1_PG_s001
                                 + 0.0458006010750252 w_k1_PG_s002
                                 + 0.0410873824646432 w_k1_PG_s003
                                 + 0.0792712688123555 w_k1_PG_s004
                                 + 0.0438334858783453 w_k1_PG_s005
                                 + 0.0410844903787522 w_k1_PG_s006
                                 + 0.037967788661195 w_k1_PG_s007
                                 + 0.0200271734416531 w_k1_PG_s010
                                 + 0.0569224508274132 w_k1_PG_s011
                                 + 0.0566378572429287 w_k1_PG_s012
                                 + 0.0493437805913467 w_k1_PG_s013
                                 + 0.0252306805585813 w_k1_PG_s014
                                 + 0.028983614466968 w_k1_PG_s015
                                 + 0.0156251789781396 w_k1_PG_s016
                                 + 0.0121345452588091 w_k1_PG_s017
                                 + 0.0366581129331906 w_k1_PG_s100
                                 + 0.0186190144892327 w_k1_PG_s101
                                 + 0.0311786206875993 w_k1_PG_s102
                                 + 0.0411328280801116 w_k1_PG_s103
                                 + 0.0562162767844087 w_k1_PG_s104
                                 + 0.0709661834759912 w_k1_PG_s105
                                 + 0.0321005783546985 w_k1_PG_s106
                                 + 0.0171533052460603 w_k1_PG_s107
                                 + 0.00443521304346891 w_k1_PG_s110
                                 + 0.00564829833382647 w_k1_PG_s111
                                 + 0.00727947304898186 w_k1_PG_s112
                                 + 0.0136780395501264 w_k1_PG_s113
                                 + 0.00501706628216391 w_k1_PG_s114
                                 + 0.00677635085981782 w_k1_PG_s115
                                 + 0.0168430950056815 w_k1_PG_s116
                                 + 0.0045977861626588 w_k1_PG_s117 >= 1.08
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
