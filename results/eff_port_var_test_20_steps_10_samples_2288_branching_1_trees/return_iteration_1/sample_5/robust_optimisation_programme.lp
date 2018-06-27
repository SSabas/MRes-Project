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
 w_k1_ADM_s0_return_constraint:  1.16662 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.894614 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.14953 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.92751 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.953635 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.11745 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.816532 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.886451 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.934832 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.1251 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.0634 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.22589 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.740901 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.49242 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.928103 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.34097 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.00513 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.34375 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.870715 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.702355 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.671156 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.1086 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.13327 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.807692 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.868306 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.910303 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.55159 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.974819 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.754588 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.02454 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.829796 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.00276 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.94364 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.27759 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.936376 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.22006 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.944528 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.808803 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.20057 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.916017 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.2572 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.925323 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.785496 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.11967 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.75312 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.837075 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.904743 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.31597 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.11577 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.39018 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.536026 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.60546 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.828604 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.32183 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.0478 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.50385 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.867084 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.592199 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.658789 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.1175 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.36249 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.842012 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.814177 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.965336 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.535391 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.22271 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.696247 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.91821 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.810491 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.33005 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.796117 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.03822 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.764226 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.40543 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.925067 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.587746 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.22482 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.838618 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.35831 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.931655 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.743449 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.23062 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.600061 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.781096 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.986406 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.03443 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.23071 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.331 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.61325 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.91313 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.895582 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.45922 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.856407 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.8344 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.701995 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.633691 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.552362 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.12789 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.57575 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.833658 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.77086 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.19148 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.522847 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.20461 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.708195 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.886654 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.658439 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.38369 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.697127 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.21844 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.81342 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.02657 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.896423 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.517005 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.13477 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.904105 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.27469 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.896413 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.856737 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.11772 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.795569 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.815418 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.03052 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.986999 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.05731 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.31277 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.738979 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.31639 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.02948 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.41273 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.980951 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.37394 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.816375 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.73238 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.762715 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.12051 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.37067 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.774021 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.843585 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.08907 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.597708 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.05361 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.762929 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.02669 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.700768 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.13496 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.834858 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.1806 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.857421 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.1545 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.90668 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.641272 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.20464 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.871616 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.950682 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.0552 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.804737 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.1986 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.668129 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.1036 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.763318 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.31262 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.35793 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.887569 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.548467 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.61979 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.580795 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.0833 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.938969 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.84323 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.14979 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.794233 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.600735 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.16246 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.890594 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.16528 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.00273 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.935525 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.896853 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.43622 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.701821 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.670077 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.44898 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.4278 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.726061 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.744841 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.05867 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.958289 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.917253 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.801246 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.12718 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.957075 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.09234 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.961764 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.882108 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.05041 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.823522 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.91649 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.12798 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.12123 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.01952 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.05827 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.780934 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.34508 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.887287 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.09884 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.927062 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.27207 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.796833 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.817109 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.717047 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.922681 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.17425 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.762958 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.782091 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.13391 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.745888 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.16466 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.835403 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.942803 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.812698 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.1124 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.778113 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.13741 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.911545 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.0939 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.935003 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.79713 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.17654 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.843898 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.04395 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.902293 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.89402 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.12458 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.715662 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.02273 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.860494 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.15964 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.36933 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.995227 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.694946 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.63467 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.756484 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.2258 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.954183 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.67196 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.944735 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.827612 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.70526 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.11394 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.03396 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.997964 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.800572 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.993765 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.690158 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.18939 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.678188 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.714148 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.1512 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.27626 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.821605 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.17243 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.9031 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.07144 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.940718 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.701048 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.15707 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.853178 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.994452 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.05159 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.867866 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.16413 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.659503 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.36213 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.926174 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.70368 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.78489 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.12965 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.40357 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.51673 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.663568 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.5065 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.1494 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.8761 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.10166 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.743702 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.666919 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.18216 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.813429 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.28531 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.818743 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.674231 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.707031 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.978362 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.534673 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.666377 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.30385 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.6364 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.747915 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.77645 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.1416 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.20747 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.788227 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.696399 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.11992 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.921515 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.16681 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.893678 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.855629 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.1353 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.774632 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.943313 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.0214 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.0712 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.11931 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.09804 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.855171 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.31666 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.916994 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.3282 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.00601 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.40508 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.870883 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.769381 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.823984 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.09047 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.28586 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.873251 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.777307 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.11609 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.715686 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.09327 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.747471 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.925221 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.710036 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.992969 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.814306 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.16936 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.863835 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.993297 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.933762 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.677689 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.13394 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.917863 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.02326 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.897969 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.877741 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.11441 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.845137 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.00775 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.966271 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.06876 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.13705 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.11313 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.834079 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.3783 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.945527 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.22839 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.01779 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.24944 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.921839 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.776089 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.828393 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.18528 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.15751 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.03091 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.964495 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.11189 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.789048 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.1293 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.804131 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.919822 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.977599 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.16394 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.844856 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.0666 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.909626 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.1166 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.910182 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.753559 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.238999696017032 z_k1_s0000
                                 - 0.353999275418618 z_k1_s0001
                                 - 0.100999921735216 z_k1_s0002
                                 - 0.108000029568314 z_k1_s0003
                                 - 0.159999475471753 z_k1_s0004
                                 - 0.0939999534020737 z_k1_s0005
                                 - 0.098999910854324 z_k1_s0006
                                 - 0.239999910707413 z_k1_s0007
                                 - 0.477001850890798 z_k1_s0010
                                 - 0.28600068820595 z_k1_s0011
                                 - 0.168000314506101 z_k1_s0012
                                 - 0.50900031230759 z_k1_s0013
                                 - 0.252000636309503 z_k1_s0014
                                 - 0.680001037928952 z_k1_s0015
                                 - 0.292000523134363 z_k1_s0016
                                 - 0.627002660611661 z_k1_s0017
                                 - 0.248999801935289 z_k1_s0020
                                 - 0.155000063964438 z_k1_s0021
                                 - 0.2319999871107 z_k1_s0022
                                 - 0.360000649930044 z_k1_s0023
                                 - 0.379001223898289 z_k1_s0024
                                 - 0.504998213375514 z_k1_s0025
                                 - 0.701998029365295 z_k1_s0026
                                 - 0.655999025651441 z_k1_s0027
                                 - 0.0889995399759465 z_k1_s0030
                                 - 0.131000080930291 z_k1_s0031
                                 - 0.0799999242567648 z_k1_s0032
                                 - 0.121999782011647 z_k1_s0033
                                 - 0.0799999242567648 z_k1_s0034
                                 - 0.0659999716718041 z_k1_s0035
                                 - 0.0759999622038997 z_k1_s0036
                                 - 0.210000142773739 z_k1_s0037
                                 - 0.234000267784371 z_k1_s0040
                                 - 0.483000500251011 z_k1_s0041
                                 - 0.352001942333902 z_k1_s0042
                                 - 0.295000250119081 z_k1_s0043
                                 - 0.896001795118863 z_k1_s0044
                                 - 0.195000170670246 z_k1_s0045
                                 - 0.315999381933269 z_k1_s0046
                                 - 0.378000117674374 z_k1_s0047
                                 - 0.190999261595563 z_k1_s0050
                                 - 0.16700063343654 z_k1_s0051
                                 - 0.211999282734309 z_k1_s0052
                                 - 0.0829999904817397 z_k1_s0053
                                 - 0.108999923586884 z_k1_s0054
                                 - 0.339000341239752 z_k1_s0055
                                 - 0.199999270655025 z_k1_s0056
                                 - 0.095000002561023 z_k1_s0057
                                 - 0.0860000700495344 z_k1_s0060
                                 - 0.138000225177221 z_k1_s0061
                                 - 0.119000106326686 z_k1_s0062
                                 - 0.240000508743057 z_k1_s0063
                                 - 0.227000604661209 z_k1_s0064
                                 - 0.0640000458647664 z_k1_s0065
                                 - 0.245000575466902 z_k1_s0066
                                 - 0.228000079205684 z_k1_s0067
                                 - 0.352001284992005 z_k1_s0070
                                 - 0.314000166904288 z_k1_s0071
                                 - 0.256000188355268 z_k1_s0072
                                 - 0.129000447602104 z_k1_s0073
                                 - 0.328001010832821 z_k1_s0074
                                 - 0.25300015408537 z_k1_s0075
                                 - 0.177000585519068 z_k1_s0076
                                 - 0.243000723854718 z_k1_s0077
                                 - 0.605997381059135 z_k1_s0100
                                 - 0.38699936276138 z_k1_s0101
                                 - 0.499996794823214 z_k1_s0102
                                 - 0.829000344929262 z_k1_s0103
                                 - 1.22099706162424 z_k1_s0104
                                 - 0.523999869512095 z_k1_s0105
                                 - 0.265999377139726 z_k1_s0106
                                 - 0.252999469646304 z_k1_s0107
                                 - 0.128999982805081 z_k1_s0110
                                 - 0.289999308632087 z_k1_s0111
                                 - 0.196999849725685 z_k1_s0112
                                 - 0.109999846636224 z_k1_s0113
                                 - 0.251000004074198 z_k1_s0114
                                 - 0.436999763886343 z_k1_s0115
                                 - 0.37599901650236 z_k1_s0116
                                 - 0.315000374116733 z_k1_s0117
                                 - 0.359999190980963 z_k1_s0120
                                 - 0.22299978719101 z_k1_s0121
                                 - 0.296000257473316 z_k1_s0122
                                 - 0.127999943015474 z_k1_s0123
                                 - 0.511999772061894 z_k1_s0124
                                 - 0.194999942381388 z_k1_s0125
                                 - 0.129999961587592 z_k1_s0126
                                 - 0.23199924796549 z_k1_s0127
                                 - 0.491998849425694 z_k1_s0130
                                 - 0.283999744113812 z_k1_s0131
                                 - 0.484999120962191 z_k1_s0132
                                 - 0.156999990078344 z_k1_s0133
                                 - 0.431998989739634 z_k1_s0134
                                 - 0.672999701977998 z_k1_s0135
                                 - 0.166999864626072 z_k1_s0136
                                 - 0.371999130053574 z_k1_s0137
                                 - 0.236999973769225 z_k1_s0140
                                 - 0.903998316149176 z_k1_s0141
                                 - 0.926998501679535 z_k1_s0142
                                 - 0.49400194486932 z_k1_s0143
                                 - 0.268999885098398 z_k1_s0144
                                 - 0.389000139082833 z_k1_s0145
                                 - 0.431000110677378 z_k1_s0146
                                 - 1.04100003326484 z_k1_s0147
                                 - 0.214999919865545 z_k1_s0150
                                 - 0.194000020362415 z_k1_s0151
                                 - 0.216000104832393 z_k1_s0152
                                 - 0.140000111354336 z_k1_s0153
                                 - 0.402999299226954 z_k1_s0154
                                 - 0.249000817537458 z_k1_s0155
                                 - 0.494998970574022 z_k1_s0156
                                 - 0.431999272064629 z_k1_s0157
                                 - 0.21499974287049 z_k1_s0160
                                 - 0.539998429667085 z_k1_s0161
                                 - 0.347000783897651 z_k1_s0162
                                 - 0.45200030044435 z_k1_s0163
                                 - 0.494999094860531 z_k1_s0164
                                 - 0.251999480910921 z_k1_s0165
                                 - 0.56499877255833 z_k1_s0166
                                 - 0.339998892531342 z_k1_s0167
                                 - 1.86099837161726 z_k1_s0170
                                 - 1.4609959506074 z_k1_s0171
                                 - 1.1369993773457 z_k1_s0172
                                 - 0.795999622473203 z_k1_s0173
                                 - 1.74399747105201 z_k1_s0174
                                 - 0.789999008898452 z_k1_s0175
                                 - 1.07300052934306 z_k1_s0176
                                 - 0.758999206109923 z_k1_s0177
                                 - 0.44599780991098 z_k1_s1000
                                 - 0.490000267894613 z_k1_s1001
                                 - 0.202999870342333 z_k1_s1002
                                 - 0.821997987977083 z_k1_s1003
                                 - 0.746000174088805 z_k1_s1004
                                 - 0.190999843155142 z_k1_s1005
                                 - 0.30299976000265 z_k1_s1006
                                 - 0.167999706821457 z_k1_s1007
                                 - 0.115000061525662 z_k1_s1010
                                 - 0.244000158741414 z_k1_s1011
                                 - 0.132999798555674 z_k1_s1012
                                 - 0.163999645940796 z_k1_s1013
                                 - 0.0509999755852763 z_k1_s1014
                                 - 0.0769999800954228 z_k1_s1015
                                 - 0.230000123051324 z_k1_s1016
                                 - 0.0669999700453636 z_k1_s1017
                                 - 0.140999987253841 z_k1_s1020
                                 - 0.215000267319686 z_k1_s1021
                                 - 0.316000652000096 z_k1_s1022
                                 - 0.233000816262728 z_k1_s1023
                                 - 0.324000895974781 z_k1_s1024
                                 - 0.104999967567754 z_k1_s1025
                                 - 0.110999970848769 z_k1_s1026
                                 - 0.351999952886186 z_k1_s1027
                                 - 0.392002013243544 z_k1_s1030
                                 - 0.414999911312059 z_k1_s1031
                                 - 0.23700051515043 z_k1_s1032
                                 - 0.523001035655809 z_k1_s1033
                                 - 0.250000583591673 z_k1_s1034
                                 - 0.355999449124574 z_k1_s1035
                                 - 0.840002906661917 z_k1_s1036
                                 - 0.271000492057892 z_k1_s1037
                                 - 0.0840000225396694 z_k1_s1040
                                 - 0.238000388762663 z_k1_s1041
                                 - 0.144000243839601 z_k1_s1042
                                 - 0.0510000393348203 z_k1_s1043
                                 - 0.0619999995364087 z_k1_s1044
                                 - 0.202000071302552 z_k1_s1045
                                 - 0.154999998841022 z_k1_s1046
                                 - 0.0900000754496878 z_k1_s1047
                                 - 0.935003108918813 z_k1_s1050
                                 - 0.289000195928348 z_k1_s1051
                                 - 0.394000421898733 z_k1_s1052
                                 - 0.469000773120709 z_k1_s1053
                                 - 0.803001002621988 z_k1_s1054
                                 - 0.536002781712119 z_k1_s1055
                                 - 0.29900044944491 z_k1_s1056
                                 - 0.70399942289937 z_k1_s1057
                                 - 0.126999941883718 z_k1_s1060
                                 - 0.424998856087546 z_k1_s1061
                                 - 0.154999915861235 z_k1_s1062
                                 - 0.185999857093496 z_k1_s1063
                                 - 0.411999557183482 z_k1_s1064
                                 - 0.17299992908963 z_k1_s1065
                                 - 0.221000663365127 z_k1_s1066
                                 - 0.397999989594591 z_k1_s1067
                                 - 0.221999910398004 z_k1_s1070
                                 - 0.134999925971771 z_k1_s1071
                                 - 0.427998876281401 z_k1_s1072
                                 - 0.39999951287948 z_k1_s1073
                                 - 0.386999980585629 z_k1_s1074
                                 - 0.378998785328754 z_k1_s1075
                                 - 0.103999873348665 z_k1_s1076
                                 - 0.355000019555271 z_k1_s1077
                                 - 0.451000249020703 z_k1_s1100
                                 - 0.219999937783255 z_k1_s1101
                                 - 0.315000542460165 z_k1_s1102
                                 - 0.236000056025683 z_k1_s1103
                                 - 0.485000175660837 z_k1_s1104
                                 - 0.162999938557138 z_k1_s1105
                                 - 0.181000006654865 z_k1_s1106
                                 - 0.546001113397632 z_k1_s1107
                                 - 2.11799857426081 z_k1_s1110
                                 - 0.887998449697842 z_k1_s1111
                                 - 0.870001579224189 z_k1_s1112
                                 - 0.450999281474766 z_k1_s1113
                                 - 0.939994503499426 z_k1_s1114
                                 - 0.698998449196662 z_k1_s1115
                                 - 1.11599701894685 z_k1_s1116
                                 - 0.559998416481051 z_k1_s1117
                                 - 0.335999725016063 z_k1_s1120
                                 - 0.293999633314143 z_k1_s1121
                                 - 0.488999194559376 z_k1_s1122
                                 - 0.434998788485682 z_k1_s1123
                                 - 0.398000004471176 z_k1_s1124
                                 - 0.585001037134044 z_k1_s1125
                                 - 0.558000834097197 z_k1_s1126
                                 - 0.26699976647706 z_k1_s1127
                                 - 0.556000182625232 z_k1_s1130
                                 - 1.33599850199451 z_k1_s1131
                                 - 0.715000380648199 z_k1_s1132
                                 - 2.37400363953962 z_k1_s1133
                                 - 0.588000128931675 z_k1_s1134
                                 - 0.551000283977417 z_k1_s1135
                                 - 0.60300056857565 z_k1_s1136
                                 - 0.714000847138952 z_k1_s1137
                                 - 0.217999956569895 z_k1_s1140
                                 - 0.265000113588164 z_k1_s1141
                                 - 0.227000090398918 z_k1_s1142
                                 - 0.462000570690285 z_k1_s1143
                                 - 0.265999857524739 z_k1_s1144
                                 - 0.725000282005356 z_k1_s1145
                                 - 0.30800038046019 z_k1_s1146
                                 - 0.576998555394713 z_k1_s1147
                                 - 0.315999182644937 z_k1_s1150
                                 - 0.130999950503811 z_k1_s1151
                                 - 0.129999882206897 z_k1_s1152
                                 - 0.28599896973587 z_k1_s1153
                                 - 0.143999920364517 z_k1_s1154
                                 - 0.425999351312069 z_k1_s1155
                                 - 0.233999824692382 z_k1_s1156
                                 - 0.168999791789016 z_k1_s1157
                                 - 0.516001099262929 z_k1_s1160
                                 - 0.669000954275167 z_k1_s1161
                                 - 0.170000522905057 z_k1_s1162
                                 - 0.285001191106719 z_k1_s1163
                                 - 0.564001879662029 z_k1_s1164
                                 - 0.330999643980565 z_k1_s1165
                                 - 0.330001720230114 z_k1_s1166
                                 - 0.37500224935351 z_k1_s1167
                                 - 0.441000948200446 z_k1_s1170
                                 - 0.494998731471173 z_k1_s1171
                                 - 0.452000985144214 z_k1_s1172
                                 - 0.452000985144214 z_k1_s1173
                                 - 0.326000295515745 z_k1_s1174
                                 - 0.20999987928148 z_k1_s1175
                                 - 0.288999958086825 z_k1_s1176
                                 - 0.265999827549885 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.943086 w_k1_ADM_s000
                                 + 0.839628 w_k1_AXP_s000
                                 + 0.946903 w_k1_DIS_s000
                                 + 1.01376 w_k1_FDX_s000
                                 + 0.777061 w_k1_IBM_s000
                                 + 1.0707 w_k1_INTC_s000
                                 + 0.735409 w_k1_KO_s000
                                 + 0.489638 w_k1_MAT_s000
                                 + 0.914979 w_k1_MSFT_s000
                                 + 0.937277 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.40294 w_k1_ADM_s000
                                 + 1.1606 w_k1_AXP_s000 + 1.29549 w_k1_DIS_s000
                                 + 1.25916 w_k1_FDX_s000
                                 + 1.07907 w_k1_IBM_s000
                                 + 1.13673 w_k1_INTC_s000
                                 + 1.12544 w_k1_KO_s000 + 1.21978 w_k1_MAT_s000
                                 + 1.12677 w_k1_MSFT_s000
                                 + 1.07854 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.09891 w_k1_ADM_s000
                                 + 1.02148 w_k1_AXP_s000
                                 + 0.885006 w_k1_DIS_s000
                                 + 0.889293 w_k1_FDX_s000
                                 + 1.07901 w_k1_IBM_s000
                                 + 1.04441 w_k1_INTC_s000
                                 + 0.940252 w_k1_KO_s000 + 1.2293 w_k1_MAT_s000
                                 + 0.879266 w_k1_MSFT_s000
                                 + 1.01576 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.11667 w_k1_ADM_s000
                                 + 0.977175 w_k1_AXP_s000
                                 + 1.10854 w_k1_DIS_s000
                                 + 0.922687 w_k1_FDX_s000
                                 + 1.23432 w_k1_IBM_s000
                                 + 0.887023 w_k1_INTC_s000
                                 + 1.16523 w_k1_KO_s000 + 1.53657 w_k1_MAT_s000
                                 + 0.971966 w_k1_MSFT_s000
                                 + 1.02084 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.874115 w_k1_ADM_s000
                                 + 0.861555 w_k1_AXP_s000
                                 + 0.669166 w_k1_DIS_s000
                                 + 0.729843 w_k1_FDX_s000
                                 + 1.24747 w_k1_IBM_s000
                                 + 0.676045 w_k1_INTC_s000
                                 + 0.887676 w_k1_KO_s000
                                 + 1.56527 w_k1_MAT_s000
                                 + 0.723206 w_k1_MSFT_s000
                                 + 0.861205 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.847539 w_k1_ADM_s000
                                 + 0.924541 w_k1_AXP_s000
                                 + 0.609836 w_k1_DIS_s000
                                 + 0.802473 w_k1_FDX_s000
                                 + 0.727885 w_k1_IBM_s000
                                 + 0.78834 w_k1_INTC_s000
                                 + 0.76556 w_k1_KO_s000
                                 + 0.961885 w_k1_MAT_s000
                                 + 0.91416 w_k1_MSFT_s000
                                 + 0.849416 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.732161 w_k1_ADM_s000
                                 + 0.654788 w_k1_AXP_s000
                                 + 0.73906 w_k1_DIS_s000
                                 + 0.728087 w_k1_FDX_s000
                                 + 0.71314 w_k1_IBM_s000
                                 + 0.743915 w_k1_INTC_s000
                                 + 0.646701 w_k1_KO_s000
                                 + 0.832915 w_k1_MAT_s000
                                 + 0.756392 w_k1_MSFT_s000
                                 + 0.806526 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.17886 w_k1_ADM_s000
                                 + 1.13257 w_k1_AXP_s000
                                 + 1.41462 w_k1_DIS_s000
                                 + 1.12464 w_k1_FDX_s000
                                 + 1.05844 w_k1_IBM_s000
                                 + 1.11419 w_k1_INTC_s000
                                 + 1.22939 w_k1_KO_s000 + 1.6351 w_k1_MAT_s000
                                 + 1.15769 w_k1_MSFT_s000
                                 + 1.11197 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.873292 w_k1_ADM_s001
                                 + 0.850722 w_k1_AXP_s001
                                 + 0.892046 w_k1_DIS_s001
                                 + 0.984443 w_k1_FDX_s001
                                 + 0.682093 w_k1_IBM_s001
                                 + 0.991393 w_k1_INTC_s001
                                 + 0.76332 w_k1_KO_s001
                                 + 0.495459 w_k1_MAT_s001
                                 + 0.920254 w_k1_MSFT_s001
                                 + 0.832271 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.956024 w_k1_ADM_s001
                                 + 1.1627 w_k1_AXP_s001
                                 + 0.893169 w_k1_DIS_s001
                                 + 1.05982 w_k1_FDX_s001
                                 + 1.09901 w_k1_IBM_s001
                                 + 1.07842 w_k1_INTC_s001
                                 + 0.885527 w_k1_KO_s001
                                 + 1.01202 w_k1_MAT_s001
                                 + 0.886441 w_k1_MSFT_s001
                                 + 0.890443 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.14927 w_k1_ADM_s001
                                 + 0.88346 w_k1_AXP_s001
                                 + 0.823394 w_k1_DIS_s001
                                 + 0.803818 w_k1_FDX_s001
                                 + 0.852775 w_k1_IBM_s001
                                 + 0.896838 w_k1_INTC_s001
                                 + 0.880956 w_k1_KO_s001
                                 + 0.872236 w_k1_MAT_s001
                                 + 0.753338 w_k1_MSFT_s001
                                 + 0.987937 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.37599 w_k1_ADM_s001
                                 + 1.15991 w_k1_AXP_s001
                                 + 1.19791 w_k1_DIS_s001
                                 + 1.16279 w_k1_FDX_s001
                                 + 0.859921 w_k1_IBM_s001
                                 + 0.93101 w_k1_INTC_s001
                                 + 0.934154 w_k1_KO_s001
                                 + 0.722906 w_k1_MAT_s001
                                 + 1.0396 w_k1_MSFT_s001 + 1.01131 w_k1_PG_s001
                                 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.688207 w_k1_ADM_s001
                                 + 0.680257 w_k1_AXP_s001
                                 + 0.596332 w_k1_DIS_s001
                                 + 0.744518 w_k1_FDX_s001
                                 + 0.683327 w_k1_IBM_s001
                                 + 0.87198 w_k1_INTC_s001
                                 + 0.716783 w_k1_KO_s001
                                 + 0.680402 w_k1_MAT_s001
                                 + 0.795762 w_k1_MSFT_s001
                                 + 0.773976 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.973465 w_k1_ADM_s001
                                 + 1.21366 w_k1_AXP_s001
                                 + 1.37287 w_k1_DIS_s001
                                 + 1.14158 w_k1_FDX_s001
                                 + 1.14422 w_k1_IBM_s001
                                 + 1.06917 w_k1_INTC_s001
                                 + 1.06945 w_k1_KO_s001
                                 + 0.805099 w_k1_MAT_s001
                                 + 1.01803 w_k1_MSFT_s001
                                 + 1.06179 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.879604 w_k1_ADM_s001
                                 + 1.26944 w_k1_AXP_s001
                                 + 1.50849 w_k1_DIS_s001
                                 + 1.10781 w_k1_FDX_s001
                                 + 1.72867 w_k1_IBM_s001
                                 + 1.08681 w_k1_INTC_s001
                                 + 1.29152 w_k1_KO_s001 + 1.42551 w_k1_MAT_s001
                                 + 1.05869 w_k1_MSFT_s001
                                 + 1.11088 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.806144 w_k1_ADM_s001
                                 + 0.799206 w_k1_AXP_s001
                                 + 0.831251 w_k1_DIS_s001
                                 + 0.828346 w_k1_FDX_s001
                                 + 1.02782 w_k1_IBM_s001
                                 + 0.97049 w_k1_INTC_s001
                                 + 0.92002 w_k1_KO_s001 + 0.7229 w_k1_MAT_s001
                                 + 0.876344 w_k1_MSFT_s001
                                 + 0.977789 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.10929 w_k1_ADM_s002
                                 + 0.781887 w_k1_AXP_s002
                                 + 1.2323 w_k1_DIS_s002 + 1.09968 w_k1_FDX_s002
                                 + 0.897305 w_k1_IBM_s002
                                 + 1.14247 w_k1_INTC_s002
                                 + 1.19419 w_k1_KO_s002
                                 + 0.501055 w_k1_MAT_s002
                                 + 1.24862 w_k1_MSFT_s002
                                 + 1.11219 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.4109 w_k1_ADM_s002
                                 + 1.32172 w_k1_AXP_s002 + 1.7554 w_k1_DIS_s002
                                 + 1.31025 w_k1_FDX_s002
                                 + 0.94016 w_k1_IBM_s002
                                 + 1.24617 w_k1_INTC_s002
                                 + 1.29717 w_k1_KO_s002
                                 + 0.688276 w_k1_MAT_s002
                                 + 1.51663 w_k1_MSFT_s002
                                 + 1.09979 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.856269 w_k1_ADM_s002
                                 + 0.753786 w_k1_AXP_s002
                                 + 0.572572 w_k1_DIS_s002
                                 + 0.745391 w_k1_FDX_s002
                                 + 0.634698 w_k1_IBM_s002
                                 + 0.689011 w_k1_INTC_s002
                                 + 0.718548 w_k1_KO_s002
                                 + 0.573796 w_k1_MAT_s002
                                 + 0.727345 w_k1_MSFT_s002
                                 + 0.754183 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.03655 w_k1_ADM_s002
                                 + 1.05749 w_k1_AXP_s002
                                 + 1.05857 w_k1_DIS_s002
                                 + 0.92996 w_k1_FDX_s002 + 1.4542 w_k1_IBM_s002
                                 + 1.04026 w_k1_INTC_s002
                                 + 1.39387 w_k1_KO_s002 + 1.23962 w_k1_MAT_s002
                                 + 1.09311 w_k1_MSFT_s002
                                 + 1.14836 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.15496 w_k1_ADM_s002
                                 + 1.38187 w_k1_AXP_s002
                                 + 1.42095 w_k1_DIS_s002
                                 + 1.23294 w_k1_FDX_s002
                                 + 1.16438 w_k1_IBM_s002
                                 + 1.12811 w_k1_INTC_s002
                                 + 1.45659 w_k1_KO_s002 + 1.29133 w_k1_MAT_s002
                                 + 1.24989 w_k1_MSFT_s002
                                 + 1.18841 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.990199 w_k1_ADM_s002
                                 + 0.7447 w_k1_AXP_s002
                                 + 0.862203 w_k1_DIS_s002
                                 + 0.848673 w_k1_FDX_s002
                                 + 0.820663 w_k1_IBM_s002
                                 + 0.788441 w_k1_INTC_s002
                                 + 0.900851 w_k1_KO_s002
                                 + 0.825212 w_k1_MAT_s002
                                 + 1.04809 w_k1_MSFT_s002
                                 + 0.90213 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.17717 w_k1_ADM_s002
                                 + 1.23808 w_k1_AXP_s002
                                 + 0.97871 w_k1_DIS_s002
                                 + 1.07482 w_k1_FDX_s002
                                 + 0.866245 w_k1_IBM_s002
                                 + 1.13231 w_k1_INTC_s002
                                 + 1.08314 w_k1_KO_s002 + 1.00134 w_k1_MAT_s002
                                 + 1.05602 w_k1_MSFT_s002
                                 + 1.10048 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.995317 w_k1_ADM_s002
                                 + 0.920613 w_k1_AXP_s002
                                 + 0.699416 w_k1_DIS_s002
                                 + 0.820933 w_k1_FDX_s002
                                 + 0.964845 w_k1_IBM_s002
                                 + 0.822996 w_k1_INTC_s002
                                 + 0.868408 w_k1_KO_s002
                                 + 0.881478 w_k1_MAT_s002
                                 + 0.839156 w_k1_MSFT_s002
                                 + 0.909453 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.880801 w_k1_ADM_s003
                                 + 0.991019 w_k1_AXP_s003
                                 + 0.850338 w_k1_DIS_s003
                                 + 1.09095 w_k1_FDX_s003
                                 + 0.657964 w_k1_IBM_s003
                                 + 1.0136 w_k1_INTC_s003
                                 + 0.604115 w_k1_KO_s003
                                 + 0.731234 w_k1_MAT_s003
                                 + 0.821811 w_k1_MSFT_s003
                                 + 0.864415 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.9287 w_k1_ADM_s003
                                 + 1.54296 w_k1_AXP_s003
                                 + 1.18259 w_k1_DIS_s003
                                 + 1.29504 w_k1_FDX_s003
                                 + 1.03358 w_k1_IBM_s003
                                 + 1.14336 w_k1_INTC_s003
                                 + 1.05658 w_k1_KO_s003 + 1.29908 w_k1_MAT_s003
                                 + 1.01284 w_k1_MSFT_s003
                                 + 1.01701 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.976757 w_k1_ADM_s003
                                 + 1.36911 w_k1_AXP_s003
                                 + 1.50207 w_k1_DIS_s003 + 1.3104 w_k1_FDX_s003
                                 + 1.25027 w_k1_IBM_s003
                                 + 1.06296 w_k1_INTC_s003
                                 + 1.19323 w_k1_KO_s003 + 1.55838 w_k1_MAT_s003
                                 + 1.08986 w_k1_MSFT_s003
                                 + 1.11776 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.713843 w_k1_ADM_s003
                                 + 1.05527 w_k1_AXP_s003 + 1.081 w_k1_DIS_s003
                                 + 0.96201 w_k1_FDX_s003
                                 + 1.03172 w_k1_IBM_s003
                                 + 1.10265 w_k1_INTC_s003
                                 + 1.02822 w_k1_KO_s003 + 0.96744 w_k1_MAT_s003
                                 + 0.876159 w_k1_MSFT_s003
                                 + 0.980845 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.01743 w_k1_ADM_s003
                                 + 1.16791 w_k1_AXP_s003
                                 + 0.917161 w_k1_DIS_s003
                                 + 0.969456 w_k1_FDX_s003
                                 + 1.20006 w_k1_IBM_s003
                                 + 0.699413 w_k1_INTC_s003
                                 + 0.964166 w_k1_KO_s003
                                 + 1.12588 w_k1_MAT_s003
                                 + 0.838989 w_k1_MSFT_s003
                                 + 1.07603 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.598075 w_k1_ADM_s003
                                 + 0.813974 w_k1_AXP_s003
                                 + 0.800654 w_k1_DIS_s003
                                 + 0.786172 w_k1_FDX_s003
                                 + 1.21712 w_k1_IBM_s003
                                 + 0.959885 w_k1_INTC_s003
                                 + 0.875726 w_k1_KO_s003
                                 + 1.44574 w_k1_MAT_s003
                                 + 0.861875 w_k1_MSFT_s003
                                 + 0.884655 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.943489 w_k1_ADM_s003
                                 + 1.41038 w_k1_AXP_s003
                                 + 1.13996 w_k1_DIS_s003
                                 + 1.07925 w_k1_FDX_s003
                                 + 1.85552 w_k1_IBM_s003
                                 + 1.2209 w_k1_INTC_s003 + 1.30008 w_k1_KO_s003
                                 + 1.78578 w_k1_MAT_s003
                                 + 1.06138 w_k1_MSFT_s003
                                 + 1.14733 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.675996 w_k1_ADM_s003
                                 + 0.899786 w_k1_AXP_s003
                                 + 0.758078 w_k1_DIS_s003
                                 + 0.852831 w_k1_FDX_s003
                                 + 0.770364 w_k1_IBM_s003
                                 + 0.879503 w_k1_INTC_s003
                                 + 0.864553 w_k1_KO_s003
                                 + 0.756999 w_k1_MAT_s003
                                 + 0.827314 w_k1_MSFT_s003
                                 + 0.854254 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.87796 w_k1_ADM_s004
                                 + 0.585224 w_k1_AXP_s004
                                 + 0.648769 w_k1_DIS_s004
                                 + 0.793484 w_k1_FDX_s004
                                 + 0.617089 w_k1_IBM_s004
                                 + 0.794215 w_k1_INTC_s004
                                 + 0.692911 w_k1_KO_s004
                                 + 0.371217 w_k1_MAT_s004
                                 + 0.808915 w_k1_MSFT_s004
                                 + 0.721514 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.859836 w_k1_ADM_s004
                                 + 0.777273 w_k1_AXP_s004
                                 + 0.826243 w_k1_DIS_s004
                                 + 0.77849 w_k1_FDX_s004
                                 + 1.01198 w_k1_IBM_s004
                                 + 1.00922 w_k1_INTC_s004
                                 + 0.956512 w_k1_KO_s004
                                 + 0.794562 w_k1_MAT_s004
                                 + 0.879906 w_k1_MSFT_s004
                                 + 1.02262 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.12961 w_k1_ADM_s004
                                 + 1.19809 w_k1_AXP_s004
                                 + 1.15194 w_k1_DIS_s004
                                 + 1.03038 w_k1_FDX_s004
                                 + 1.79617 w_k1_IBM_s004
                                 + 1.08564 w_k1_INTC_s004
                                 + 1.29734 w_k1_KO_s004 + 1.53446 w_k1_MAT_s004
                                 + 0.958585 w_k1_MSFT_s004
                                 + 1.22021 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.694096 w_k1_ADM_s004
                                 + 0.81913 w_k1_AXP_s004
                                 + 0.645571 w_k1_DIS_s004
                                 + 0.813418 w_k1_FDX_s004
                                 + 0.961061 w_k1_IBM_s004
                                 + 1.00973 w_k1_INTC_s004
                                 + 0.747033 w_k1_KO_s004
                                 + 0.84944 w_k1_MAT_s004
                                 + 0.800817 w_k1_MSFT_s004
                                 + 0.796316 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.02077 w_k1_ADM_s004
                                 + 0.985304 w_k1_AXP_s004
                                 + 1.05686 w_k1_DIS_s004
                                 + 1.07315 w_k1_FDX_s004
                                 + 0.973381 w_k1_IBM_s004
                                 + 0.998372 w_k1_INTC_s004
                                 + 1.00944 w_k1_KO_s004 + 0.91279 w_k1_MAT_s004
                                 + 1.12922 w_k1_MSFT_s004
                                 + 1.08327 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.19402 w_k1_ADM_s004
                                 + 1.14592 w_k1_AXP_s004 + 1.4883 w_k1_DIS_s004
                                 + 1.33749 w_k1_FDX_s004
                                 + 0.967749 w_k1_IBM_s004
                                 + 1.25718 w_k1_INTC_s004
                                 + 1.10035 w_k1_KO_s004
                                 + 0.871221 w_k1_MAT_s004
                                 + 1.29664 w_k1_MSFT_s004
                                 + 1.07744 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.14989 w_k1_ADM_s004
                                 + 1.36278 w_k1_AXP_s004
                                 + 1.39613 w_k1_DIS_s004
                                 + 1.28337 w_k1_FDX_s004 + 1.1406 w_k1_IBM_s004
                                 + 1.43151 w_k1_INTC_s004 + 1.1646 w_k1_KO_s004
                                 + 1.11241 w_k1_MAT_s004
                                 + 1.23518 w_k1_MSFT_s004
                                 + 1.26706 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.867167 w_k1_ADM_s004
                                 + 0.938975 w_k1_AXP_s004
                                 + 0.933089 w_k1_DIS_s004
                                 + 1.08414 w_k1_FDX_s004
                                 + 0.656794 w_k1_IBM_s004
                                 + 0.974477 w_k1_INTC_s004
                                 + 0.792498 w_k1_KO_s004
                                 + 0.651344 w_k1_MAT_s004
                                 + 1.04082 w_k1_MSFT_s004
                                 + 1.00563 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.832767 w_k1_ADM_s005
                                 + 0.833157 w_k1_AXP_s005
                                 + 0.674027 w_k1_DIS_s005
                                 + 0.861465 w_k1_FDX_s005
                                 + 0.891429 w_k1_IBM_s005
                                 + 1.07226 w_k1_INTC_s005
                                 + 0.851304 w_k1_KO_s005
                                 + 0.894854 w_k1_MAT_s005
                                 + 1.04095 w_k1_MSFT_s005
                                 + 0.89393 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.0098 w_k1_ADM_s005
                                 + 0.826029 w_k1_AXP_s005
                                 + 0.904752 w_k1_DIS_s005
                                 + 0.85175 w_k1_FDX_s005
                                 + 1.12157 w_k1_IBM_s005
                                 + 1.21109 w_k1_INTC_s005
                                 + 1.32553 w_k1_KO_s005
                                 + 0.897843 w_k1_MAT_s005
                                 + 1.09352 w_k1_MSFT_s005
                                 + 1.12778 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.24157 w_k1_ADM_s005
                                 + 1.41247 w_k1_AXP_s005
                                 + 1.07319 w_k1_DIS_s005
                                 + 1.11651 w_k1_FDX_s005
                                 + 1.29053 w_k1_IBM_s005
                                 + 1.47741 w_k1_INTC_s005
                                 + 1.19935 w_k1_KO_s005 + 1.24968 w_k1_MAT_s005
                                 + 1.04782 w_k1_MSFT_s005 + 1.1821 w_k1_PG_s005
                                 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.04571 w_k1_ADM_s005
                                 + 0.739852 w_k1_AXP_s005
                                 + 1.20564 w_k1_DIS_s005
                                 + 0.960407 w_k1_FDX_s005
                                 + 0.927832 w_k1_IBM_s005
                                 + 1.01079 w_k1_INTC_s005
                                 + 1.17411 w_k1_KO_s005
                                 + 0.617503 w_k1_MAT_s005
                                 + 1.16185 w_k1_MSFT_s005 + 1.0035 w_k1_PG_s005
                                 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.863784 w_k1_ADM_s005
                                 + 0.65522 w_k1_AXP_s005
                                 + 0.57963 w_k1_DIS_s005
                                 + 0.712048 w_k1_FDX_s005
                                 + 0.851594 w_k1_IBM_s005
                                 + 0.762686 w_k1_INTC_s005
                                 + 0.860902 w_k1_KO_s005
                                 + 0.848601 w_k1_MAT_s005
                                 + 0.775957 w_k1_MSFT_s005
                                 + 0.7923 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.959301 w_k1_ADM_s005
                                 + 1.12745 w_k1_AXP_s005
                                 + 1.56944 w_k1_DIS_s005
                                 + 1.18511 w_k1_FDX_s005
                                 + 1.19815 w_k1_IBM_s005
                                 + 1.44254 w_k1_INTC_s005
                                 + 1.56986 w_k1_KO_s005 + 1.58345 w_k1_MAT_s005
                                 + 1.50407 w_k1_MSFT_s005
                                 + 1.13976 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.1518 w_k1_ADM_s005
                                 + 1.08457 w_k1_AXP_s005
                                 + 0.90575 w_k1_DIS_s005
                                 + 1.02461 w_k1_FDX_s005
                                 + 0.962634 w_k1_IBM_s005
                                 + 1.07618 w_k1_INTC_s005
                                 + 1.06158 w_k1_KO_s005 + 1.19227 w_k1_MAT_s005
                                 + 1.03577 w_k1_MSFT_s005
                                 + 1.14631 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.866603 w_k1_ADM_s005
                                 + 0.660785 w_k1_AXP_s005
                                 + 0.803083 w_k1_DIS_s005
                                 + 0.893672 w_k1_FDX_s005
                                 + 0.750304 w_k1_IBM_s005
                                 + 1.0893 w_k1_INTC_s005
                                 + 0.839196 w_k1_KO_s005
                                 + 0.580301 w_k1_MAT_s005
                                 + 0.948896 w_k1_MSFT_s005
                                 + 0.830119 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.09358 w_k1_ADM_s006
                                 + 1.79331 w_k1_AXP_s006
                                 + 1.99439 w_k1_DIS_s006
                                 + 1.49666 w_k1_FDX_s006
                                 + 1.47182 w_k1_IBM_s006
                                 + 1.4096 w_k1_INTC_s006 + 1.72175 w_k1_KO_s006
                                 + 3.02187 w_k1_MAT_s006
                                 + 1.42169 w_k1_MSFT_s006
                                 + 1.49977 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.03897 w_k1_ADM_s006
                                 + 0.91598 w_k1_AXP_s006
                                 + 0.765695 w_k1_DIS_s006
                                 + 0.971448 w_k1_FDX_s006
                                 + 1.04198 w_k1_IBM_s006
                                 + 1.05282 w_k1_INTC_s006
                                 + 0.993973 w_k1_KO_s006
                                 + 1.29223 w_k1_MAT_s006
                                 + 0.907293 w_k1_MSFT_s006
                                 + 0.974975 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.22791 w_k1_ADM_s006
                                 + 1.09527 w_k1_AXP_s006
                                 + 0.906641 w_k1_DIS_s006
                                 + 1.07933 w_k1_FDX_s006
                                 + 0.687697 w_k1_IBM_s006
                                 + 1.06682 w_k1_INTC_s006
                                 + 0.953428 w_k1_KO_s006
                                 + 1.14349 w_k1_MAT_s006
                                 + 0.95911 w_k1_MSFT_s006
                                 + 0.97103 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.27832 w_k1_ADM_s006
                                 + 1.36036 w_k1_AXP_s006
                                 + 1.32028 w_k1_DIS_s006
                                 + 1.30866 w_k1_FDX_s006
                                 + 1.05444 w_k1_IBM_s006
                                 + 1.30334 w_k1_INTC_s006
                                 + 1.02339 w_k1_KO_s006 + 1.34953 w_k1_MAT_s006
                                 + 1.09051 w_k1_MSFT_s006
                                 + 1.06073 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.06958 w_k1_ADM_s006
                                 + 1.13317 w_k1_AXP_s006
                                 + 1.13028 w_k1_DIS_s006
                                 + 1.03596 w_k1_FDX_s006
                                 + 1.63832 w_k1_IBM_s006
                                 + 1.08665 w_k1_INTC_s006
                                 + 1.17487 w_k1_KO_s006 + 1.99187 w_k1_MAT_s006
                                 + 1.09284 w_k1_MSFT_s006 + 1.0816 w_k1_PG_s006
                                 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.831942 w_k1_ADM_s006
                                 + 0.650269 w_k1_AXP_s006
                                 + 0.467084 w_k1_DIS_s006
                                 + 0.695403 w_k1_FDX_s006
                                 + 0.634321 w_k1_IBM_s006
                                 + 0.807369 w_k1_INTC_s006
                                 + 0.587415 w_k1_KO_s006
                                 + 0.805524 w_k1_MAT_s006
                                 + 0.627235 w_k1_MSFT_s006
                                 + 0.768045 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.672273 w_k1_ADM_s006
                                 + 0.792292 w_k1_AXP_s006
                                 + 0.70068 w_k1_DIS_s006
                                 + 0.816103 w_k1_FDX_s006
                                 + 0.962554 w_k1_IBM_s006
                                 + 0.742475 w_k1_INTC_s006
                                 + 0.912821 w_k1_KO_s006 + 1.3841 w_k1_MAT_s006
                                 + 0.791499 w_k1_MSFT_s006
                                 + 0.911176 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.870099 w_k1_ADM_s006
                                 + 1.14955 w_k1_AXP_s006
                                 + 1.24571 w_k1_DIS_s006
                                 + 1.02628 w_k1_FDX_s006
                                 + 1.03061 w_k1_IBM_s006
                                 + 1.03939 w_k1_INTC_s006
                                 + 1.07334 w_k1_KO_s006 + 1.3532 w_k1_MAT_s006
                                 + 1.00045 w_k1_MSFT_s006
                                 + 1.05581 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.772452 w_k1_ADM_s007
                                 + 0.621556 w_k1_AXP_s007
                                 + 0.703219 w_k1_DIS_s007
                                 + 0.779528 w_k1_FDX_s007
                                 + 0.693525 w_k1_IBM_s007
                                 + 0.939053 w_k1_INTC_s007
                                 + 0.807816 w_k1_KO_s007
                                 + 0.767337 w_k1_MAT_s007
                                 + 0.901508 w_k1_MSFT_s007
                                 + 0.806029 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.789066 w_k1_ADM_s007
                                 + 0.840348 w_k1_AXP_s007
                                 + 0.691602 w_k1_DIS_s007
                                 + 0.824535 w_k1_FDX_s007
                                 + 1.13949 w_k1_IBM_s007
                                 + 0.749631 w_k1_INTC_s007
                                 + 0.899333 w_k1_KO_s007
                                 + 1.62206 w_k1_MAT_s007
                                 + 0.798207 w_k1_MSFT_s007
                                 + 0.925486 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.05276 w_k1_ADM_s007
                                 + 1.39999 w_k1_AXP_s007
                                 + 1.64085 w_k1_DIS_s007
                                 + 1.33498 w_k1_FDX_s007
                                 + 1.26557 w_k1_IBM_s007
                                 + 1.09193 w_k1_INTC_s007
                                 + 1.39883 w_k1_KO_s007 + 1.63588 w_k1_MAT_s007
                                 + 1.2432 w_k1_MSFT_s007 + 1.11038 w_k1_PG_s007
                                 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.15104 w_k1_ADM_s007
                                 + 1.49416 w_k1_AXP_s007
                                 + 1.29265 w_k1_DIS_s007 + 1.4439 w_k1_FDX_s007
                                 + 0.83773 w_k1_IBM_s007
                                 + 1.43281 w_k1_INTC_s007
                                 + 0.993731 w_k1_KO_s007
                                 + 1.38537 w_k1_MAT_s007
                                 + 1.35624 w_k1_MSFT_s007
                                 + 1.02478 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.763627 w_k1_ADM_s007
                                 + 1.08612 w_k1_AXP_s007
                                 + 1.09783 w_k1_DIS_s007 + 1.1095 w_k1_FDX_s007
                                 + 1.16658 w_k1_IBM_s007
                                 + 1.18383 w_k1_INTC_s007
                                 + 1.26707 w_k1_KO_s007 + 1.81776 w_k1_MAT_s007
                                 + 1.28688 w_k1_MSFT_s007
                                 + 1.09017 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.866005 w_k1_ADM_s007
                                 + 0.895231 w_k1_AXP_s007
                                 + 0.854343 w_k1_DIS_s007
                                 + 1.00893 w_k1_FDX_s007
                                 + 0.834672 w_k1_IBM_s007
                                 + 1.11297 w_k1_INTC_s007
                                 + 0.935741 w_k1_KO_s007 + 1.2507 w_k1_MAT_s007
                                 + 1.03841 w_k1_MSFT_s007
                                 + 1.09243 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.964976 w_k1_ADM_s007
                                 + 1.14538 w_k1_AXP_s007
                                 + 0.98692 w_k1_DIS_s007
                                 + 0.981879 w_k1_FDX_s007
                                 + 1.85488 w_k1_IBM_s007
                                 + 1.06968 w_k1_INTC_s007
                                 + 1.26437 w_k1_KO_s007 + 2.31026 w_k1_MAT_s007
                                 + 1.04591 w_k1_MSFT_s007 + 0.9373 w_k1_PG_s007
                                 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.913852 w_k1_ADM_s007
                                 + 0.784999 w_k1_AXP_s007
                                 + 1.05642 w_k1_DIS_s007
                                 + 1.21899 w_k1_FDX_s007
                                 + 0.587601 w_k1_IBM_s007
                                 + 1.16304 w_k1_INTC_s007
                                 + 0.855296 w_k1_KO_s007
                                 + 0.440556 w_k1_MAT_s007
                                 + 1.26277 w_k1_MSFT_s007
                                 + 0.985702 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.955291 w_k1_ADM_s010
                                 + 1.18823 w_k1_AXP_s010 + 0.8844 w_k1_DIS_s010
                                 + 0.89285 w_k1_FDX_s010
                                 + 1.25198 w_k1_IBM_s010
                                 + 0.89981 w_k1_INTC_s010
                                 + 1.21655 w_k1_KO_s010 + 1.16561 w_k1_MAT_s010
                                 + 1.0226 w_k1_MSFT_s010 + 1.12358 w_k1_PG_s010
                                 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.31703 w_k1_ADM_s010
                                 + 1.59003 w_k1_AXP_s010
                                 + 1.74824 w_k1_DIS_s010
                                 + 1.33561 w_k1_FDX_s010
                                 + 1.34113 w_k1_IBM_s010
                                 + 1.4425 w_k1_INTC_s010 + 1.38691 w_k1_KO_s010
                                 + 0.924059 w_k1_MAT_s010
                                 + 1.47701 w_k1_MSFT_s010
                                 + 1.21417 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.661262 w_k1_ADM_s010
                                 + 0.774965 w_k1_AXP_s010
                                 + 0.686256 w_k1_DIS_s010
                                 + 0.745455 w_k1_FDX_s010
                                 + 0.928321 w_k1_IBM_s010
                                 + 0.789285 w_k1_INTC_s010
                                 + 0.880222 w_k1_KO_s010
                                 + 0.820025 w_k1_MAT_s010
                                 + 0.823048 w_k1_MSFT_s010
                                 + 0.945117 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.830355 w_k1_ADM_s010
                                 + 0.753109 w_k1_AXP_s010
                                 + 0.689608 w_k1_DIS_s010
                                 + 0.871365 w_k1_FDX_s010
                                 + 0.700815 w_k1_IBM_s010
                                 + 0.920997 w_k1_INTC_s010
                                 + 0.919205 w_k1_KO_s010
                                 + 0.865434 w_k1_MAT_s010
                                 + 0.96571 w_k1_MSFT_s010
                                 + 0.935352 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.06731 w_k1_ADM_s010
                                 + 1.03374 w_k1_AXP_s010
                                 + 1.23267 w_k1_DIS_s010
                                 + 1.02521 w_k1_FDX_s010
                                 + 0.964194 w_k1_IBM_s010
                                 + 1.08472 w_k1_INTC_s010
                                 + 1.27137 w_k1_KO_s010
                                 + 0.869953 w_k1_MAT_s010
                                 + 1.30531 w_k1_MSFT_s010
                                 + 1.05321 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.22053 w_k1_ADM_s010
                                 + 1.14461 w_k1_AXP_s010
                                 + 0.823771 w_k1_DIS_s010
                                 + 1.01226 w_k1_FDX_s010
                                 + 1.00244 w_k1_IBM_s010
                                 + 1.03377 w_k1_INTC_s010
                                 + 0.895242 w_k1_KO_s010
                                 + 0.841224 w_k1_MAT_s010
                                 + 1.0322 w_k1_MSFT_s010
                                 + 0.946524 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.2211 w_k1_ADM_s010
                                 + 1.37637 w_k1_AXP_s010
                                 + 1.07503 w_k1_DIS_s010
                                 + 1.27137 w_k1_FDX_s010
                                 + 1.13995 w_k1_IBM_s010
                                 + 1.4276 w_k1_INTC_s010 + 1.24576 w_k1_KO_s010
                                 + 1.03522 w_k1_MAT_s010
                                 + 1.36153 w_k1_MSFT_s010
                                 + 1.05277 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.06864 w_k1_ADM_s010
                                 + 1.38263 w_k1_AXP_s010 + 1.2826 w_k1_DIS_s010
                                 + 1.06492 w_k1_FDX_s010
                                 + 1.60108 w_k1_IBM_s010
                                 + 1.15992 w_k1_INTC_s010
                                 + 1.34645 w_k1_KO_s010 + 1.60489 w_k1_MAT_s010
                                 + 1.08998 w_k1_MSFT_s010
                                 + 1.07405 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.819434 w_k1_ADM_s011
                                 + 1.00327 w_k1_AXP_s011 + 1.2421 w_k1_DIS_s011
                                 + 1.19578 w_k1_FDX_s011
                                 + 0.690483 w_k1_IBM_s011
                                 + 1.38245 w_k1_INTC_s011
                                 + 1.20673 w_k1_KO_s011 + 0.75323 w_k1_MAT_s011
                                 + 1.34027 w_k1_MSFT_s011
                                 + 1.14358 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.786657 w_k1_ADM_s011
                                 + 0.724373 w_k1_AXP_s011
                                 + 0.803777 w_k1_DIS_s011
                                 + 0.821731 w_k1_FDX_s011
                                 + 0.886021 w_k1_IBM_s011
                                 + 0.978887 w_k1_INTC_s011
                                 + 0.893334 w_k1_KO_s011
                                 + 0.509601 w_k1_MAT_s011
                                 + 0.881061 w_k1_MSFT_s011
                                 + 0.995346 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.12408 w_k1_ADM_s011
                                 + 1.8185 w_k1_AXP_s011 + 1.34429 w_k1_DIS_s011
                                 + 1.07939 w_k1_FDX_s011
                                 + 1.63093 w_k1_IBM_s011
                                 + 0.998231 w_k1_INTC_s011
                                 + 1.31499 w_k1_KO_s011 + 1.4991 w_k1_MAT_s011
                                 + 0.987333 w_k1_MSFT_s011
                                 + 1.04088 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.719061 w_k1_ADM_s011
                                 + 1.01115 w_k1_AXP_s011
                                 + 0.635733 w_k1_DIS_s011
                                 + 0.735836 w_k1_FDX_s011
                                 + 1.11858 w_k1_IBM_s011
                                 + 0.786075 w_k1_INTC_s011
                                 + 0.928855 w_k1_KO_s011
                                 + 0.996014 w_k1_MAT_s011
                                 + 0.702369 w_k1_MSFT_s011
                                 + 0.917502 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.703604 w_k1_ADM_s011
                                 + 0.911212 w_k1_AXP_s011
                                 + 0.706631 w_k1_DIS_s011
                                 + 0.875576 w_k1_FDX_s011
                                 + 0.566744 w_k1_IBM_s011
                                 + 0.810529 w_k1_INTC_s011
                                 + 0.66535 w_k1_KO_s011
                                 + 0.389665 w_k1_MAT_s011
                                 + 0.804854 w_k1_MSFT_s011
                                 + 0.808141 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.01883 w_k1_ADM_s011
                                 + 1.09041 w_k1_AXP_s011
                                 + 0.939259 w_k1_DIS_s011
                                 + 0.930743 w_k1_FDX_s011 + 1.091 w_k1_IBM_s011
                                 + 1.13516 w_k1_INTC_s011
                                 + 1.11233 w_k1_KO_s011
                                 + 0.868474 w_k1_MAT_s011
                                 + 0.903273 w_k1_MSFT_s011
                                 + 0.956133 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.21469 w_k1_ADM_s011
                                 + 1.42253 w_k1_AXP_s011
                                 + 1.20529 w_k1_DIS_s011
                                 + 1.22078 w_k1_FDX_s011
                                 + 0.890071 w_k1_IBM_s011
                                 + 1.33381 w_k1_INTC_s011
                                 + 1.05277 w_k1_KO_s011
                                 + 0.734818 w_k1_MAT_s011
                                 + 1.03459 w_k1_MSFT_s011 + 1.0349 w_k1_PG_s011
                                 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.96492 w_k1_ADM_s011
                                 + 1.06187 w_k1_AXP_s011
                                 + 1.28273 w_k1_DIS_s011
                                 + 1.05155 w_k1_FDX_s011
                                 + 1.33618 w_k1_IBM_s011
                                 + 1.26246 w_k1_INTC_s011
                                 + 1.41397 w_k1_KO_s011 + 1.1014 w_k1_MAT_s011
                                 + 1.31223 w_k1_MSFT_s011 + 1.2082 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.762323 w_k1_ADM_s012
                                 + 0.830031 w_k1_AXP_s012
                                 + 0.768146 w_k1_DIS_s012
                                 + 0.858781 w_k1_FDX_s012
                                 + 0.733809 w_k1_IBM_s012
                                 + 1.09549 w_k1_INTC_s012
                                 + 0.826213 w_k1_KO_s012
                                 + 0.659786 w_k1_MAT_s012
                                 + 0.961829 w_k1_MSFT_s012
                                 + 0.867648 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.814493 w_k1_ADM_s012
                                 + 0.831007 w_k1_AXP_s012
                                 + 0.80175 w_k1_DIS_s012
                                 + 0.799635 w_k1_FDX_s012
                                 + 1.11814 w_k1_IBM_s012
                                 + 0.877178 w_k1_INTC_s012
                                 + 1.02471 w_k1_KO_s012 + 1.29222 w_k1_MAT_s012
                                 + 0.869095 w_k1_MSFT_s012
                                 + 0.892685 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.894491 w_k1_ADM_s012
                                 + 0.797353 w_k1_AXP_s012
                                 + 1.16159 w_k1_DIS_s012
                                 + 0.978557 w_k1_FDX_s012
                                 + 0.821477 w_k1_IBM_s012
                                 + 0.963646 w_k1_INTC_s012
                                 + 1.02898 w_k1_KO_s012
                                 + 0.616247 w_k1_MAT_s012
                                 + 0.954197 w_k1_MSFT_s012
                                 + 1.00129 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.752204 w_k1_ADM_s012
                                 + 1.05931 w_k1_AXP_s012
                                 + 1.03334 w_k1_DIS_s012
                                 + 0.916713 w_k1_FDX_s012
                                 + 1.60374 w_k1_IBM_s012
                                 + 0.848462 w_k1_INTC_s012
                                 + 1.36341 w_k1_KO_s012 + 2.05839 w_k1_MAT_s012
                                 + 0.958383 w_k1_MSFT_s012
                                 + 0.976487 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.06514 w_k1_ADM_s012
                                 + 1.02571 w_k1_AXP_s012 + 1.0525 w_k1_DIS_s012
                                 + 1.08228 w_k1_FDX_s012
                                 + 0.987118 w_k1_IBM_s012
                                 + 1.12774 w_k1_INTC_s012
                                 + 1.08027 w_k1_KO_s012
                                 + 0.750894 w_k1_MAT_s012
                                 + 0.990132 w_k1_MSFT_s012
                                 + 0.979645 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.999723 w_k1_ADM_s012
                                 + 1.28209 w_k1_AXP_s012
                                 + 1.42436 w_k1_DIS_s012
                                 + 1.30478 w_k1_FDX_s012
                                 + 0.84847 w_k1_IBM_s012
                                 + 1.07397 w_k1_INTC_s012
                                 + 1.07656 w_k1_KO_s012 + 1.09577 w_k1_MAT_s012
                                 + 1.08386 w_k1_MSFT_s012
                                 + 1.13637 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.821228 w_k1_ADM_s012
                                 + 1.20314 w_k1_AXP_s012
                                 + 1.04233 w_k1_DIS_s012
                                 + 1.01198 w_k1_FDX_s012
                                 + 1.24051 w_k1_IBM_s012
                                 + 0.916878 w_k1_INTC_s012
                                 + 1.03936 w_k1_KO_s012 + 1.53945 w_k1_MAT_s012
                                 + 0.983917 w_k1_MSFT_s012
                                 + 1.07819 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.25059 w_k1_ADM_s012
                                 + 1.29208 w_k1_AXP_s012
                                 + 1.45163 w_k1_DIS_s012
                                 + 1.20441 w_k1_FDX_s012
                                 + 1.45307 w_k1_IBM_s012
                                 + 1.22201 w_k1_INTC_s012
                                 + 1.42628 w_k1_KO_s012 + 1.63707 w_k1_MAT_s012
                                 + 1.19844 w_k1_MSFT_s012
                                 + 1.06232 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.09207 w_k1_ADM_s013
                                 + 1.31798 w_k1_AXP_s013
                                 + 1.16796 w_k1_DIS_s013
                                 + 1.16799 w_k1_FDX_s013
                                 + 0.775423 w_k1_IBM_s013
                                 + 1.12678 w_k1_INTC_s013
                                 + 1.03718 w_k1_KO_s013
                                 + 0.770835 w_k1_MAT_s013
                                 + 1.03177 w_k1_MSFT_s013
                                 + 1.16794 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.15721 w_k1_ADM_s013
                                 + 1.40219 w_k1_AXP_s013
                                 + 1.69602 w_k1_DIS_s013
                                 + 1.34789 w_k1_FDX_s013
                                 + 1.03186 w_k1_IBM_s013
                                 + 1.48015 w_k1_INTC_s013
                                 + 1.51442 w_k1_KO_s013 + 1.2223 w_k1_MAT_s013
                                 + 1.24751 w_k1_MSFT_s013
                                 + 1.36331 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.04521 w_k1_ADM_s013
                                 + 1.08792 w_k1_AXP_s013
                                 + 1.12889 w_k1_DIS_s013 + 1.0303 w_k1_FDX_s013
                                 + 1.29478 w_k1_IBM_s013
                                 + 1.10687 w_k1_INTC_s013
                                 + 1.44495 w_k1_KO_s013 + 1.19182 w_k1_MAT_s013
                                 + 1.06017 w_k1_MSFT_s013 + 1.2357 w_k1_PG_s013
                                 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.858872 w_k1_ADM_s013
                                 + 1.41149 w_k1_AXP_s013
                                 + 1.26557 w_k1_DIS_s013
                                 + 1.11315 w_k1_FDX_s013
                                 + 2.13172 w_k1_IBM_s013
                                 + 1.10777 w_k1_INTC_s013 + 1.7447 w_k1_KO_s013
                                 + 2.14789 w_k1_MAT_s013
                                 + 1.22751 w_k1_MSFT_s013
                                 + 1.24459 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.660417 w_k1_ADM_s013
                                 + 0.708511 w_k1_AXP_s013
                                 + 0.63735 w_k1_DIS_s013
                                 + 0.75417 w_k1_FDX_s013
                                 + 0.705796 w_k1_IBM_s013
                                 + 0.840895 w_k1_INTC_s013
                                 + 0.811013 w_k1_KO_s013
                                 + 0.776124 w_k1_MAT_s013
                                 + 0.834831 w_k1_MSFT_s013
                                 + 0.847102 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.642715 w_k1_ADM_s013
                                 + 1.05165 w_k1_AXP_s013
                                 + 0.755679 w_k1_DIS_s013
                                 + 0.808292 w_k1_FDX_s013
                                 + 1.13413 w_k1_IBM_s013
                                 + 0.974205 w_k1_INTC_s013
                                 + 1.05506 w_k1_KO_s013 + 1.25424 w_k1_MAT_s013
                                 + 0.838465 w_k1_MSFT_s013
                                 + 1.03209 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.799714 w_k1_ADM_s013
                                 + 0.872449 w_k1_AXP_s013
                                 + 0.570416 w_k1_DIS_s013
                                 + 0.733317 w_k1_FDX_s013
                                 + 0.997465 w_k1_IBM_s013
                                 + 0.720284 w_k1_INTC_s013
                                 + 0.749153 w_k1_KO_s013
                                 + 1.28836 w_k1_MAT_s013
                                 + 0.772824 w_k1_MSFT_s013
                                 + 0.96331 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.05694 w_k1_ADM_s013
                                 + 0.923676 w_k1_AXP_s013
                                 + 0.82549 w_k1_DIS_s013
                                 + 0.992358 w_k1_FDX_s013
                                 + 0.724216 w_k1_IBM_s013
                                 + 0.847578 w_k1_INTC_s013
                                 + 0.818668 w_k1_KO_s013
                                 + 0.615142 w_k1_MAT_s013
                                 + 0.920928 w_k1_MSFT_s013
                                 + 0.990761 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.05472 w_k1_ADM_s014
                                 + 0.764445 w_k1_AXP_s014
                                 + 1.41479 w_k1_DIS_s014
                                 + 1.06022 w_k1_FDX_s014
                                 + 0.872306 w_k1_IBM_s014
                                 + 1.06232 w_k1_INTC_s014
                                 + 1.13543 w_k1_KO_s014 + 0.9101 w_k1_MAT_s014
                                 + 1.16234 w_k1_MSFT_s014
                                 + 1.06899 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.873917 w_k1_ADM_s014
                                 + 0.77898 w_k1_AXP_s014
                                 + 0.897001 w_k1_DIS_s014
                                 + 0.872257 w_k1_FDX_s014
                                 + 0.661874 w_k1_IBM_s014
                                 + 1.09735 w_k1_INTC_s014
                                 + 0.836508 w_k1_KO_s014
                                 + 0.566206 w_k1_MAT_s014
                                 + 0.908834 w_k1_MSFT_s014
                                 + 0.885901 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.817669 w_k1_ADM_s014
                                 + 0.835675 w_k1_AXP_s014
                                 + 0.923818 w_k1_DIS_s014
                                 + 0.965604 w_k1_FDX_s014
                                 + 0.942067 w_k1_IBM_s014
                                 + 0.908173 w_k1_INTC_s014
                                 + 1.01753 w_k1_KO_s014 + 1.04065 w_k1_MAT_s014
                                 + 0.913765 w_k1_MSFT_s014
                                 + 1.01679 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.2727 w_k1_ADM_s014
                                 + 1.12447 w_k1_AXP_s014 + 1.5797 w_k1_DIS_s014
                                 + 1.22837 w_k1_FDX_s014
                                 + 1.02475 w_k1_IBM_s014
                                 + 1.2036 w_k1_INTC_s014 + 1.42421 w_k1_KO_s014
                                 + 1.15716 w_k1_MAT_s014
                                 + 1.15403 w_k1_MSFT_s014
                                 + 1.21831 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 1.02626 w_k1_ADM_s014
                                 + 1.52354 w_k1_AXP_s014
                                 + 1.84563 w_k1_DIS_s014 + 1.3706 w_k1_FDX_s014
                                 + 1.36444 w_k1_IBM_s014
                                 + 1.3753 w_k1_INTC_s014 + 1.53504 w_k1_KO_s014
                                 + 1.36611 w_k1_MAT_s014
                                 + 1.24088 w_k1_MSFT_s014 + 1.3173 w_k1_PG_s014
                                 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.625645 w_k1_ADM_s014
                                 + 0.575024 w_k1_AXP_s014
                                 + 0.748623 w_k1_DIS_s014
                                 + 0.661692 w_k1_FDX_s014
                                 + 1.00065 w_k1_IBM_s014
                                 + 0.78987 w_k1_INTC_s014
                                 + 0.950692 w_k1_KO_s014
                                 + 0.877725 w_k1_MAT_s014
                                 + 0.754977 w_k1_MSFT_s014
                                 + 0.827043 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.03652 w_k1_ADM_s014
                                 + 1.19335 w_k1_AXP_s014
                                 + 1.15762 w_k1_DIS_s014
                                 + 1.19941 w_k1_FDX_s014
                                 + 0.750367 w_k1_IBM_s014
                                 + 1.2079 w_k1_INTC_s014 + 0.9623 w_k1_KO_s014
                                 + 0.860438 w_k1_MAT_s014
                                 + 1.12697 w_k1_MSFT_s014
                                 + 1.04162 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.09289 w_k1_ADM_s014
                                 + 1.08499 w_k1_AXP_s014
                                 + 1.07763 w_k1_DIS_s014
                                 + 0.941503 w_k1_FDX_s014
                                 + 1.4655 w_k1_IBM_s014
                                 + 0.994844 w_k1_INTC_s014
                                 + 1.1699 w_k1_KO_s014 + 1.50814 w_k1_MAT_s014
                                 + 0.891695 w_k1_MSFT_s014
                                 + 1.08254 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.909593 w_k1_ADM_s015
                                 + 0.921507 w_k1_AXP_s015
                                 + 0.706966 w_k1_DIS_s015
                                 + 0.826777 w_k1_FDX_s015
                                 + 1.05511 w_k1_IBM_s015
                                 + 0.979074 w_k1_INTC_s015
                                 + 0.793593 w_k1_KO_s015
                                 + 0.847637 w_k1_MAT_s015
                                 + 1.02062 w_k1_MSFT_s015
                                 + 0.944378 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.21973 w_k1_ADM_s015
                                 + 1.34078 w_k1_AXP_s015
                                 + 1.90588 w_k1_DIS_s015
                                 + 1.38672 w_k1_FDX_s015
                                 + 1.19502 w_k1_IBM_s015
                                 + 1.3092 w_k1_INTC_s015 + 1.37432 w_k1_KO_s015
                                 + 0.898839 w_k1_MAT_s015
                                 + 1.50424 w_k1_MSFT_s015
                                 + 1.34035 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.25735 w_k1_ADM_s015
                                 + 1.68926 w_k1_AXP_s015
                                 + 1.28506 w_k1_DIS_s015
                                 + 1.16952 w_k1_FDX_s015
                                 + 1.39344 w_k1_IBM_s015
                                 + 1.09908 w_k1_INTC_s015
                                 + 1.08871 w_k1_KO_s015 + 1.66556 w_k1_MAT_s015
                                 + 1.14927 w_k1_MSFT_s015
                                 + 1.19366 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.764911 w_k1_ADM_s015
                                 + 0.815847 w_k1_AXP_s015
                                 + 0.764501 w_k1_DIS_s015
                                 + 0.760761 w_k1_FDX_s015
                                 + 1.37694 w_k1_IBM_s015
                                 + 0.735215 w_k1_INTC_s015
                                 + 1.09486 w_k1_KO_s015 + 1.19472 w_k1_MAT_s015
                                 + 0.867887 w_k1_MSFT_s015
                                 + 0.948048 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.777946 w_k1_ADM_s015
                                 + 0.798011 w_k1_AXP_s015
                                 + 0.515845 w_k1_DIS_s015
                                 + 0.722062 w_k1_FDX_s015
                                 + 0.791213 w_k1_IBM_s015
                                 + 0.656628 w_k1_INTC_s015
                                 + 0.696054 w_k1_KO_s015
                                 + 0.654769 w_k1_MAT_s015
                                 + 0.719963 w_k1_MSFT_s015
                                 + 0.903384 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.07043 w_k1_ADM_s015
                                 + 1.05976 w_k1_AXP_s015
                                 + 1.01678 w_k1_DIS_s015
                                 + 0.932658 w_k1_FDX_s015
                                 + 1.00185 w_k1_IBM_s015
                                 + 1.03062 w_k1_INTC_s015
                                 + 1.03088 w_k1_KO_s015
                                 + 0.695245 w_k1_MAT_s015
                                 + 0.961839 w_k1_MSFT_s015
                                 + 0.983524 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.14044 w_k1_ADM_s015
                                 + 1.29124 w_k1_AXP_s015
                                 + 1.18879 w_k1_DIS_s015
                                 + 1.14433 w_k1_FDX_s015
                                 + 1.12095 w_k1_IBM_s015
                                 + 0.837932 w_k1_INTC_s015
                                 + 1.12167 w_k1_KO_s015 + 1.08277 w_k1_MAT_s015
                                 + 1.05833 w_k1_MSFT_s015
                                 + 1.20282 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.993359 w_k1_ADM_s015
                                 + 1.21939 w_k1_AXP_s015
                                 + 0.899037 w_k1_DIS_s015
                                 + 1.0408 w_k1_FDX_s015
                                 + 0.762815 w_k1_IBM_s015
                                 + 1.0082 w_k1_INTC_s015
                                 + 0.796369 w_k1_KO_s015
                                 + 0.695344 w_k1_MAT_s015
                                 + 1.00187 w_k1_MSFT_s015
                                 + 1.14306 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.5971 w_k1_ADM_s016
                                 + 2.00647 w_k1_AXP_s016
                                 + 1.73211 w_k1_DIS_s016
                                 + 1.44227 w_k1_FDX_s016
                                 + 1.55206 w_k1_IBM_s016
                                 + 1.25068 w_k1_INTC_s016
                                 + 1.49353 w_k1_KO_s016 + 1.41321 w_k1_MAT_s016
                                 + 1.22753 w_k1_MSFT_s016
                                 + 1.31044 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.22115 w_k1_ADM_s016
                                 + 1.01786 w_k1_AXP_s016
                                 + 0.990239 w_k1_DIS_s016
                                 + 1.09197 w_k1_FDX_s016
                                 + 0.820858 w_k1_IBM_s016
                                 + 0.919727 w_k1_INTC_s016
                                 + 0.780383 w_k1_KO_s016
                                 + 0.863601 w_k1_MAT_s016
                                 + 0.964333 w_k1_MSFT_s016
                                 + 1.00757 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.73722 w_k1_ADM_s016
                                 + 1.13672 w_k1_AXP_s016
                                 + 1.68654 w_k1_DIS_s016
                                 + 1.28985 w_k1_FDX_s016
                                 + 1.06205 w_k1_IBM_s016
                                 + 1.25027 w_k1_INTC_s016
                                 + 1.25102 w_k1_KO_s016
                                 + 0.877983 w_k1_MAT_s016
                                 + 1.20925 w_k1_MSFT_s016
                                 + 1.06069 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.20345 w_k1_ADM_s016
                                 + 0.904409 w_k1_AXP_s016
                                 + 0.970326 w_k1_DIS_s016
                                 + 0.945116 w_k1_FDX_s016
                                 + 0.836617 w_k1_IBM_s016
                                 + 1.07669 w_k1_INTC_s016
                                 + 0.984647 w_k1_KO_s016
                                 + 0.772774 w_k1_MAT_s016
                                 + 0.944382 w_k1_MSFT_s016
                                 + 1.03087 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.02621 w_k1_ADM_s016
                                 + 0.774837 w_k1_AXP_s016
                                 + 0.590253 w_k1_DIS_s016
                                 + 0.735357 w_k1_FDX_s016
                                 + 0.845581 w_k1_IBM_s016
                                 + 0.919337 w_k1_INTC_s016
                                 + 0.780468 w_k1_KO_s016
                                 + 0.799549 w_k1_MAT_s016
                                 + 0.67296 w_k1_MSFT_s016
                                 + 0.811079 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.10467 w_k1_ADM_s016
                                 + 0.783151 w_k1_AXP_s016
                                 + 0.760401 w_k1_DIS_s016
                                 + 0.927509 w_k1_FDX_s016
                                 + 0.613825 w_k1_IBM_s016
                                 + 0.852771 w_k1_INTC_s016
                                 + 0.787559 w_k1_KO_s016
                                 + 0.691811 w_k1_MAT_s016
                                 + 0.953767 w_k1_MSFT_s016
                                 + 0.864316 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.45138 w_k1_ADM_s016
                                 + 1.3805 w_k1_AXP_s016 + 1.20065 w_k1_DIS_s016
                                 + 1.16856 w_k1_FDX_s016 + 1.1886 w_k1_IBM_s016
                                 + 1.19329 w_k1_INTC_s016
                                 + 1.05057 w_k1_KO_s016 + 1.1032 w_k1_MAT_s016
                                 + 1.05299 w_k1_MSFT_s016
                                 + 1.06275 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.04896 w_k1_ADM_s016
                                 + 1.16138 w_k1_AXP_s016
                                 + 1.12353 w_k1_DIS_s016
                                 + 0.970988 w_k1_FDX_s016
                                 + 1.53637 w_k1_IBM_s016
                                 + 0.912849 w_k1_INTC_s016
                                 + 1.13582 w_k1_KO_s016 + 1.69282 w_k1_MAT_s016
                                 + 0.955046 w_k1_MSFT_s016
                                 + 0.98472 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.994296 w_k1_ADM_s017
                                 + 1.21151 w_k1_AXP_s017
                                 + 1.16589 w_k1_DIS_s017
                                 + 1.11735 w_k1_FDX_s017
                                 + 1.29645 w_k1_IBM_s017
                                 + 1.18677 w_k1_INTC_s017
                                 + 1.35529 w_k1_KO_s017 + 1.46465 w_k1_MAT_s017
                                 + 1.14439 w_k1_MSFT_s017
                                 + 1.06845 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.811587 w_k1_ADM_s017
                                 + 0.864361 w_k1_AXP_s017
                                 + 0.823406 w_k1_DIS_s017
                                 + 0.873832 w_k1_FDX_s017
                                 + 0.763178 w_k1_IBM_s017
                                 + 0.870282 w_k1_INTC_s017
                                 + 0.884871 w_k1_KO_s017
                                 + 0.819371 w_k1_MAT_s017
                                 + 0.821704 w_k1_MSFT_s017
                                 + 0.870664 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.894666 w_k1_ADM_s017
                                 + 0.838977 w_k1_AXP_s017
                                 + 1.02283 w_k1_DIS_s017
                                 + 0.99538 w_k1_FDX_s017
                                 + 0.715075 w_k1_IBM_s017
                                 + 1.226 w_k1_INTC_s017 + 0.923698 w_k1_KO_s017
                                 + 0.64112 w_k1_MAT_s017
                                 + 1.00637 w_k1_MSFT_s017
                                 + 0.866082 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.804767 w_k1_ADM_s017
                                 + 0.885282 w_k1_AXP_s017
                                 + 0.875633 w_k1_DIS_s017
                                 + 0.770092 w_k1_FDX_s017
                                 + 1.28708 w_k1_IBM_s017
                                 + 0.875406 w_k1_INTC_s017
                                 + 1.17292 w_k1_KO_s017 + 1.51071 w_k1_MAT_s017
                                 + 0.851124 w_k1_MSFT_s017
                                 + 0.942832 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.865263 w_k1_ADM_s017
                                 + 0.880097 w_k1_AXP_s017
                                 + 0.871263 w_k1_DIS_s017
                                 + 0.852703 w_k1_FDX_s017
                                 + 0.984559 w_k1_IBM_s017
                                 + 1.18082 w_k1_INTC_s017 + 1.0286 w_k1_KO_s017
                                 + 0.931324 w_k1_MAT_s017
                                 + 0.836113 w_k1_MSFT_s017
                                 + 0.930089 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.05996 w_k1_ADM_s017
                                 + 1.03973 w_k1_AXP_s017
                                 + 1.17032 w_k1_DIS_s017
                                 + 1.20287 w_k1_FDX_s017
                                 + 0.621454 w_k1_IBM_s017
                                 + 1.16719 w_k1_INTC_s017
                                 + 0.980977 w_k1_KO_s017
                                 + 0.886001 w_k1_MAT_s017
                                 + 1.17853 w_k1_MSFT_s017
                                 + 0.933887 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.21194 w_k1_ADM_s017
                                 + 1.07484 w_k1_AXP_s017
                                 + 1.28158 w_k1_DIS_s017
                                 + 1.08566 w_k1_FDX_s017
                                 + 0.848686 w_k1_IBM_s017
                                 + 1.15112 w_k1_INTC_s017
                                 + 1.23677 w_k1_KO_s017
                                 + 0.963142 w_k1_MAT_s017
                                 + 1.16427 w_k1_MSFT_s017
                                 + 1.04087 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.1613 w_k1_ADM_s017
                                 + 1.37953 w_k1_AXP_s017
                                 + 1.49099 w_k1_DIS_s017
                                 + 1.18652 w_k1_FDX_s017
                                 + 1.79476 w_k1_IBM_s017
                                 + 1.15835 w_k1_INTC_s017
                                 + 1.71331 w_k1_KO_s017 + 1.84925 w_k1_MAT_s017
                                 + 1.2457 w_k1_MSFT_s017 + 1.14791 w_k1_PG_s017
                                 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.837333 w_k1_ADM_s100
                                 + 0.640281 w_k1_AXP_s100
                                 + 0.62287 w_k1_DIS_s100
                                 + 0.786473 w_k1_FDX_s100
                                 + 0.69791 w_k1_IBM_s100
                                 + 0.825308 w_k1_INTC_s100
                                 + 0.660977 w_k1_KO_s100 + 0.4109 w_k1_MAT_s100
                                 + 0.788924 w_k1_MSFT_s100
                                 + 0.833149 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.25432 w_k1_ADM_s100
                                 + 1.48859 w_k1_AXP_s100 + 1.56 w_k1_DIS_s100
                                 + 1.40431 w_k1_FDX_s100
                                 + 1.14798 w_k1_IBM_s100
                                 + 1.39059 w_k1_INTC_s100
                                 + 1.27083 w_k1_KO_s100 + 1.1571 w_k1_MAT_s100
                                 + 1.41494 w_k1_MSFT_s100 + 1.1751 w_k1_PG_s100
                                 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.883482 w_k1_ADM_s100
                                 + 1.20833 w_k1_AXP_s100
                                 + 0.868968 w_k1_DIS_s100
                                 + 0.955154 w_k1_FDX_s100
                                 + 1.34245 w_k1_IBM_s100
                                 + 1.00684 w_k1_INTC_s100
                                 + 1.11297 w_k1_KO_s100 + 1.15369 w_k1_MAT_s100
                                 + 0.893045 w_k1_MSFT_s100
                                 + 0.949532 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.0894 w_k1_ADM_s100
                                 + 0.925113 w_k1_AXP_s100
                                 + 0.670103 w_k1_DIS_s100
                                 + 0.901681 w_k1_FDX_s100
                                 + 0.850993 w_k1_IBM_s100
                                 + 1.10816 w_k1_INTC_s100
                                 + 0.967364 w_k1_KO_s100
                                 + 0.868768 w_k1_MAT_s100
                                 + 0.854223 w_k1_MSFT_s100
                                 + 0.935118 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.2079 w_k1_ADM_s100
                                 + 1.13289 w_k1_AXP_s100
                                 + 1.03393 w_k1_DIS_s100
                                 + 1.18504 w_k1_FDX_s100
                                 + 0.988182 w_k1_IBM_s100
                                 + 1.05231 w_k1_INTC_s100
                                 + 0.977252 w_k1_KO_s100
                                 + 1.51518 w_k1_MAT_s100
                                 + 1.10752 w_k1_MSFT_s100
                                 + 0.918677 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.30991 w_k1_ADM_s100
                                 + 1.43579 w_k1_AXP_s100
                                 + 1.37524 w_k1_DIS_s100
                                 + 1.24962 w_k1_FDX_s100
                                 + 1.88629 w_k1_IBM_s100
                                 + 1.12217 w_k1_INTC_s100
                                 + 1.39698 w_k1_KO_s100 + 1.63157 w_k1_MAT_s100
                                 + 1.12506 w_k1_MSFT_s100
                                 + 1.24178 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.16646 w_k1_ADM_s100
                                 + 0.793947 w_k1_AXP_s100
                                 + 1.06357 w_k1_DIS_s100 + 1.0012 w_k1_FDX_s100
                                 + 1.01958 w_k1_IBM_s100
                                 + 0.95673 w_k1_INTC_s100
                                 + 1.15825 w_k1_KO_s100
                                 + 0.763851 w_k1_MAT_s100
                                 + 1.03176 w_k1_MSFT_s100
                                 + 1.07499 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.11661 w_k1_ADM_s100
                                 + 0.963803 w_k1_AXP_s100
                                 + 1.08691 w_k1_DIS_s100
                                 + 0.919718 w_k1_FDX_s100
                                 + 1.63067 w_k1_IBM_s100
                                 + 0.912379 w_k1_INTC_s100
                                 + 1.2176 w_k1_KO_s100 + 1.25926 w_k1_MAT_s100
                                 + 0.856953 w_k1_MSFT_s100
                                 + 1.03717 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.10729 w_k1_ADM_s101
                                 + 1.24167 w_k1_AXP_s101
                                 + 0.850677 w_k1_DIS_s101
                                 + 0.988869 w_k1_FDX_s101
                                 + 0.76825 w_k1_IBM_s101
                                 + 0.916007 w_k1_INTC_s101
                                 + 0.731804 w_k1_KO_s101
                                 + 0.743555 w_k1_MAT_s101
                                 + 0.881547 w_k1_MSFT_s101
                                 + 0.964501 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.13307 w_k1_ADM_s101
                                 + 1.44733 w_k1_AXP_s101
                                 + 1.29845 w_k1_DIS_s101
                                 + 1.33479 w_k1_FDX_s101 + 0.9918 w_k1_IBM_s101
                                 + 1.13212 w_k1_INTC_s101
                                 + 1.08087 w_k1_KO_s101
                                 + 0.993207 w_k1_MAT_s101
                                 + 1.13374 w_k1_MSFT_s101
                                 + 1.07537 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.865047 w_k1_ADM_s101
                                 + 1.00089 w_k1_AXP_s101
                                 + 0.55441 w_k1_DIS_s101
                                 + 0.813266 w_k1_FDX_s101
                                 + 0.910971 w_k1_IBM_s101
                                 + 0.95221 w_k1_INTC_s101
                                 + 0.722479 w_k1_KO_s101
                                 + 0.791552 w_k1_MAT_s101
                                 + 0.752283 w_k1_MSFT_s101
                                 + 0.829749 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.959228 w_k1_ADM_s101
                                 + 0.792522 w_k1_AXP_s101
                                 + 0.865838 w_k1_DIS_s101
                                 + 0.915074 w_k1_FDX_s101
                                 + 0.809656 w_k1_IBM_s101
                                 + 1.02008 w_k1_INTC_s101
                                 + 0.797148 w_k1_KO_s101
                                 + 0.626506 w_k1_MAT_s101
                                 + 0.967437 w_k1_MSFT_s101
                                 + 0.823234 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.04076 w_k1_ADM_s101
                                 + 1.05697 w_k1_AXP_s101
                                 + 0.867935 w_k1_DIS_s101
                                 + 0.886238 w_k1_FDX_s101
                                 + 1.43853 w_k1_IBM_s101
                                 + 1.03158 w_k1_INTC_s101
                                 + 1.07099 w_k1_KO_s101 + 1.48038 w_k1_MAT_s101
                                 + 0.915038 w_k1_MSFT_s101
                                 + 0.934187 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.868382 w_k1_ADM_s101
                                 + 0.581487 w_k1_AXP_s101
                                 + 0.936942 w_k1_DIS_s101
                                 + 0.795394 w_k1_FDX_s101
                                 + 1.1991 w_k1_IBM_s101
                                 + 0.987571 w_k1_INTC_s101
                                 + 1.07713 w_k1_KO_s101
                                 + 0.503947 w_k1_MAT_s101
                                 + 0.921123 w_k1_MSFT_s101
                                 + 0.836654 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.24029 w_k1_ADM_s101
                                 + 1.14514 w_k1_AXP_s101
                                 + 1.13689 w_k1_DIS_s101
                                 + 1.09461 w_k1_FDX_s101
                                 + 1.21069 w_k1_IBM_s101
                                 + 1.08795 w_k1_INTC_s101
                                 + 1.10805 w_k1_KO_s101
                                 + 0.946724 w_k1_MAT_s101
                                 + 0.848378 w_k1_MSFT_s101
                                 + 0.995895 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.19863 w_k1_ADM_s101
                                 + 1.29723 w_k1_AXP_s101
                                 + 1.33079 w_k1_DIS_s101
                                 + 1.06659 w_k1_FDX_s101
                                 + 1.77333 w_k1_IBM_s101
                                 + 1.06674 w_k1_INTC_s101
                                 + 1.27326 w_k1_KO_s101 + 1.59135 w_k1_MAT_s101
                                 + 0.9481 w_k1_MSFT_s101 + 1.09222 w_k1_PG_s101
                                 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.03253 w_k1_ADM_s102
                                 + 0.809436 w_k1_AXP_s102
                                 + 0.966207 w_k1_DIS_s102
                                 + 0.96479 w_k1_FDX_s102
                                 + 0.805927 w_k1_IBM_s102
                                 + 1.18682 w_k1_INTC_s102
                                 + 0.967365 w_k1_KO_s102
                                 + 0.738668 w_k1_MAT_s102
                                 + 1.20533 w_k1_MSFT_s102
                                 + 0.899689 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.6578 w_k1_ADM_s102
                                 + 0.75912 w_k1_AXP_s102
                                 + 0.819422 w_k1_DIS_s102
                                 + 0.800304 w_k1_FDX_s102
                                 + 0.717155 w_k1_IBM_s102
                                 + 0.822356 w_k1_INTC_s102
                                 + 0.742648 w_k1_KO_s102
                                 + 0.808571 w_k1_MAT_s102
                                 + 0.938613 w_k1_MSFT_s102
                                 + 0.7895 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.840905 w_k1_ADM_s102
                                 + 0.874336 w_k1_AXP_s102
                                 + 1.07295 w_k1_DIS_s102
                                 + 0.872679 w_k1_FDX_s102
                                 + 1.30721 w_k1_IBM_s102
                                 + 0.979795 w_k1_INTC_s102
                                 + 1.20688 w_k1_KO_s102 + 1.07314 w_k1_MAT_s102
                                 + 1.07866 w_k1_MSFT_s102
                                 + 0.934619 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.22682 w_k1_ADM_s102
                                 + 1.21459 w_k1_AXP_s102
                                 + 1.20626 w_k1_DIS_s102 + 1.1599 w_k1_FDX_s102
                                 + 0.88262 w_k1_IBM_s102
                                 + 1.12926 w_k1_INTC_s102
                                 + 1.02621 w_k1_KO_s102
                                 + 0.966157 w_k1_MAT_s102
                                 + 1.18887 w_k1_MSFT_s102
                                 + 1.04582 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.12087 w_k1_ADM_s102
                                 + 0.877267 w_k1_AXP_s102
                                 + 1.09583 w_k1_DIS_s102
                                 + 1.01817 w_k1_FDX_s102
                                 + 1.02558 w_k1_IBM_s102
                                 + 1.12716 w_k1_INTC_s102
                                 + 1.16627 w_k1_KO_s102 + 1.02243 w_k1_MAT_s102
                                 + 1.22039 w_k1_MSFT_s102
                                 + 0.966802 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.95672 w_k1_ADM_s102
                                 + 0.810517 w_k1_AXP_s102
                                 + 0.803803 w_k1_DIS_s102
                                 + 0.847246 w_k1_FDX_s102
                                 + 0.917099 w_k1_IBM_s102
                                 + 0.900662 w_k1_INTC_s102
                                 + 0.924671 w_k1_KO_s102
                                 + 0.980736 w_k1_MAT_s102
                                 + 1.00519 w_k1_MSFT_s102
                                 + 0.891965 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.734533 w_k1_ADM_s102
                                 + 0.837882 w_k1_AXP_s102
                                 + 0.945008 w_k1_DIS_s102
                                 + 0.893048 w_k1_FDX_s102
                                 + 1.02886 w_k1_IBM_s102
                                 + 1.21784 w_k1_INTC_s102
                                 + 0.991932 w_k1_KO_s102
                                 + 0.798101 w_k1_MAT_s102
                                 + 1.04603 w_k1_MSFT_s102
                                 + 0.84475 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.32335 w_k1_ADM_s102
                                 + 1.55118 w_k1_AXP_s102
                                 + 1.53375 w_k1_DIS_s102
                                 + 1.28026 w_k1_FDX_s102
                                 + 1.35197 w_k1_IBM_s102
                                 + 1.27447 w_k1_INTC_s102
                                 + 1.40161 w_k1_KO_s102 + 1.5796 w_k1_MAT_s102
                                 + 1.34681 w_k1_MSFT_s102
                                 + 1.19476 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.2424 w_k1_ADM_s103
                                 + 1.16365 w_k1_AXP_s103
                                 + 0.85736 w_k1_DIS_s103
                                 + 1.01818 w_k1_FDX_s103
                                 + 0.838081 w_k1_IBM_s103
                                 + 0.843133 w_k1_INTC_s103
                                 + 0.847264 w_k1_KO_s103
                                 + 1.20471 w_k1_MAT_s103
                                 + 0.898598 w_k1_MSFT_s103
                                 + 0.948657 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.21387 w_k1_ADM_s103
                                 + 1.25882 w_k1_AXP_s103
                                 + 1.07168 w_k1_DIS_s103
                                 + 1.12667 w_k1_FDX_s103
                                 + 1.13564 w_k1_IBM_s103
                                 + 0.773688 w_k1_INTC_s103
                                 + 1.07029 w_k1_KO_s103 + 2.00165 w_k1_MAT_s103
                                 + 1.09257 w_k1_MSFT_s103
                                 + 0.978025 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.991054 w_k1_ADM_s103
                                 + 1.28211 w_k1_AXP_s103
                                 + 1.57237 w_k1_DIS_s103
                                 + 1.25249 w_k1_FDX_s103
                                 + 1.58094 w_k1_IBM_s103
                                 + 1.18586 w_k1_INTC_s103
                                 + 1.38579 w_k1_KO_s103 + 2.07339 w_k1_MAT_s103
                                 + 1.22523 w_k1_MSFT_s103
                                 + 1.04685 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.875354 w_k1_ADM_s103
                                 + 1.03572 w_k1_AXP_s103
                                 + 0.901848 w_k1_DIS_s103
                                 + 0.856038 w_k1_FDX_s103
                                 + 1.22488 w_k1_IBM_s103
                                 + 0.912181 w_k1_INTC_s103
                                 + 0.932117 w_k1_KO_s103
                                 + 1.11598 w_k1_MAT_s103
                                 + 0.900894 w_k1_MSFT_s103
                                 + 0.846707 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.04572 w_k1_ADM_s103
                                 + 0.990772 w_k1_AXP_s103
                                 + 1.14496 w_k1_DIS_s103
                                 + 1.02165 w_k1_FDX_s103
                                 + 0.832658 w_k1_IBM_s103
                                 + 0.906245 w_k1_INTC_s103
                                 + 1.04647 w_k1_KO_s103
                                 + 0.722587 w_k1_MAT_s103
                                 + 1.1102 w_k1_MSFT_s103
                                 + 0.918023 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.35748 w_k1_ADM_s103
                                 + 1.35932 w_k1_AXP_s103
                                 + 1.43596 w_k1_DIS_s103
                                 + 1.31545 w_k1_FDX_s103 + 1.0078 w_k1_IBM_s103
                                 + 1.17612 w_k1_INTC_s103
                                 + 1.15396 w_k1_KO_s103 + 1.28758 w_k1_MAT_s103
                                 + 1.33266 w_k1_MSFT_s103 + 1.0199 w_k1_PG_s103
                                 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.765259 w_k1_ADM_s103
                                 + 0.738502 w_k1_AXP_s103
                                 + 0.750119 w_k1_DIS_s103
                                 + 0.794256 w_k1_FDX_s103
                                 + 0.800221 w_k1_IBM_s103
                                 + 0.770401 w_k1_INTC_s103
                                 + 0.798501 w_k1_KO_s103
                                 + 0.894903 w_k1_MAT_s103
                                 + 0.882674 w_k1_MSFT_s103
                                 + 0.755343 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.03521 w_k1_ADM_s103
                                 + 0.951868 w_k1_AXP_s103
                                 + 1.35288 w_k1_DIS_s103
                                 + 1.04564 w_k1_FDX_s103
                                 + 1.03433 w_k1_IBM_s103
                                 + 0.920571 w_k1_INTC_s103
                                 + 1.24433 w_k1_KO_s103 + 1.52984 w_k1_MAT_s103
                                 + 1.14697 w_k1_MSFT_s103
                                 + 1.00634 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.023 w_k1_ADM_s104
                                 + 0.837668 w_k1_AXP_s104
                                 + 0.809687 w_k1_DIS_s104
                                 + 0.819936 w_k1_FDX_s104
                                 + 0.80672 w_k1_IBM_s104
                                 + 0.837667 w_k1_INTC_s104
                                 + 0.807015 w_k1_KO_s104
                                 + 0.647702 w_k1_MAT_s104
                                 + 0.703243 w_k1_MSFT_s104
                                 + 0.915513 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.32775 w_k1_ADM_s104
                                 + 1.32224 w_k1_AXP_s104
                                 + 1.16702 w_k1_DIS_s104
                                 + 1.08407 w_k1_FDX_s104
                                 + 1.57376 w_k1_IBM_s104
                                 + 1.04474 w_k1_INTC_s104
                                 + 1.28289 w_k1_KO_s104 + 1.69426 w_k1_MAT_s104
                                 + 0.927411 w_k1_MSFT_s104
                                 + 1.07163 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.52676 w_k1_ADM_s104
                                 + 1.35459 w_k1_AXP_s104
                                 + 1.52293 w_k1_DIS_s104
                                 + 1.35358 w_k1_FDX_s104
                                 + 1.05833 w_k1_IBM_s104
                                 + 1.00734 w_k1_INTC_s104
                                 + 1.23051 w_k1_KO_s104 + 1.09644 w_k1_MAT_s104
                                 + 1.1906 w_k1_MSFT_s104 + 1.10797 w_k1_PG_s104
                                 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.758126 w_k1_ADM_s104
                                 + 0.75398 w_k1_AXP_s104
                                 + 0.561561 w_k1_DIS_s104
                                 + 0.663949 w_k1_FDX_s104
                                 + 0.873271 w_k1_IBM_s104
                                 + 0.626383 w_k1_INTC_s104
                                 + 0.728259 w_k1_KO_s104
                                 + 0.87491 w_k1_MAT_s104
                                 + 0.695705 w_k1_MSFT_s104
                                 + 0.849673 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.845829 w_k1_ADM_s104
                                 + 0.751727 w_k1_AXP_s104
                                 + 1.07704 w_k1_DIS_s104
                                 + 0.911562 w_k1_FDX_s104
                                 + 0.965647 w_k1_IBM_s104
                                 + 0.861333 w_k1_INTC_s104
                                 + 1.11601 w_k1_KO_s104
                                 + 0.757727 w_k1_MAT_s104
                                 + 0.912267 w_k1_MSFT_s104
                                 + 0.944139 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.24619 w_k1_ADM_s104
                                 + 1.12431 w_k1_AXP_s104
                                 + 1.01475 w_k1_DIS_s104
                                 + 1.11837 w_k1_FDX_s104
                                 + 1.03023 w_k1_IBM_s104
                                 + 0.881365 w_k1_INTC_s104
                                 + 0.976661 w_k1_KO_s104 + 1.2144 w_k1_MAT_s104
                                 + 1.02257 w_k1_MSFT_s104 + 1.0359 w_k1_PG_s104
                                 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.11424 w_k1_ADM_s104
                                 + 0.982383 w_k1_AXP_s104
                                 + 0.737441 w_k1_DIS_s104
                                 + 1.00387 w_k1_FDX_s104
                                 + 0.62688 w_k1_IBM_s104
                                 + 0.821246 w_k1_INTC_s104
                                 + 0.791235 w_k1_KO_s104
                                 + 0.587378 w_k1_MAT_s104
                                 + 0.794027 w_k1_MSFT_s104
                                 + 0.859735 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.08042 w_k1_ADM_s104
                                 + 0.728007 w_k1_AXP_s104
                                 + 0.820993 w_k1_DIS_s104
                                 + 0.78957 w_k1_FDX_s104
                                 + 1.20321 w_k1_IBM_s104
                                 + 0.788784 w_k1_INTC_s104
                                 + 1.11753 w_k1_KO_s104
                                 + 0.982164 w_k1_MAT_s104
                                 + 0.815189 w_k1_MSFT_s104
                                 + 0.93907 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.992714 w_k1_ADM_s105
                                 + 0.843986 w_k1_AXP_s105
                                 + 0.992836 w_k1_DIS_s105
                                 + 0.96726 w_k1_FDX_s105
                                 + 1.11934 w_k1_IBM_s105
                                 + 0.712879 w_k1_INTC_s105
                                 + 0.983825 w_k1_KO_s105
                                 + 1.70373 w_k1_MAT_s105
                                 + 0.98182 w_k1_MSFT_s105
                                 + 1.03761 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.950002 w_k1_ADM_s105
                                 + 0.847162 w_k1_AXP_s105
                                 + 0.76069 w_k1_DIS_s105
                                 + 0.747567 w_k1_FDX_s105
                                 + 1.38198 w_k1_IBM_s105
                                 + 0.824022 w_k1_INTC_s105
                                 + 0.992799 w_k1_KO_s105
                                 + 1.60444 w_k1_MAT_s105
                                 + 0.806314 w_k1_MSFT_s105
                                 + 0.914185 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.39438 w_k1_ADM_s105
                                 + 1.35294 w_k1_AXP_s105
                                 + 1.39275 w_k1_DIS_s105
                                 + 1.35955 w_k1_FDX_s105
                                 + 0.817904 w_k1_IBM_s105
                                 + 1.1785 w_k1_INTC_s105 + 1.08757 w_k1_KO_s105
                                 + 1.10213 w_k1_MAT_s105
                                 + 1.22699 w_k1_MSFT_s105
                                 + 1.05202 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.17953 w_k1_ADM_s105
                                 + 0.958773 w_k1_AXP_s105
                                 + 1.29839 w_k1_DIS_s105
                                 + 1.18812 w_k1_FDX_s105
                                 + 0.805197 w_k1_IBM_s105
                                 + 0.881454 w_k1_INTC_s105
                                 + 0.963784 w_k1_KO_s105
                                 + 1.01278 w_k1_MAT_s105
                                 + 1.14431 w_k1_MSFT_s105
                                 + 0.997236 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.782436 w_k1_ADM_s105
                                 + 0.721609 w_k1_AXP_s105
                                 + 0.561643 w_k1_DIS_s105
                                 + 0.732631 w_k1_FDX_s105
                                 + 0.808602 w_k1_IBM_s105
                                 + 0.685554 w_k1_INTC_s105
                                 + 0.709412 w_k1_KO_s105
                                 + 1.17998 w_k1_MAT_s105
                                 + 0.808048 w_k1_MSFT_s105
                                 + 0.807313 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.34372 w_k1_ADM_s105
                                 + 1.33017 w_k1_AXP_s105
                                 + 1.70517 w_k1_DIS_s105
                                 + 1.36268 w_k1_FDX_s105 + 1.2567 w_k1_IBM_s105
                                 + 0.962569 w_k1_INTC_s105
                                 + 1.35444 w_k1_KO_s105 + 1.82309 w_k1_MAT_s105
                                 + 1.12299 w_k1_MSFT_s105
                                 + 1.17798 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.935721 w_k1_ADM_s105
                                 + 1.18874 w_k1_AXP_s105
                                 + 1.28886 w_k1_DIS_s105
                                 + 0.987333 w_k1_FDX_s105
                                 + 1.69993 w_k1_IBM_s105
                                 + 1.05808 w_k1_INTC_s105
                                 + 1.40426 w_k1_KO_s105 + 2.53863 w_k1_MAT_s105
                                 + 1.05257 w_k1_MSFT_s105
                                 + 1.07671 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.945649 w_k1_ADM_s105
                                 + 0.669168 w_k1_AXP_s105
                                 + 0.91263 w_k1_DIS_s105
                                 + 0.89123 w_k1_FDX_s105
                                 + 0.752273 w_k1_IBM_s105
                                 + 0.785454 w_k1_INTC_s105
                                 + 0.793217 w_k1_KO_s105
                                 + 0.860118 w_k1_MAT_s105
                                 + 0.945919 w_k1_MSFT_s105
                                 + 0.88813 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.81954 w_k1_ADM_s106
                                 + 0.530843 w_k1_AXP_s106
                                 + 0.517113 w_k1_DIS_s106
                                 + 0.725478 w_k1_FDX_s106
                                 + 0.53985 w_k1_IBM_s106
                                 + 0.722481 w_k1_INTC_s106
                                 + 0.610276 w_k1_KO_s106
                                 + 0.487004 w_k1_MAT_s106
                                 + 0.728145 w_k1_MSFT_s106
                                 + 0.768678 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.938961 w_k1_ADM_s106
                                 + 1.19266 w_k1_AXP_s106
                                 + 1.23454 w_k1_DIS_s106
                                 + 1.16377 w_k1_FDX_s106
                                 + 1.37484 w_k1_IBM_s106
                                 + 1.16144 w_k1_INTC_s106
                                 + 1.25641 w_k1_KO_s106 + 1.23434 w_k1_MAT_s106
                                 + 1.1013 w_k1_MSFT_s106 + 1.08748 w_k1_PG_s106
                                 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.14892 w_k1_ADM_s106
                                 + 1.44391 w_k1_AXP_s106
                                 + 1.68227 w_k1_DIS_s106
                                 + 1.48022 w_k1_FDX_s106 + 1.3268 w_k1_IBM_s106
                                 + 1.53393 w_k1_INTC_s106
                                 + 1.43544 w_k1_KO_s106 + 1.49886 w_k1_MAT_s106
                                 + 1.42384 w_k1_MSFT_s106 + 1.3662 w_k1_PG_s106
                                 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.12253 w_k1_ADM_s106
                                 + 1.29478 w_k1_AXP_s106
                                 + 1.07918 w_k1_DIS_s106
                                 + 1.18522 w_k1_FDX_s106
                                 + 0.869467 w_k1_IBM_s106
                                 + 0.968231 w_k1_INTC_s106
                                 + 1.04302 w_k1_KO_s106 + 1.06941 w_k1_MAT_s106
                                 + 1.1901 w_k1_MSFT_s106 + 1.02609 w_k1_PG_s106
                                 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.932413 w_k1_ADM_s106
                                 + 0.649251 w_k1_AXP_s106
                                 + 0.786854 w_k1_DIS_s106
                                 + 0.831744 w_k1_FDX_s106
                                 + 0.899149 w_k1_IBM_s106
                                 + 0.849798 w_k1_INTC_s106
                                 + 0.885207 w_k1_KO_s106
                                 + 0.747407 w_k1_MAT_s106
                                 + 0.909844 w_k1_MSFT_s106
                                 + 0.914199 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.02068 w_k1_ADM_s106
                                 + 0.883298 w_k1_AXP_s106
                                 + 0.84223 w_k1_DIS_s106
                                 + 0.917285 w_k1_FDX_s106
                                 + 1.38899 w_k1_IBM_s106
                                 + 0.77258 w_k1_INTC_s106
                                 + 1.16816 w_k1_KO_s106 + 1.72675 w_k1_MAT_s106
                                 + 0.932442 w_k1_MSFT_s106
                                 + 1.0009 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.16056 w_k1_ADM_s106
                                 + 0.819987 w_k1_AXP_s106
                                 + 1.25246 w_k1_DIS_s106
                                 + 1.09914 w_k1_FDX_s106
                                 + 0.974855 w_k1_IBM_s106
                                 + 1.03322 w_k1_INTC_s106
                                 + 0.974631 w_k1_KO_s106
                                 + 0.779564 w_k1_MAT_s106
                                 + 1.18005 w_k1_MSFT_s106
                                 + 0.981579 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.734203 w_k1_ADM_s106
                                 + 0.913583 w_k1_AXP_s106
                                 + 0.711061 w_k1_DIS_s106
                                 + 0.981308 w_k1_FDX_s106
                                 + 0.823243 w_k1_IBM_s106
                                 + 0.939051 w_k1_INTC_s106
                                 + 0.851508 w_k1_KO_s106
                                 + 0.947815 w_k1_MAT_s106
                                 + 1.05177 w_k1_MSFT_s106
                                 + 0.990051 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.10996 w_k1_ADM_s107
                                 + 0.905077 w_k1_AXP_s107
                                 + 0.950807 w_k1_DIS_s107
                                 + 0.972947 w_k1_FDX_s107
                                 + 1.22736 w_k1_IBM_s107
                                 + 0.954704 w_k1_INTC_s107
                                 + 1.30904 w_k1_KO_s107 + 1.19434 w_k1_MAT_s107
                                 + 1.06702 w_k1_MSFT_s107
                                 + 1.14315 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.741741 w_k1_ADM_s107
                                 + 0.623755 w_k1_AXP_s107
                                 + 0.669601 w_k1_DIS_s107
                                 + 0.661543 w_k1_FDX_s107
                                 + 0.851036 w_k1_IBM_s107
                                 + 0.717607 w_k1_INTC_s107
                                 + 0.862111 w_k1_KO_s107
                                 + 0.505319 w_k1_MAT_s107
                                 + 0.768989 w_k1_MSFT_s107
                                 + 0.82868 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.12507 w_k1_ADM_s107
                                 + 1.14714 w_k1_AXP_s107
                                 + 1.31698 w_k1_DIS_s107
                                 + 1.20507 w_k1_FDX_s107
                                 + 0.881356 w_k1_IBM_s107
                                 + 1.2229 w_k1_INTC_s107 + 1.10512 w_k1_KO_s107
                                 + 0.77062 w_k1_MAT_s107
                                 + 1.1426 w_k1_MSFT_s107 + 1.07554 w_k1_PG_s107
                                 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.802553 w_k1_ADM_s107
                                 + 0.65591 w_k1_AXP_s107
                                 + 0.669954 w_k1_DIS_s107
                                 + 0.830838 w_k1_FDX_s107
                                 + 0.690902 w_k1_IBM_s107
                                 + 0.771086 w_k1_INTC_s107
                                 + 0.842944 w_k1_KO_s107
                                 + 0.674167 w_k1_MAT_s107
                                 + 0.884835 w_k1_MSFT_s107
                                 + 0.869483 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.937221 w_k1_ADM_s107
                                 + 0.888042 w_k1_AXP_s107
                                 + 1.07426 w_k1_DIS_s107
                                 + 1.01131 w_k1_FDX_s107
                                 + 0.862288 w_k1_IBM_s107
                                 + 0.8868 w_k1_INTC_s107
                                 + 0.989555 w_k1_KO_s107
                                 + 0.755745 w_k1_MAT_s107
                                 + 1.12168 w_k1_MSFT_s107
                                 + 1.04563 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.913047 w_k1_ADM_s107
                                 + 0.975742 w_k1_AXP_s107
                                 + 0.764624 w_k1_DIS_s107
                                 + 0.899777 w_k1_FDX_s107
                                 + 1.05039 w_k1_IBM_s107
                                 + 0.884208 w_k1_INTC_s107
                                 + 0.946812 w_k1_KO_s107
                                 + 1.13033 w_k1_MAT_s107
                                 + 0.898647 w_k1_MSFT_s107
                                 + 0.980849 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.62963 w_k1_ADM_s107
                                 + 1.52365 w_k1_AXP_s107
                                 + 1.93326 w_k1_DIS_s107
                                 + 1.39124 w_k1_FDX_s107
                                 + 1.46433 w_k1_IBM_s107
                                 + 1.14149 w_k1_INTC_s107
                                 + 1.79321 w_k1_KO_s107 + 1.84957 w_k1_MAT_s107
                                 + 1.31753 w_k1_MSFT_s107
                                 + 1.30031 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.952001 w_k1_ADM_s107
                                 + 1.19886 w_k1_AXP_s107
                                 + 1.22245 w_k1_DIS_s107
                                 + 1.04332 w_k1_FDX_s107
                                 + 1.33343 w_k1_IBM_s107
                                 + 1.02926 w_k1_INTC_s107
                                 + 1.27188 w_k1_KO_s107 + 1.49799 w_k1_MAT_s107
                                 + 1.11382 w_k1_MSFT_s107 + 1.1176 w_k1_PG_s107
                                 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.668181 w_k1_ADM_s110
                                 + 0.902074 w_k1_AXP_s110
                                 + 0.985809 w_k1_DIS_s110
                                 + 0.851826 w_k1_FDX_s110
                                 + 1.31351 w_k1_IBM_s110
                                 + 1.09434 w_k1_INTC_s110
                                 + 1.07521 w_k1_KO_s110 + 1.53329 w_k1_MAT_s110
                                 + 1.05192 w_k1_MSFT_s110
                                 + 0.844747 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.868783 w_k1_ADM_s110
                                 + 0.683105 w_k1_AXP_s110
                                 + 0.790013 w_k1_DIS_s110
                                 + 0.806572 w_k1_FDX_s110
                                 + 0.622987 w_k1_IBM_s110
                                 + 0.809998 w_k1_INTC_s110
                                 + 0.727432 w_k1_KO_s110
                                 + 0.672513 w_k1_MAT_s110
                                 + 0.917663 w_k1_MSFT_s110
                                 + 0.862899 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.00226 w_k1_ADM_s110
                                 + 1.41869 w_k1_AXP_s110
                                 + 1.30329 w_k1_DIS_s110
                                 + 1.21102 w_k1_FDX_s110
                                 + 1.20601 w_k1_IBM_s110
                                 + 1.37008 w_k1_INTC_s110
                                 + 1.06052 w_k1_KO_s110 + 1.36857 w_k1_MAT_s110
                                 + 1.24366 w_k1_MSFT_s110
                                 + 0.919833 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.08681 w_k1_ADM_s110
                                 + 0.923755 w_k1_AXP_s110
                                 + 1.2056 w_k1_DIS_s110 + 1.07756 w_k1_FDX_s110
                                 + 0.820175 w_k1_IBM_s110
                                 + 1.12969 w_k1_INTC_s110
                                 + 0.898695 w_k1_KO_s110
                                 + 0.747632 w_k1_MAT_s110
                                 + 1.15901 w_k1_MSFT_s110
                                 + 0.815063 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.1031 w_k1_ADM_s110
                                 + 1.13045 w_k1_AXP_s110
                                 + 1.82199 w_k1_DIS_s110
                                 + 1.28565 w_k1_FDX_s110
                                 + 1.31366 w_k1_IBM_s110
                                 + 1.08047 w_k1_INTC_s110
                                 + 1.37644 w_k1_KO_s110 + 1.48949 w_k1_MAT_s110
                                 + 1.31697 w_k1_MSFT_s110
                                 + 1.26433 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.01708 w_k1_ADM_s110
                                 + 0.909335 w_k1_AXP_s110
                                 + 1.12096 w_k1_DIS_s110
                                 + 1.05833 w_k1_FDX_s110
                                 + 0.664631 w_k1_IBM_s110
                                 + 1.17533 w_k1_INTC_s110
                                 + 0.844525 w_k1_KO_s110
                                 + 0.556059 w_k1_MAT_s110
                                 + 1.26406 w_k1_MSFT_s110
                                 + 0.889809 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.69518 w_k1_ADM_s110
                                 + 0.686817 w_k1_AXP_s110
                                 + 0.618993 w_k1_DIS_s110
                                 + 0.703356 w_k1_FDX_s110
                                 + 0.734713 w_k1_IBM_s110
                                 + 0.734388 w_k1_INTC_s110
                                 + 0.677461 w_k1_KO_s110
                                 + 0.90406 w_k1_MAT_s110
                                 + 0.795454 w_k1_MSFT_s110
                                 + 0.727724 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.864051 w_k1_ADM_s110
                                 + 1.08777 w_k1_AXP_s110
                                 + 1.04489 w_k1_DIS_s110
                                 + 0.988723 w_k1_FDX_s110
                                 + 0.960388 w_k1_IBM_s110
                                 + 1.0049 w_k1_INTC_s110 + 0.87942 w_k1_KO_s110
                                 + 1.11221 w_k1_MAT_s110
                                 + 1.00969 w_k1_MSFT_s110
                                 + 0.871569 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.939254 w_k1_ADM_s111
                                 + 0.887263 w_k1_AXP_s111
                                 + 0.907764 w_k1_DIS_s111
                                 + 0.870926 w_k1_FDX_s111
                                 + 1.18383 w_k1_IBM_s111
                                 + 0.973651 w_k1_INTC_s111
                                 + 1.12647 w_k1_KO_s111
                                 + 0.983042 w_k1_MAT_s111
                                 + 0.931556 w_k1_MSFT_s111
                                 + 0.931993 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.752 w_k1_ADM_s111
                                 + 1.38502 w_k1_AXP_s111 + 1.6199 w_k1_DIS_s111
                                 + 1.48766 w_k1_FDX_s111
                                 + 1.08738 w_k1_IBM_s111
                                 + 1.55882 w_k1_INTC_s111
                                 + 1.53252 w_k1_KO_s111
                                 + 0.970164 w_k1_MAT_s111
                                 + 1.52391 w_k1_MSFT_s111
                                 + 1.27025 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.48782 w_k1_ADM_s111
                                 + 1.5388 w_k1_AXP_s111 + 1.40582 w_k1_DIS_s111
                                 + 1.19536 w_k1_FDX_s111
                                 + 1.67746 w_k1_IBM_s111
                                 + 1.13727 w_k1_INTC_s111
                                 + 1.53941 w_k1_KO_s111 + 1.65043 w_k1_MAT_s111
                                 + 1.22999 w_k1_MSFT_s111
                                 + 1.15591 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.49537 w_k1_ADM_s111
                                 + 0.932116 w_k1_AXP_s111
                                 + 1.1549 w_k1_DIS_s111 + 1.16816 w_k1_FDX_s111
                                 + 1.03957 w_k1_IBM_s111
                                 + 1.15251 w_k1_INTC_s111
                                 + 1.29316 w_k1_KO_s111
                                 + 0.822671 w_k1_MAT_s111
                                 + 1.39296 w_k1_MSFT_s111
                                 + 0.996292 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.29835 w_k1_ADM_s111
                                 + 1.0231 w_k1_AXP_s111
                                 + 0.888964 w_k1_DIS_s111
                                 + 1.02374 w_k1_FDX_s111
                                 + 0.754137 w_k1_IBM_s111
                                 + 1.0997 w_k1_INTC_s111 + 1.05866 w_k1_KO_s111
                                 + 0.711366 w_k1_MAT_s111
                                 + 1.16122 w_k1_MSFT_s111
                                 + 1.06085 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.969106 w_k1_ADM_s111
                                 + 0.737935 w_k1_AXP_s111
                                 + 0.612332 w_k1_DIS_s111
                                 + 0.825854 w_k1_FDX_s111
                                 + 0.651617 w_k1_IBM_s111
                                 + 0.895726 w_k1_INTC_s111
                                 + 0.779743 w_k1_KO_s111
                                 + 0.533462 w_k1_MAT_s111
                                 + 0.93667 w_k1_MSFT_s111
                                 + 0.77494 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.894724 w_k1_ADM_s111
                                 + 0.609722 w_k1_AXP_s111
                                 + 0.685197 w_k1_DIS_s111
                                 + 0.752633 w_k1_FDX_s111
                                 + 0.827832 w_k1_IBM_s111
                                 + 0.662067 w_k1_INTC_s111
                                 + 0.798263 w_k1_KO_s111
                                 + 0.608245 w_k1_MAT_s111
                                 + 0.836549 w_k1_MSFT_s111
                                 + 0.813011 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.05669 w_k1_ADM_s111
                                 + 1.03219 w_k1_AXP_s111
                                 + 1.25038 w_k1_DIS_s111
                                 + 1.21402 w_k1_FDX_s111
                                 + 0.798445 w_k1_IBM_s111
                                 + 1.1713 w_k1_INTC_s111
                                 + 0.898212 w_k1_KO_s111
                                 + 0.695977 w_k1_MAT_s111
                                 + 1.34973 w_k1_MSFT_s111
                                 + 0.984244 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.14096 w_k1_ADM_s112
                                 + 1.1108 w_k1_AXP_s112
                                 + 0.857299 w_k1_DIS_s112
                                 + 1.05673 w_k1_FDX_s112
                                 + 0.787653 w_k1_IBM_s112
                                 + 0.999862 w_k1_INTC_s112
                                 + 0.971332 w_k1_KO_s112
                                 + 0.997686 w_k1_MAT_s112
                                 + 1.01739 w_k1_MSFT_s112
                                 + 1.02405 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.879698 w_k1_ADM_s112
                                 + 0.893163 w_k1_AXP_s112
                                 + 1.01362 w_k1_DIS_s112
                                 + 1.14105 w_k1_FDX_s112
                                 + 0.66189 w_k1_IBM_s112
                                 + 1.12254 w_k1_INTC_s112
                                 + 0.796129 w_k1_KO_s112
                                 + 0.855475 w_k1_MAT_s112
                                 + 1.07898 w_k1_MSFT_s112
                                 + 0.920987 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.888718 w_k1_ADM_s112
                                 + 1.27983 w_k1_AXP_s112
                                 + 0.838032 w_k1_DIS_s112
                                 + 0.880955 w_k1_FDX_s112
                                 + 1.2669 w_k1_IBM_s112
                                 + 1.02844 w_k1_INTC_s112
                                 + 1.01479 w_k1_KO_s112 + 1.30012 w_k1_MAT_s112
                                 + 0.874301 w_k1_MSFT_s112
                                 + 1.07941 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.01703 w_k1_ADM_s112
                                 + 1.0762 w_k1_AXP_s112 + 1.21409 w_k1_DIS_s112
                                 + 1.02499 w_k1_FDX_s112
                                 + 1.40953 w_k1_IBM_s112
                                 + 1.15324 w_k1_INTC_s112
                                 + 1.32716 w_k1_KO_s112 + 1.58678 w_k1_MAT_s112
                                 + 1.0704 w_k1_MSFT_s112 + 1.17309 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.879276 w_k1_ADM_s112
                                 + 1.36154 w_k1_AXP_s112 + 1.405 w_k1_DIS_s112
                                 + 1.36774 w_k1_FDX_s112
                                 + 1.07293 w_k1_IBM_s112
                                 + 1.56282 w_k1_INTC_s112
                                 + 1.08582 w_k1_KO_s112 + 1.05 w_k1_MAT_s112
                                 + 1.30574 w_k1_MSFT_s112
                                 + 1.13117 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.911557 w_k1_ADM_s112
                                 + 0.790259 w_k1_AXP_s112
                                 + 0.638264 w_k1_DIS_s112
                                 + 0.885418 w_k1_FDX_s112
                                 + 0.651459 w_k1_IBM_s112
                                 + 1.02751 w_k1_INTC_s112
                                 + 0.74245 w_k1_KO_s112
                                 + 0.722492 w_k1_MAT_s112
                                 + 0.903422 w_k1_MSFT_s112
                                 + 0.812134 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.909422 w_k1_ADM_s112
                                 + 1.01049 w_k1_AXP_s112
                                 + 1.07206 w_k1_DIS_s112 + 0.9961 w_k1_FDX_s112
                                 + 1.10786 w_k1_IBM_s112
                                 + 0.925468 w_k1_INTC_s112
                                 + 1.00226 w_k1_KO_s112 + 1.06996 w_k1_MAT_s112
                                 + 1.07519 w_k1_MSFT_s112
                                 + 0.990892 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.764647 w_k1_ADM_s112
                                 + 0.693573 w_k1_AXP_s112
                                 + 0.78988 w_k1_DIS_s112
                                 + 0.725315 w_k1_FDX_s112
                                 + 0.995907 w_k1_IBM_s112
                                 + 0.856346 w_k1_INTC_s112
                                 + 1.01725 w_k1_KO_s112 + 0.74549 w_k1_MAT_s112
                                 + 0.800856 w_k1_MSFT_s112
                                 + 0.86192 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.706404 w_k1_ADM_s113
                                 + 0.917541 w_k1_AXP_s113
                                 + 0.836968 w_k1_DIS_s113
                                 + 0.909584 w_k1_FDX_s113
                                 + 0.91433 w_k1_IBM_s113
                                 + 0.915951 w_k1_INTC_s113
                                 + 0.835019 w_k1_KO_s113
                                 + 0.977574 w_k1_MAT_s113
                                 + 0.785655 w_k1_MSFT_s113
                                 + 0.878523 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.809889 w_k1_ADM_s113
                                 + 1.00886 w_k1_AXP_s113
                                 + 0.992668 w_k1_DIS_s113
                                 + 0.978183 w_k1_FDX_s113
                                 + 1.19222 w_k1_IBM_s113
                                 + 0.872687 w_k1_INTC_s113
                                 + 0.911692 w_k1_KO_s113
                                 + 1.17179 w_k1_MAT_s113
                                 + 0.857505 w_k1_MSFT_s113
                                 + 1.01172 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.14359 w_k1_ADM_s113
                                 + 1.77894 w_k1_AXP_s113
                                 + 1.94676 w_k1_DIS_s113
                                 + 1.46177 w_k1_FDX_s113 + 1.4983 w_k1_IBM_s113
                                 + 1.53548 w_k1_INTC_s113
                                 + 1.36558 w_k1_KO_s113 + 1.42722 w_k1_MAT_s113
                                 + 1.35749 w_k1_MSFT_s113
                                 + 1.18995 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.06993 w_k1_ADM_s113
                                 + 1.24396 w_k1_AXP_s113
                                 + 1.23531 w_k1_DIS_s113
                                 + 1.20003 w_k1_FDX_s113
                                 + 1.04738 w_k1_IBM_s113
                                 + 1.00183 w_k1_INTC_s113
                                 + 1.04594 w_k1_KO_s113 + 1.35781 w_k1_MAT_s113
                                 + 1.11786 w_k1_MSFT_s113
                                 + 0.964466 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.778978 w_k1_ADM_s113
                                 + 0.780405 w_k1_AXP_s113
                                 + 0.846602 w_k1_DIS_s113
                                 + 0.780251 w_k1_FDX_s113
                                 + 1.26172 w_k1_IBM_s113
                                 + 0.919051 w_k1_INTC_s113
                                 + 0.807412 w_k1_KO_s113
                                 + 1.01565 w_k1_MAT_s113
                                 + 0.785367 w_k1_MSFT_s113
                                 + 0.889426 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.78189 w_k1_ADM_s113
                                 + 0.888116 w_k1_AXP_s113
                                 + 0.559439 w_k1_DIS_s113
                                 + 0.712155 w_k1_FDX_s113
                                 + 0.863375 w_k1_IBM_s113
                                 + 0.764598 w_k1_INTC_s113
                                 + 0.689797 w_k1_KO_s113
                                 + 1.08528 w_k1_MAT_s113
                                 + 0.604043 w_k1_MSFT_s113
                                 + 0.813366 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.9599 w_k1_ADM_s113
                                 + 1.00167 w_k1_AXP_s113
                                 + 0.73898 w_k1_DIS_s113
                                 + 0.970796 w_k1_FDX_s113
                                 + 0.773523 w_k1_IBM_s113
                                 + 0.86658 w_k1_INTC_s113
                                 + 0.661977 w_k1_KO_s113
                                 + 0.986002 w_k1_MAT_s113
                                 + 0.892232 w_k1_MSFT_s113
                                 + 0.877514 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.679995 w_k1_ADM_s113
                                 + 0.653044 w_k1_AXP_s113
                                 + 0.642593 w_k1_DIS_s113
                                 + 0.789579 w_k1_FDX_s113
                                 + 0.679402 w_k1_IBM_s113
                                 + 0.823247 w_k1_INTC_s113
                                 + 0.60513 w_k1_KO_s113
                                 + 0.581872 w_k1_MAT_s113
                                 + 0.741583 w_k1_MSFT_s113
                                 + 0.771509 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.680251 w_k1_ADM_s114
                                 + 0.686852 w_k1_AXP_s114
                                 + 0.602191 w_k1_DIS_s114
                                 + 0.687212 w_k1_FDX_s114
                                 + 0.730165 w_k1_IBM_s114
                                 + 0.813048 w_k1_INTC_s114
                                 + 0.876602 w_k1_KO_s114
                                 + 0.687739 w_k1_MAT_s114
                                 + 0.739948 w_k1_MSFT_s114
                                 + 0.797794 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.995164 w_k1_ADM_s114
                                 + 0.722393 w_k1_AXP_s114
                                 + 0.751875 w_k1_DIS_s114
                                 + 0.834296 w_k1_FDX_s114
                                 + 0.69222 w_k1_IBM_s114
                                 + 0.89157 w_k1_INTC_s114
                                 + 0.96656 w_k1_KO_s114
                                 + 0.547591 w_k1_MAT_s114
                                 + 0.937993 w_k1_MSFT_s114
                                 + 0.88044 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.22555 w_k1_ADM_s114
                                 + 1.35156 w_k1_AXP_s114
                                 + 1.27922 w_k1_DIS_s114
                                 + 1.07421 w_k1_FDX_s114
                                 + 1.60774 w_k1_IBM_s114
                                 + 1.17702 w_k1_INTC_s114
                                 + 1.60409 w_k1_KO_s114 + 1.3582 w_k1_MAT_s114
                                 + 1.19241 w_k1_MSFT_s114 + 1.2421 w_k1_PG_s114
                                 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.704247 w_k1_ADM_s114
                                 + 0.865504 w_k1_AXP_s114
                                 + 0.811703 w_k1_DIS_s114
                                 + 0.79393 w_k1_FDX_s114
                                 + 1.14446 w_k1_IBM_s114
                                 + 0.783769 w_k1_INTC_s114
                                 + 1.16371 w_k1_KO_s114 + 1.0504 w_k1_MAT_s114
                                 + 0.867625 w_k1_MSFT_s114
                                 + 0.881368 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.19511 w_k1_ADM_s114
                                 + 0.920734 w_k1_AXP_s114
                                 + 1.12216 w_k1_DIS_s114
                                 + 1.02489 w_k1_FDX_s114
                                 + 1.00356 w_k1_IBM_s114
                                 + 0.931376 w_k1_INTC_s114
                                 + 1.17705 w_k1_KO_s114
                                 + 0.666688 w_k1_MAT_s114
                                 + 1.06879 w_k1_MSFT_s114
                                 + 1.10166 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.99401 w_k1_ADM_s114
                                 + 1.52761 w_k1_AXP_s114
                                 + 1.32408 w_k1_DIS_s114
                                 + 1.23623 w_k1_FDX_s114 + 1.245 w_k1_IBM_s114
                                 + 1.06216 w_k1_INTC_s114
                                 + 1.45138 w_k1_KO_s114 + 1.41629 w_k1_MAT_s114
                                 + 1.06119 w_k1_MSFT_s114
                                 + 1.14491 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.07968 w_k1_ADM_s114
                                 + 1.1611 w_k1_AXP_s114
                                 + 0.922269 w_k1_DIS_s114
                                 + 1.09958 w_k1_FDX_s114
                                 + 0.990426 w_k1_IBM_s114
                                 + 1.0109 w_k1_INTC_s114 + 1.18608 w_k1_KO_s114
                                 + 1.14489 w_k1_MAT_s114
                                 + 0.996083 w_k1_MSFT_s114
                                 + 1.03895 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.0549 w_k1_ADM_s114
                                 + 1.17095 w_k1_AXP_s114
                                 + 1.08468 w_k1_DIS_s114
                                 + 1.00353 w_k1_FDX_s114
                                 + 0.864002 w_k1_IBM_s114
                                 + 0.940279 w_k1_INTC_s114
                                 + 1.02283 w_k1_KO_s114
                                 + 0.684028 w_k1_MAT_s114
                                 + 0.980953 w_k1_MSFT_s114
                                 + 0.983498 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.998446 w_k1_ADM_s115
                                 + 0.812699 w_k1_AXP_s115
                                 + 0.693875 w_k1_DIS_s115
                                 + 0.920702 w_k1_FDX_s115
                                 + 0.671889 w_k1_IBM_s115
                                 + 0.749073 w_k1_INTC_s115
                                 + 0.692595 w_k1_KO_s115
                                 + 0.683034 w_k1_MAT_s115
                                 + 0.901713 w_k1_MSFT_s115
                                 + 0.850389 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.23092 w_k1_ADM_s115
                                 + 1.13521 w_k1_AXP_s115
                                 + 1.00722 w_k1_DIS_s115
                                 + 1.17967 w_k1_FDX_s115
                                 + 0.716386 w_k1_IBM_s115
                                 + 1.12945 w_k1_INTC_s115
                                 + 0.880314 w_k1_KO_s115
                                 + 0.856443 w_k1_MAT_s115
                                 + 1.15271 w_k1_MSFT_s115
                                 + 0.969073 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.32274 w_k1_ADM_s115
                                 + 1.51512 w_k1_AXP_s115
                                 + 1.57282 w_k1_DIS_s115
                                 + 1.34348 w_k1_FDX_s115
                                 + 1.57159 w_k1_IBM_s115
                                 + 1.15672 w_k1_INTC_s115
                                 + 1.36192 w_k1_KO_s115 + 1.7131 w_k1_MAT_s115
                                 + 1.29665 w_k1_MSFT_s115
                                 + 1.13692 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.31617 w_k1_ADM_s115
                                 + 1.5513 w_k1_AXP_s115 + 1.40044 w_k1_DIS_s115
                                 + 1.24517 w_k1_FDX_s115
                                 + 1.10799 w_k1_IBM_s115
                                 + 0.953745 w_k1_INTC_s115
                                 + 1.14088 w_k1_KO_s115 + 1.23935 w_k1_MAT_s115
                                 + 1.09346 w_k1_MSFT_s115
                                 + 1.06937 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.04377 w_k1_ADM_s115
                                 + 1.22604 w_k1_AXP_s115
                                 + 0.954052 w_k1_DIS_s115 + 1.008 w_k1_FDX_s115
                                 + 1.18858 w_k1_IBM_s115
                                 + 1.02216 w_k1_INTC_s115
                                 + 1.08787 w_k1_KO_s115 + 1.36808 w_k1_MAT_s115
                                 + 1.06378 w_k1_MSFT_s115
                                 + 1.12276 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.926375 w_k1_ADM_s115
                                 + 0.908845 w_k1_AXP_s115
                                 + 0.917368 w_k1_DIS_s115
                                 + 0.917686 w_k1_FDX_s115
                                 + 1.10531 w_k1_IBM_s115
                                 + 0.851361 w_k1_INTC_s115
                                 + 0.915715 w_k1_KO_s115
                                 + 0.908065 w_k1_MAT_s115
                                 + 0.925998 w_k1_MSFT_s115
                                 + 0.91198 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.87437 w_k1_ADM_s115
                                 + 0.767006 w_k1_AXP_s115
                                 + 0.623869 w_k1_DIS_s115
                                 + 0.787614 w_k1_FDX_s115
                                 + 0.880591 w_k1_IBM_s115
                                 + 0.829484 w_k1_INTC_s115
                                 + 0.774624 w_k1_KO_s115
                                 + 0.865392 w_k1_MAT_s115
                                 + 0.816677 w_k1_MSFT_s115
                                 + 0.802298 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.12532 w_k1_ADM_s115
                                 + 1.16687 w_k1_AXP_s115
                                 + 1.37287 w_k1_DIS_s115
                                 + 1.40731 w_k1_FDX_s115
                                 + 0.93819 w_k1_IBM_s115
                                 + 1.12997 w_k1_INTC_s115
                                 + 1.18973 w_k1_KO_s115 + 1.4779 w_k1_MAT_s115
                                 + 1.22591 w_k1_MSFT_s115
                                 + 1.04144 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.858861 w_k1_ADM_s116
                                 + 0.827296 w_k1_AXP_s116
                                 + 0.665317 w_k1_DIS_s116
                                 + 0.794685 w_k1_FDX_s116
                                 + 0.788802 w_k1_IBM_s116
                                 + 0.912516 w_k1_INTC_s116
                                 + 0.837364 w_k1_KO_s116
                                 + 1.23305 w_k1_MAT_s116
                                 + 0.829121 w_k1_MSFT_s116
                                 + 0.851627 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.97169 w_k1_ADM_s116
                                 + 0.853456 w_k1_AXP_s116
                                 + 0.984345 w_k1_DIS_s116
                                 + 0.922021 w_k1_FDX_s116
                                 + 1.02199 w_k1_IBM_s116
                                 + 1.00093 w_k1_INTC_s116
                                 + 1.00535 w_k1_KO_s116 + 1.3033 w_k1_MAT_s116
                                 + 0.999071 w_k1_MSFT_s116
                                 + 0.897873 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.870921 w_k1_ADM_s116
                                 + 0.743378 w_k1_AXP_s116
                                 + 1.13194 w_k1_DIS_s116
                                 + 0.876448 w_k1_FDX_s116
                                 + 1.23045 w_k1_IBM_s116
                                 + 1.19103 w_k1_INTC_s116
                                 + 1.22304 w_k1_KO_s116 + 1.55728 w_k1_MAT_s116
                                 + 1.00003 w_k1_MSFT_s116
                                 + 1.06494 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.896362 w_k1_ADM_s116
                                 + 1.27847 w_k1_AXP_s116
                                 + 0.952445 w_k1_DIS_s116
                                 + 1.08158 w_k1_FDX_s116 + 1.0012 w_k1_IBM_s116
                                 + 1.11467 w_k1_INTC_s116 + 1.1307 w_k1_KO_s116
                                 + 1.46474 w_k1_MAT_s116
                                 + 0.918645 w_k1_MSFT_s116
                                 + 1.09648 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.12701 w_k1_ADM_s116
                                 + 1.34126 w_k1_AXP_s116
                                 + 1.20088 w_k1_DIS_s116
                                 + 1.06533 w_k1_FDX_s116
                                 + 1.52508 w_k1_IBM_s116
                                 + 1.18457 w_k1_INTC_s116
                                 + 1.55601 w_k1_KO_s116 + 2.14219 w_k1_MAT_s116
                                 + 1.05667 w_k1_MSFT_s116
                                 + 1.16035 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.964736 w_k1_ADM_s116
                                 + 0.870163 w_k1_AXP_s116
                                 + 1.27272 w_k1_DIS_s116
                                 + 1.19551 w_k1_FDX_s116
                                 + 0.914345 w_k1_IBM_s116
                                 + 1.03081 w_k1_INTC_s116
                                 + 1.27604 w_k1_KO_s116 + 1.25367 w_k1_MAT_s116
                                 + 1.20128 w_k1_MSFT_s116
                                 + 1.13371 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.778686 w_k1_ADM_s116
                                 + 0.780878 w_k1_AXP_s116
                                 + 0.909048 w_k1_DIS_s116
                                 + 0.966613 w_k1_FDX_s116
                                 + 0.693206 w_k1_IBM_s116
                                 + 1.06445 w_k1_INTC_s116
                                 + 0.922884 w_k1_KO_s116
                                 + 0.760837 w_k1_MAT_s116
                                 + 0.90367 w_k1_MSFT_s116 + 1.0022 w_k1_PG_s116
                                 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.085 w_k1_ADM_s116
                                 + 1.27361 w_k1_AXP_s116
                                 + 1.48908 w_k1_DIS_s116
                                 + 1.37642 w_k1_FDX_s116
                                 + 1.11692 w_k1_IBM_s116
                                 + 1.3042 w_k1_INTC_s116 + 1.5171 w_k1_KO_s116
                                 + 1.66681 w_k1_MAT_s116
                                 + 1.23989 w_k1_MSFT_s116
                                 + 1.14565 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.892854 w_k1_ADM_s117
                                 + 1.00485 w_k1_AXP_s117
                                 + 1.02926 w_k1_DIS_s117 + 1.0265 w_k1_FDX_s117
                                 + 0.724883 w_k1_IBM_s117
                                 + 0.792297 w_k1_INTC_s117
                                 + 0.94737 w_k1_KO_s117
                                 + 0.797698 w_k1_MAT_s117
                                 + 1.02329 w_k1_MSFT_s117
                                 + 0.880771 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.01273 w_k1_ADM_s117
                                 + 0.958102 w_k1_AXP_s117
                                 + 1.35818 w_k1_DIS_s117 + 1.0869 w_k1_FDX_s117
                                 + 1.49029 w_k1_IBM_s117
                                 + 0.87644 w_k1_INTC_s117 + 1.4389 w_k1_KO_s117
                                 + 1.56666 w_k1_MAT_s117
                                 + 1.11256 w_k1_MSFT_s117
                                 + 1.19985 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.827903 w_k1_ADM_s117
                                 + 0.874221 w_k1_AXP_s117
                                 + 1.08582 w_k1_DIS_s117
                                 + 0.982232 w_k1_FDX_s117
                                 + 1.02773 w_k1_IBM_s117
                                 + 0.942789 w_k1_INTC_s117
                                 + 1.12659 w_k1_KO_s117
                                 + 0.872688 w_k1_MAT_s117
                                 + 1.08973 w_k1_MSFT_s117 + 1.0619 w_k1_PG_s117
                                 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.12835 w_k1_ADM_s117
                                 + 1.27163 w_k1_AXP_s117
                                 + 1.60675 w_k1_DIS_s117 + 1.2566 w_k1_FDX_s117
                                 + 1.18631 w_k1_IBM_s117
                                 + 1.00596 w_k1_INTC_s117
                                 + 1.24686 w_k1_KO_s117 + 1.61691 w_k1_MAT_s117
                                 + 1.20849 w_k1_MSFT_s117
                                 + 1.17046 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.690726 w_k1_ADM_s117
                                 + 0.843236 w_k1_AXP_s117
                                 + 0.742748 w_k1_DIS_s117
                                 + 0.80317 w_k1_FDX_s117
                                 + 0.90088 w_k1_IBM_s117
                                 + 0.791072 w_k1_INTC_s117
                                 + 0.970096 w_k1_KO_s117
                                 + 0.955955 w_k1_MAT_s117
                                 + 0.777533 w_k1_MSFT_s117
                                 + 0.842167 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.701104 w_k1_ADM_s117
                                 + 0.91461 w_k1_AXP_s117
                                 + 0.914977 w_k1_DIS_s117
                                 + 0.74542 w_k1_FDX_s117
                                 + 1.50368 w_k1_IBM_s117
                                 + 0.741154 w_k1_INTC_s117
                                 + 1.28295 w_k1_KO_s117 + 1.55843 w_k1_MAT_s117
                                 + 0.874138 w_k1_MSFT_s117
                                 + 0.963012 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.756412 w_k1_ADM_s117
                                 + 0.725352 w_k1_AXP_s117
                                 + 0.777383 w_k1_DIS_s117
                                 + 0.779543 w_k1_FDX_s117
                                 + 0.703846 w_k1_IBM_s117
                                 + 0.728254 w_k1_INTC_s117
                                 + 0.929138 w_k1_KO_s117
                                 + 0.820956 w_k1_MAT_s117
                                 + 0.785382 w_k1_MSFT_s117
                                 + 0.904209 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.05797 w_k1_ADM_s117
                                 + 1.22199 w_k1_AXP_s117
                                 + 1.49251 w_k1_DIS_s117
                                 + 1.31955 w_k1_FDX_s117
                                 + 0.758979 w_k1_IBM_s117
                                 + 1.28322 w_k1_INTC_s117
                                 + 1.14726 w_k1_KO_s117
                                 + 0.602898 w_k1_MAT_s117
                                 + 1.24713 w_k1_MSFT_s117 + 1.0257 w_k1_PG_s117
                                 >= 1
 k1_return_constraint:           0.015285639642618 w_k1_ADM_s000
                                 + 0.0318112755678145 w_k1_ADM_s001
                                 + 0.0348379363306655 w_k1_ADM_s002
                                 + 0.00699811476226567 w_k1_ADM_s003
                                 + 0.03061727114446 w_k1_ADM_s004
                                 + 0.0140974319474352 w_k1_ADM_s005
                                 + 0.0134947446713118 w_k1_ADM_s006
                                 + 0.0180010022663983 w_k1_ADM_s007
                                 + 0.0464550420408503 w_k1_ADM_s010
                                 + 0.0201688263149714 w_k1_ADM_s011
                                 + 0.019543119769575 w_k1_ADM_s012
                                 + 0.0275229272459979 w_k1_ADM_s013
                                 + 0.0453056089706469 w_k1_ADM_s014
                                 + 0.0238456205278739 w_k1_ADM_s015
                                 + 0.0411259407963893 w_k1_ADM_s016
                                 + 0.0922215934343679 w_k1_ADM_s017
                                 + 0.037552106272596 w_k1_ADM_s100
                                 + 0.0116169096512973 w_k1_ADM_s101
                                 + 0.0184956121484175 w_k1_ADM_s102
                                 + 0.0335152554690905 w_k1_ADM_s103
                                 + 0.0123457225403339 w_k1_ADM_s104
                                 + 0.0459938013075594 w_k1_ADM_s105
                                 + 0.0199944043316733 w_k1_ADM_s106
                                 + 0.0236528540540781 w_k1_ADM_s107
                                 + 0.0236306973185128 w_k1_ADM_s110
                                 + 0.0900202796009687 w_k1_ADM_s111
                                 + 0.0311381478704745 w_k1_ADM_s112
                                 + 0.0678565427144719 w_k1_ADM_s113
                                 + 0.0299534998509447 w_k1_ADM_s114
                                 + 0.0196485710553975 w_k1_ADM_s115
                                 + 0.031155642114463 w_k1_ADM_s116
                                 + 0.0265170931845218 w_k1_ADM_s117
                                 + 0.0138162253627249 w_k1_AXP_s000
                                 + 0.0334564176788417 w_k1_AXP_s001
                                 + 0.0332798536422531 w_k1_AXP_s002
                                 + 0.00971898513881486 w_k1_AXP_s003
                                 + 0.0306759842903288 w_k1_AXP_s004
                                 + 0.0139124272420172 w_k1_AXP_s005
                                 + 0.0149251157840559 w_k1_AXP_s006
                                 + 0.0201003323013163 w_k1_AXP_s007
                                 + 0.0492509971429344 w_k1_AXP_s010
                                 + 0.0238354251005319 w_k1_AXP_s011
                                 + 0.0208707369000131 w_k1_AXP_s012
                                 + 0.0329905211058699 w_k1_AXP_s013
                                 + 0.044928550021878 w_k1_AXP_s014
                                 + 0.0268875132969011 w_k1_AXP_s015
                                 + 0.0354001518495915 w_k1_AXP_s016
                                 + 0.0973268519801705 w_k1_AXP_s017
                                 + 0.035425614544107 w_k1_AXP_s100
                                 + 0.0120800901559552 w_k1_AXP_s101
                                 + 0.0184499239331715 w_k1_AXP_s102
                                 + 0.0343401796653776 w_k1_AXP_s103
                                 + 0.0111007988748568 w_k1_AXP_s104
                                 + 0.0413563897766487 w_k1_AXP_s105
                                 + 0.0200405630063999 w_k1_AXP_s106
                                 + 0.0233600420318324 w_k1_AXP_s107
                                 + 0.0263674051786622 w_k1_AXP_s110
                                 + 0.0760425595518829 w_k1_AXP_s111
                                 + 0.0348303297695269 w_k1_AXP_s112
                                 + 0.0810160871444833 w_k1_AXP_s113
                                 + 0.0343352401114368 w_k1_AXP_s114
                                 + 0.0198655739621814 w_k1_AXP_s115
                                 + 0.0326838314776754 w_k1_AXP_s116
                                 + 0.0288896484503694 w_k1_AXP_s117
                                 + 0.0147108625236989 w_k1_DIS_s000
                                 + 0.0347452708301507 w_k1_DIS_s001
                                 + 0.0321267362233485 w_k1_DIS_s002
                                 + 0.00854695684112133 w_k1_DIS_s003
                                 + 0.0317786780607784 w_k1_DIS_s004
                                 + 0.0146007898287697 w_k1_DIS_s005
                                 + 0.0144409863195121 w_k1_DIS_s006
                                 + 0.0205914197563717 w_k1_DIS_s007
                                 + 0.0467451969469191 w_k1_DIS_s010
                                 + 0.0217314974302386 w_k1_DIS_s011
                                 + 0.0222033022444093 w_k1_DIS_s012
                                 + 0.0298875661184366 w_k1_DIS_s013
                                 + 0.0519138128363204 w_k1_DIS_s014
                                 + 0.0234423909255573 w_k1_DIS_s015
                                 + 0.0347510914524298 w_k1_DIS_s016
                                 + 0.101835011871748 w_k1_DIS_s017
                                 + 0.0330826738322754 w_k1_DIS_s100
                                 + 0.0109744036640826 w_k1_DIS_s101
                                 + 0.0201674856687874 w_k1_DIS_s102
                                 + 0.0341932881223225 w_k1_DIS_s103
                                 + 0.0105365570232737 w_k1_DIS_s104
                                 + 0.0469867077509881 w_k1_DIS_s105
                                 + 0.0208151643426029 w_k1_DIS_s106
                                 + 0.0247368112340567 w_k1_DIS_s107
                                 + 0.0306239236153695 w_k1_DIS_s110
                                 + 0.0783356525830424 w_k1_DIS_s111
                                 + 0.0326566404669846 w_k1_DIS_s112
                                 + 0.0782659233837319 w_k1_DIS_s113
                                 + 0.031642856465918 w_k1_DIS_s114
                                 + 0.0186238208063621 w_k1_DIS_s115
                                 + 0.0342267459642965 w_k1_DIS_s116
                                 + 0.0339919848969701 w_k1_DIS_s117
                                 + 0.0141169977560764 w_k1_FDX_s000
                                 + 0.0330634337248805 w_k1_FDX_s001
                                 + 0.031735427305837 w_k1_FDX_s002
                                 + 0.00879502631253674 w_k1_FDX_s003
                                 + 0.0320204508178515 w_k1_FDX_s004
                                 + 0.0139238102643567 w_k1_FDX_s005
                                 + 0.0141889935434837 w_k1_FDX_s006
                                 + 0.0215050304030978 w_k1_FDX_s007
                                 + 0.0454284230561058 w_k1_FDX_s010
                                 + 0.0210289330983937 w_k1_FDX_s011
                                 + 0.0211400994927123 w_k1_FDX_s012
                                 + 0.0299331039015018 w_k1_FDX_s013
                                 + 0.046648616212862 w_k1_FDX_s014
                                 + 0.0234519386314436 w_k1_FDX_s015
                                 + 0.0336262408772009 w_k1_FDX_s016
                                 + 0.0960365040164199 w_k1_FDX_s017
                                 + 0.0355454660209412 w_k1_FDX_s100
                                 + 0.0112731036518374 w_k1_FDX_s101
                                 + 0.0182275428783475 w_k1_FDX_s102
                                 + 0.032855049704302 w_k1_FDX_s103
                                 + 0.0106474987082914 w_k1_FDX_s104
                                 + 0.044546727033128 w_k1_FDX_s105
                                 + 0.0217146095123262 w_k1_FDX_s106
                                 + 0.0240086454860414 w_k1_FDX_s107
                                 + 0.0266059520557529 w_k1_FDX_s110
                                 + 0.0779183030856587 w_k1_FDX_s111
                                 + 0.0337899950554141 w_k1_FDX_s112
                                 + 0.0770695749745544 w_k1_FDX_s113
                                 + 0.0306813671235971 w_k1_FDX_s114
                                 + 0.0193446956530719 w_k1_FDX_s115
                                 + 0.0331584552469331 w_k1_FDX_s116
                                 + 0.0299731559826988 w_k1_FDX_s117
                                 + 0.0140263586662556 w_k1_IBM_s000
                                 + 0.0332013181097076 w_k1_IBM_s001
                                 + 0.0313669095707809 w_k1_IBM_s002
                                 + 0.00898979271847314 w_k1_IBM_s003
                                 + 0.0321851384945385 w_k1_IBM_s004
                                 + 0.0147096858233679 w_k1_IBM_s005
                                 + 0.0148857389955972 w_k1_IBM_s006
                                 + 0.020988893229398 w_k1_IBM_s007
                                 + 0.0473370130867844 w_k1_IBM_s010
                                 + 0.0216493687513777 w_k1_IBM_s011
                                 + 0.0213118485220512 w_k1_IBM_s012
                                 + 0.0314135893337544 w_k1_IBM_s013
                                 + 0.047898736723034 w_k1_IBM_s014
                                 + 0.0240516178356281 w_k1_IBM_s015
                                 + 0.0329080490037721 w_k1_IBM_s016
                                 + 0.0984612472290551 w_k1_IBM_s017
                                 + 0.0352615816417886 w_k1_IBM_s100
                                 + 0.0114725798839939 w_k1_IBM_s101
                                 + 0.0190523561664927 w_k1_IBM_s102
                                 + 0.0333455017720177 w_k1_IBM_s103
                                 + 0.011126886604807 w_k1_IBM_s104
                                 + 0.0450665391168081 w_k1_IBM_s105
                                 + 0.0217407944050833 w_k1_IBM_s106
                                 + 0.0239840323207366 w_k1_IBM_s107
                                 + 0.0270572170659914 w_k1_IBM_s110
                                 + 0.079365300657195 w_k1_IBM_s111
                                 + 0.0338412760656723 w_k1_IBM_s112
                                 + 0.0782808058854275 w_k1_IBM_s113
                                 + 0.0320946427417702 w_k1_IBM_s114
                                 + 0.0183398224135476 w_k1_IBM_s115
                                 + 0.0339565786123719 w_k1_IBM_s116
                                 + 0.0307287352332302 w_k1_IBM_s117
                                 + 0.0138290619721387 w_k1_INTC_s000
                                 + 0.0327850373017408 w_k1_INTC_s001
                                 + 0.0317245638243249 w_k1_INTC_s002
                                 + 0.0085633944463225 w_k1_INTC_s003
                                 + 0.033137203934317 w_k1_INTC_s004
                                 + 0.0169503207192668 w_k1_INTC_s005
                                 + 0.014234987169898 w_k1_INTC_s006
                                 + 0.021721321051284 w_k1_INTC_s007
                                 + 0.0480101101854235 w_k1_INTC_s010
                                 + 0.0234402612067781 w_k1_INTC_s011
                                 + 0.0217335478967154 w_k1_INTC_s012
                                 + 0.0313997861512191 w_k1_INTC_s013
                                 + 0.0491368086808506 w_k1_INTC_s014
                                 + 0.0217638022521932 w_k1_INTC_s015
                                 + 0.0334059975426698 w_k1_INTC_s016
                                 + 0.10666604734206 w_k1_INTC_s017
                                 + 0.0360729481493193 w_k1_INTC_s100
                                 + 0.0112586850216321 w_k1_INTC_s101
                                 + 0.019604466404536 w_k1_INTC_s102
                                 + 0.0295158544766802 w_k1_INTC_s103
                                 + 0.00925737523310657 w_k1_INTC_s104
                                 + 0.0371818447461424 w_k1_INTC_s105
                                 + 0.0208907284353931 w_k1_INTC_s106
                                 + 0.0230306242084368 w_k1_INTC_s107
                                 + 0.0276713757794452 w_k1_INTC_s110
                                 + 0.0801024303505317 w_k1_INTC_s111
                                 + 0.0363869692052742 w_k1_INTC_s112
                                 + 0.0722343740840776 w_k1_INTC_s113
                                 + 0.0291450534376369 w_k1_INTC_s114
                                 + 0.0170274189862018 w_k1_INTC_s115
                                 + 0.0351028609905284 w_k1_INTC_s116
                                 + 0.0262941051475411 w_k1_INTC_s117
                                 + 0.0136804485512993 w_k1_KO_s000
                                 + 0.0310268833261972 w_k1_KO_s001
                                 + 0.0350392622473246 w_k1_KO_s002
                                 + 0.00828372330543532 w_k1_KO_s003
                                 + 0.0308778396307949 w_k1_KO_s004
                                 + 0.0165373541940999 w_k1_KO_s005
                                 + 0.0141696533509813 w_k1_KO_s006
                                 + 0.0213701010481545 w_k1_KO_s007
                                 + 0.0516955708631827 w_k1_KO_s010
                                 + 0.0227029349523347 w_k1_KO_s011
                                 + 0.0223408064995138 w_k1_KO_s012
                                 + 0.0340516771727372 w_k1_KO_s013
                                 + 0.0508747224824618 w_k1_KO_s014
                                 + 0.0226213809606555 w_k1_KO_s015
                                 + 0.0318622353636741 w_k1_KO_s016
                                 + 0.109951758098979 w_k1_KO_s017
                                 + 0.0348996440025479 w_k1_KO_s100
                                 + 0.0105243097844591 w_k1_KO_s101
                                 + 0.0199498463274521 w_k1_KO_s102
                                 + 0.0327261498734748 w_k1_KO_s103
                                 + 0.010771499941439 w_k1_KO_s104
                                 + 0.0436125785542101 w_k1_KO_s105
                                 + 0.0214906324133648 w_k1_KO_s106
                                 + 0.0259696839291025 w_k1_KO_s107
                                 + 0.0259912732768008 w_k1_KO_s110
                                 + 0.0860327075483044 w_k1_KO_s111
                                 + 0.0333133121704974 w_k1_KO_s112
                                 + 0.0682781799232251 w_k1_KO_s113
                                 + 0.0366983153211188 w_k1_KO_s114
                                 + 0.0176659368034059 w_k1_KO_s115
                                 + 0.0380825971072685 w_k1_KO_s116
                                 + 0.0336220921421564 w_k1_KO_s117
                                 + 0.0165467526744814 w_k1_MAT_s000
                                 + 0.0262870836028252 w_k1_MAT_s001
                                 + 0.0299816374202477 w_k1_MAT_s002
                                 + 0.00958135050173358 w_k1_MAT_s003
                                 + 0.0284683533714232 w_k1_MAT_s004
                                 + 0.0155991007113413 w_k1_MAT_s005
                                 + 0.0204951665184543 w_k1_MAT_s006
                                 + 0.0280555686873769 w_k1_MAT_s007
                                 + 0.0437583732470356 w_k1_MAT_s010
                                 + 0.017503955635872 w_k1_MAT_s011
                                 + 0.0214963037803409 w_k1_MAT_s012
                                 + 0.0326500934593253 w_k1_MAT_s013
                                 + 0.0491360447152603 w_k1_MAT_s014
                                 + 0.021569848977668 w_k1_MAT_s015
                                 + 0.0319311345523169 w_k1_MAT_s016
                                 + 0.106154843788491 w_k1_MAT_s017
                                 + 0.0358352041686968 w_k1_MAT_s100
                                 + 0.00974544846790334 w_k1_MAT_s101
                                 + 0.0192107726663452 w_k1_MAT_s102
                                 + 0.0418331996808123 w_k1_MAT_s103
                                 + 0.0113187597669569 w_k1_MAT_s104
                                 + 0.0625519113008711 w_k1_MAT_s105
                                 + 0.0217384935064689 w_k1_MAT_s106
                                 + 0.023778608886077 w_k1_MAT_s107
                                 + 0.0303095173508827 w_k1_MAT_s110
                                 + 0.0686060128405855 w_k1_MAT_s111
                                 + 0.0354937937076305 w_k1_MAT_s112
                                 + 0.0855815851894147 w_k1_MAT_s113
                                 + 0.0304009495741359 w_k1_MAT_s114
                                 + 0.0194129267825548 w_k1_MAT_s115
                                 + 0.0468965368644619 w_k1_MAT_s116
                                 + 0.0328911977362698 w_k1_MAT_s117
                                 + 0.0136570543560709 w_k1_MSFT_s000
                                 + 0.0309960538338992 w_k1_MSFT_s001
                                 + 0.0340305111530776 w_k1_MSFT_s002
                                 + 0.00778306527414213 w_k1_MSFT_s003
                                 + 0.0323632072941659 w_k1_MSFT_s004
                                 + 0.0159176770536098 w_k1_MSFT_s005
                                 + 0.0133356802394757 w_k1_MSFT_s006
                                 + 0.0223798064416049 w_k1_MSFT_s007
                                 + 0.0517597296945181 w_k1_MSFT_s010
                                 + 0.0209927303114253 w_k1_MSFT_s011
                                 + 0.0206942753577589 w_k1_MSFT_s012
                                 + 0.0296540151148699 w_k1_MSFT_s013
                                 + 0.045556722603122 w_k1_MSFT_s014
                                 + 0.0236732517900658 w_k1_MSFT_s015
                                 + 0.0312424460532443 w_k1_MSFT_s016
                                 + 0.0973591471597044 w_k1_MSFT_s017
                                 + 0.0342632424322169 w_k1_MSFT_s100
                                 + 0.0101296659645522 w_k1_MSFT_s101
                                 + 0.0208075723115478 w_k1_MSFT_s102
                                 + 0.0337147234817708 w_k1_MSFT_s103
                                 + 0.00946285301148923 w_k1_MSFT_s104
                                 + 0.0440258095991709 w_k1_MSFT_s105
                                 + 0.0221814312986513 w_k1_MSFT_s106
                                 + 0.0249076222206056 w_k1_MSFT_s107
                                 + 0.0288162341157887 w_k1_MSFT_s110
                                 + 0.0846028534651878 w_k1_MSFT_s111
                                 + 0.0341419028718754 w_k1_MSFT_s112
                                 + 0.0706899034947361 w_k1_MSFT_s113
                                 + 0.0300786075021179 w_k1_MSFT_s114
                                 + 0.0186317789682053 w_k1_MSFT_s115
                                 + 0.0328478865391032 w_k1_MSFT_s116
                                 + 0.0303654194531077 w_k1_MSFT_s117
                                 + 0.0138300984278368 w_k1_PG_s000
                                 + 0.0318690754301009 w_k1_PG_s001
                                 + 0.0331090135865175 w_k1_PG_s002
                                 + 0.00830304543217444 w_k1_PG_s003
                                 + 0.0328842143370055 w_k1_PG_s004
                                 + 0.0147383692632366 w_k1_PG_s005
                                 + 0.0139229768469133 w_k1_PG_s006
                                 + 0.0203017133866717 w_k1_PG_s007
                                 + 0.0473244716746596 w_k1_PG_s010
                                 + 0.0213252785877945 w_k1_PG_s011
                                 + 0.02042585722154 w_k1_PG_s012
                                 + 0.0334649719907548 w_k1_PG_s013
                                 + 0.048505498321728 w_k1_PG_s014
                                 + 0.0255178378103163 w_k1_PG_s015
                                 + 0.0321439100121155 w_k1_PG_s016
                                 + 0.0934360840721948 w_k1_PG_s017
                                 + 0.0333128088663337 w_k1_PG_s100
                                 + 0.010329751947935 w_k1_PG_s101
                                 + 0.0175683967500346 w_k1_PG_s102
                                 + 0.0296848744393781 w_k1_PG_s103
                                 + 0.0102039656211453 w_k1_PG_s104
                                 + 0.0434342248540217 w_k1_PG_s105
                                 + 0.0212318840580685 w_k1_PG_s106
                                 + 0.024821530905381 w_k1_PG_s107
                                 + 0.0241875641897657 w_k1_PG_s110
                                 + 0.0755427833858964 w_k1_PG_s111
                                 + 0.0336132960664543 w_k1_PG_s112
                                 + 0.0703172287020408 w_k1_PG_s113
                                 + 0.031069604850561 w_k1_PG_s114
                                 + 0.0176322990181979 w_k1_PG_s115
                                 + 0.0332370306329662 w_k1_PG_s116
                                 + 0.0300230547413955 w_k1_PG_s117 >= 1.041
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
