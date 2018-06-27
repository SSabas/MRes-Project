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
 w_k1_ADM_s0_return_constraint:  1.02409 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.984683 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.17878 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.890038 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.16032 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.00395 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.845829 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.966479 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 1.3423 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.856658 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.06687 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.26541 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.11801 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.41968 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.979983 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.795673 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.997955 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.805014 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.867374 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.11008 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.30499 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.10621 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.12646 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.29468 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.08166 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.756805 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.13253 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.826817 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.997575 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.926251 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.929218 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.922012 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.704667 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.25436 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.1817 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.672804 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.27138 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.24284 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  0.988569 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.1079 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.25031 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.837948 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.35493 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.927721 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.82037 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.17756 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.62218 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.711457 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.977921 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.163 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.25606 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.72455 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.02441 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.734883 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.16694 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.705194 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.881406 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.3901 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.68919 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.16841 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.2815 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.70293 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.989857 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.704886 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.07124 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.837788 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.990754 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.809974 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.848509 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.895857 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.645419 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.1717 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.50563 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.551359 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.40994 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.17057 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.04163 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.13434 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.33917 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.749185 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.46698 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.885299 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.689594 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.89827 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.48438 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.644658 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.962299 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.39624 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.22793 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.65127 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.06113 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.747191 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.08445 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.609647 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.779255 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.57286 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.85433 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.0846 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.27533 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.82672 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.02822 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.526444 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.42983 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.657271 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.885373 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.6864 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.839065 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.799618 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.644714 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.43122 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.71664 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.508663 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.10333 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.29166 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.02189 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.999364 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.23008 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.853665 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.2519 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.940426 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.768272 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.909545 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 1.50426 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.78844 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.01484 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.23249 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.16698 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.26924 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.971558 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.732902 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.09348 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.718507 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.922037 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.16185 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.42571 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.938292 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.01252 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.55879 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.09921 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.65377 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.22292 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.848681 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.902257 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.737653 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.965774 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.888699 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.699236 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.16544 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.38258 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.666515 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.14601 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.30048 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.941268 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.3659 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.0527 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.027 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.44651 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.927632 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.991395 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.25496 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.03879 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.590553 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.826334 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.822096 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.24609 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.90682 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.99733 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.16033 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.788856 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.699648 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.550157 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.74509 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.26257 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.32805 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.48448 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 1.11917 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.800462 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.93844 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.09807 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.653277 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.06024 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.46293 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.71072 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.27316 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.922185 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.27081 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.37844 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.608114 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.954352 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.859986 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.02578 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.05241 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.20318 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.861168 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.28228 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.817389 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.718036 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.931996 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.2643 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.787931 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.903447 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.07691 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.08935 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.14586 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.0317 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.783228 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.0774 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.833718 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.841576 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.14723 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.39813 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.1133 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.01925 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.04159 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.882291 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.590158 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.14896 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.69216 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.683875 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.822665 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.06902 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.07141 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.923455 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.15458 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.34252 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.787744 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.35897 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.31596 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.988926 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.16783 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.21529 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.913049 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.42653 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.917804 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.764109 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.00505 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.3002 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.667838 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.900361 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.0519 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.1909 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.44844 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.03697 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.974848 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.00191 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.694194 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.699641 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.46036 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.44366 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.15249 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.22304 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.23674 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.862374 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.68739 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.18888 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.627991 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.897498 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.06879 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.866737 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.08382 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.856048 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.21671 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.42408 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.571988 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.00639 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.06845 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.950738 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.47372 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.05069 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.990593 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.23326 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.804933 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.03666 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.50942 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.37478 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.504456 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.798791 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.925232 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.40881 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 2.44288 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.885425 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.960205 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.828359 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.647256 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.629475 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.84409 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.36069 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.42253 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.78361 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 2.02897 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.01747 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.05742 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.24045 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.809995 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.27679 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.6745 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.915699 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.28515 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.858313 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.43374 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.8313 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.503308 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.18096 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.925032 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 0.997005 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.04608 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.25122 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.854753 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.22198 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.85999 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.754388 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 0.902764 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.30361 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.734925 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.08219 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.10527 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.14769 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.11663 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.04094 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.813267 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.07853 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.799754 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.951436 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.26987 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.52523 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.07882 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.0388 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.35618 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.03567 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.625504 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.16121 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.819125 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.784688 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.8218 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.958295 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.955599 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.818154 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.25263 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.53572 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.829579 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.14065 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.26578 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.00412 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.07265 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.08761 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.918759 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.18263 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.936734 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.889414 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.0024 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.24912 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.876762 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.01625 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.1219 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.10431 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.46789 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.06197 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.870402 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.984989 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.765178 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.873224 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.20403 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.15496 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.03086 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.05681 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.32116 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.985436 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.799063 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.09959 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.850399 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.948904 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.962237 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.940775 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.04411 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.909782 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.17009 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.1831 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.700848 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.1191 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.16249 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.469001965843038 z_k1_s0000
                                 - 0.416001125066221 z_k1_s0001
                                 - 0.618000590397147 z_k1_s0002
                                 - 0.581000376486118 z_k1_s0003
                                 - 0.377001235123142 z_k1_s0004
                                 - 0.685002841809043 z_k1_s0005
                                 - 0.352000916609777 z_k1_s0006
                                 - 1.10000114015052 z_k1_s0007
                                 - 0.927005196306931 z_k1_s0010
                                 - 0.32100164270232 z_k1_s0011
                                 - 0.27600119714866 z_k1_s0012
                                 - 0.449001122882219 z_k1_s0013
                                 - 0.362001640898208 z_k1_s0014
                                 - 0.343001565558297 z_k1_s0015
                                 - 0.238001046468838 z_k1_s0016
                                 - 0.988005479282625 z_k1_s0017
                                 - 0.491999103587395 z_k1_s0020
                                 - 0.461999126661327 z_k1_s0021
                                 - 0.500999355665275 z_k1_s0022
                                 - 0.146000146706924 z_k1_s0023
                                 - 0.246000069793817 z_k1_s0024
                                 - 0.384001258654029 z_k1_s0025
                                 - 0.220000089791225 z_k1_s0026
                                 - 0.139000152090841 z_k1_s0027
                                 - 1.23400295825003 z_k1_s0030
                                 - 1.44400382023455 z_k1_s0031
                                 - 0.944004401559426 z_k1_s0032
                                 - 0.453000279222365 z_k1_s0033
                                 - 0.412000374297046 z_k1_s0034
                                 - 0.764000502312801 z_k1_s0035
                                 - 1.94200428416788 z_k1_s0036
                                 - 0.70800053581716 z_k1_s0037
                                 - 0.48099942913314 z_k1_s0040
                                 - 0.563999478517749 z_k1_s0041
                                 - 1.35799769552402 z_k1_s0042
                                 - 0.443999377046048 z_k1_s0043
                                 - 1.0139992350371 z_k1_s0044
                                 - 0.822999219127865 z_k1_s0045
                                 - 0.954001680299366 z_k1_s0046
                                 - 0.601999801850235 z_k1_s0047
                                 - 0.230000560634987 z_k1_s0050
                                 - 0.30600019362696 z_k1_s0051
                                 - 0.476999915359352 z_k1_s0052
                                 - 0.233999503949494 z_k1_s0053
                                 - 0.147000148840448 z_k1_s0054
                                 - 0.230999201462706 z_k1_s0055
                                 - 0.399999451537845 z_k1_s0056
                                 - 0.165000211759724 z_k1_s0057
                                 - 0.485001254616767 z_k1_s0060
                                 - 1.05900210168751 z_k1_s0061
                                 - 1.09500541802075 z_k1_s0062
                                 - 0.497001687259193 z_k1_s0063
                                 - 0.553002865254696 z_k1_s0064
                                 - 0.232000627197379 z_k1_s0065
                                 - 0.413001181769664 z_k1_s0066
                                 - 0.712002794849698 z_k1_s0067
                                 - 0.506001492035359 z_k1_s0070
                                 - 0.148000173083782 z_k1_s0071
                                 - 0.506001492035359 z_k1_s0072
                                 - 0.587999586348753 z_k1_s0073
                                 - 0.476998971435773 z_k1_s0074
                                 - 0.303000017843048 z_k1_s0075
                                 - 0.349000447482714 z_k1_s0076
                                 - 0.35700136585508 z_k1_s0077
                                 - 1.02900144807668 z_k1_s0100
                                 - 1.14299323326949 z_k1_s0101
                                 - 2.35699340879963 z_k1_s0102
                                 - 0.498998687202289 z_k1_s0103
                                 - 0.520998678769417 z_k1_s0104
                                 - 0.73299819696261 z_k1_s0105
                                 - 1.01899984973146 z_k1_s0106
                                 - 1.51099918048827 z_k1_s0107
                                 - 0.12999995858255 z_k1_s0110
                                 - 0.410999263892657 z_k1_s0111
                                 - 0.291999991271554 z_k1_s0112
                                 - 0.424999638789155 z_k1_s0113
                                 - 0.167999997658976 z_k1_s0114
                                 - 0.25799908080863 z_k1_s0115
                                 - 0.132000012139192 z_k1_s0116
                                 - 0.141000057444127 z_k1_s0117
                                 - 0.351000784387356 z_k1_s0120
                                 - 0.915998552582719 z_k1_s0121
                                 - 0.723003522224746 z_k1_s0122
                                 - 0.763999695177365 z_k1_s0123
                                 - 0.72700285236601 z_k1_s0124
                                 - 1.06600382292677 z_k1_s0125
                                 - 1.15900344373449 z_k1_s0126
                                 - 0.372000306633657 z_k1_s0127
                                 - 1.3309968056746 z_k1_s0130
                                 - 0.831995212009026 z_k1_s0131
                                 - 1.39700023834423 z_k1_s0132
                                 - 1.35899431178617 z_k1_s0133
                                 - 0.315999581136673 z_k1_s0134
                                 - 0.482998983925609 z_k1_s0135
                                 - 0.319999328323795 z_k1_s0136
                                 - 1.2039941505586 z_k1_s0137
                                 - 1.14399784396995 z_k1_s0140
                                 - 0.605997917543755 z_k1_s0141
                                 - 0.563000506219087 z_k1_s0142
                                 - 0.494000138650394 z_k1_s0143
                                 - 0.85399784197071 z_k1_s0144
                                 - 0.299999673616536 z_k1_s0145
                                 - 1.17499923527297 z_k1_s0146
                                 - 0.466999848653985 z_k1_s0147
                                 - 0.297999610724422 z_k1_s0150
                                 - 0.946000032239174 z_k1_s0151
                                 - 0.769000449517409 z_k1_s0152
                                 - 0.403999647984587 z_k1_s0153
                                 - 0.476999397571619 z_k1_s0154
                                 - 0.899999444549563 z_k1_s0155
                                 - 0.279999827193197 z_k1_s0156
                                 - 1.06000037260165 z_k1_s0157
                                 - 0.54100072893448 z_k1_s0160
                                 - 0.279999998390546 z_k1_s0161
                                 - 0.429997799940459 z_k1_s0162
                                 - 0.739000777278477 z_k1_s0163
                                 - 0.599998541194958 z_k1_s0164
                                 - 0.220999741131632 z_k1_s0165
                                 - 0.817998676745367 z_k1_s0166
                                 - 0.446000314704024 z_k1_s0167
                                 - 0.527999974646096 z_k1_s0170
                                 - 0.247000138176869 z_k1_s0171
                                 - 0.473998476864087 z_k1_s0172
                                 - 0.333999883936733 z_k1_s0173
                                 - 0.504998225313002 z_k1_s0174
                                 - 0.631001720137689 z_k1_s0175
                                 - 0.541999473848826 z_k1_s0176
                                 - 0.740000964716134 z_k1_s0177
                                 - 0.0490000669754559 z_k1_s1000
                                 - 0.0339999432075709 z_k1_s1001
                                 - 0.0399999467147747 z_k1_s1002
                                 - 0.0349999821253968 z_k1_s1003
                                 - 0.0879997447726537 z_k1_s1004
                                 - 0.0899998226083053 z_k1_s1005
                                 - 0.0389999537968991 z_k1_s1006
                                 - 0.0850000880186785 z_k1_s1007
                                 - 0.0759999642278212 z_k1_s1010
                                 - 0.0320000207906862 z_k1_s1011
                                 - 0.0470000550988372 z_k1_s1012
                                 - 0.0639998319812283 z_k1_s1013
                                 - 0.0580000278581119 z_k1_s1014
                                 - 0.087000303787348 z_k1_s1015
                                 - 0.0639998319812283 z_k1_s1016
                                 - 0.0960000099720225 z_k1_s1017
                                 - 0.232000801864684 z_k1_s1020
                                 - 0.256000748160926 z_k1_s1021
                                 - 0.110000018057979 z_k1_s1022
                                 - 0.234000797389371 z_k1_s1023
                                 - 0.100000040434545 z_k1_s1024
                                 - 0.100000040434545 z_k1_s1025
                                 - 0.103000033721575 z_k1_s1026
                                 - 0.185999583795879 z_k1_s1027
                                 - 0.101000160117599 z_k1_s1030
                                 - 0.0579997605133883 z_k1_s1031
                                 - 0.053000145210168 z_k1_s1032
                                 - 0.102999903238984 z_k1_s1033
                                 - 0.0509999385892185 z_k1_s1034
                                 - 0.0239999559537641 z_k1_s1035
                                 - 0.031999958438336 z_k1_s1036
                                 - 0.0930001576330275 z_k1_s1037
                                 - 0.0319999460825376 z_k1_s1040
                                 - 0.0359999130929072 z_k1_s1041
                                 - 0.067999964175235 z_k1_s1042
                                 - 0.067999964175235 z_k1_s1043
                                 - 0.136999526353849 z_k1_s1044
                                 - 0.0339999295877224 z_k1_s1045
                                 - 0.0719999311856046 z_k1_s1046
                                 - 0.0779996192016834 z_k1_s1047
                                 - 0.0440000376037704 z_k1_s1050
                                 - 0.0769999850565824 z_k1_s1051
                                 - 0.0689999194922491 z_k1_s1052
                                 - 0.0840000424253741 z_k1_s1053
                                 - 0.0890000834030825 z_k1_s1054
                                 - 0.101000181749583 z_k1_s1055
                                 - 0.0560000067502452 z_k1_s1056
                                 - 0.125999740637998 z_k1_s1057
                                 - 0.171000041293922 z_k1_s1060
                                 - 0.0849998420890564 z_k1_s1061
                                 - 0.0769999122582396 z_k1_s1062
                                 - 0.0409999175700247 z_k1_s1063
                                 - 0.144999448869985 z_k1_s1064
                                 - 0.139999437288307 z_k1_s1065
                                 - 0.085999844405392 z_k1_s1066
                                 - 0.141999619320715 z_k1_s1067
                                 - 0.0530000260220603 z_k1_s1070
                                 - 0.0789998742406622 z_k1_s1071
                                 - 0.0759998544847224 z_k1_s1072
                                 - 0.0319999523303902 z_k1_s1073
                                 - 0.0199999702064939 z_k1_s1074
                                 - 0.0159999761651951 z_k1_s1075
                                 - 0.0219999833771204 z_k1_s1076
                                 - 0.0249999708331059 z_k1_s1077
                                 - 0.38299989146658 z_k1_s1100
                                 - 0.662000310419718 z_k1_s1101
                                 - 0.17100003303594 z_k1_s1102
                                 - 0.393999224328975 z_k1_s1103
                                 - 0.52699982802322 z_k1_s1104
                                 - 0.281999948006706 z_k1_s1105
                                 - 0.2309998621013 z_k1_s1106
                                 - 0.239999894261067 z_k1_s1107
                                 - 0.348999875011676 z_k1_s1110
                                 - 0.401000640802193 z_k1_s1111
                                 - 0.32700049865513 z_k1_s1112
                                 - 0.208000325559235 z_k1_s1113
                                 - 0.314000307207501 z_k1_s1114
                                 - 0.379999805078555 z_k1_s1115
                                 - 1.0090016977293 z_k1_s1116
                                 - 0.433999020936304 z_k1_s1117
                                 - 0.274998658900948 z_k1_s1120
                                 - 0.130999753084558 z_k1_s1121
                                 - 0.194999608946538 z_k1_s1122
                                 - 0.186999498651613 z_k1_s1123
                                 - 0.552998796258882 z_k1_s1124
                                 - 0.156999700944096 z_k1_s1125
                                 - 0.228999256502033 z_k1_s1126
                                 - 0.325999156731612 z_k1_s1127
                                 - 0.186000170588505 z_k1_s1130
                                 - 0.169999423118642 z_k1_s1131
                                 - 0.137999958178595 z_k1_s1132
                                 - 0.308999301139124 z_k1_s1133
                                 - 0.181000571379073 z_k1_s1134
                                 - 0.106999978080503 z_k1_s1135
                                 - 0.221000265054074 z_k1_s1136
                                 - 0.137999958178595 z_k1_s1137
                                 - 0.112999980041714 z_k1_s1140
                                 - 0.106999884575839 z_k1_s1141
                                 - 0.313999165050886 z_k1_s1142
                                 - 0.268998963556521 z_k1_s1143
                                 - 0.254000268390925 z_k1_s1144
                                 - 0.333000667858782 z_k1_s1145
                                 - 0.449999956944857 z_k1_s1146
                                 - 0.217999695595676 z_k1_s1147
                                 - 0.166000490894177 z_k1_s1150
                                 - 0.195000462315482 z_k1_s1151
                                 - 0.177000127640293 z_k1_s1152
                                 - 0.110000017460088 z_k1_s1153
                                 - 0.260999742791751 z_k1_s1154
                                 - 0.0929999284890087 z_k1_s1155
                                 - 0.147000056514835 z_k1_s1156
                                 - 0.311000348000705 z_k1_s1157
                                 - 0.0550000414386929 z_k1_s1160
                                 - 0.143999894112156 z_k1_s1161
                                 - 0.190999739877896 z_k1_s1162
                                 - 0.0550000414386929 z_k1_s1163
                                 - 0.0900000018451158 z_k1_s1164
                                 - 0.229000329199327 z_k1_s1165
                                 - 0.0500000211806254 z_k1_s1166
                                 - 0.0930001772600007 z_k1_s1167
                                 - 0.359000768427777 z_k1_s1170
                                 - 0.237999263007259 z_k1_s1171
                                 - 0.129000012683955 z_k1_s1172
                                 - 0.398000672290322 z_k1_s1173
                                 - 0.199999807602934 z_k1_s1174
                                 - 0.100999961459509 z_k1_s1175
                                 - 0.298999262946181 z_k1_s1176
                                 - 0.229999583343285 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.03867 w_k1_ADM_s000
                                 + 1.02065 w_k1_AXP_s000
                                 + 1.26759 w_k1_DIS_s000
                                 + 1.17657 w_k1_FDX_s000
                                 + 0.833511 w_k1_IBM_s000
                                 + 1.3047 w_k1_INTC_s000 + 1.07836 w_k1_KO_s000
                                 + 0.808153 w_k1_MAT_s000
                                 + 1.30609 w_k1_MSFT_s000 + 1.007 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.03978 w_k1_ADM_s000
                                 + 1.23941 w_k1_AXP_s000
                                 + 1.17183 w_k1_DIS_s000
                                 + 1.08221 w_k1_FDX_s000
                                 + 1.64997 w_k1_IBM_s000
                                 + 1.1632 w_k1_INTC_s000 + 1.27106 w_k1_KO_s000
                                 + 1.68012 w_k1_MAT_s000
                                 + 1.0677 w_k1_MSFT_s000 + 1.05948 w_k1_PG_s000
                                 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.907475 w_k1_ADM_s000
                                 + 1.03856 w_k1_AXP_s000
                                 + 0.878427 w_k1_DIS_s000
                                 + 0.955212 w_k1_FDX_s000
                                 + 0.94974 w_k1_IBM_s000
                                 + 1.18498 w_k1_INTC_s000
                                 + 0.969202 w_k1_KO_s000
                                 + 0.887876 w_k1_MAT_s000
                                 + 1.02436 w_k1_MSFT_s000
                                 + 0.975888 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.798302 w_k1_ADM_s000
                                 + 0.685466 w_k1_AXP_s000
                                 + 0.588922 w_k1_DIS_s000
                                 + 0.783475 w_k1_FDX_s000
                                 + 0.699121 w_k1_IBM_s000
                                 + 0.709793 w_k1_INTC_s000
                                 + 0.755686 w_k1_KO_s000
                                 + 0.711355 w_k1_MAT_s000
                                 + 0.778675 w_k1_MSFT_s000
                                 + 0.86027 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.884262 w_k1_ADM_s000
                                 + 0.980869 w_k1_AXP_s000
                                 + 0.935465 w_k1_DIS_s000
                                 + 0.930298 w_k1_FDX_s000
                                 + 1.21883 w_k1_IBM_s000
                                 + 1.05771 w_k1_INTC_s000
                                 + 0.948272 w_k1_KO_s000
                                 + 1.07979 w_k1_MAT_s000
                                 + 0.959638 w_k1_MSFT_s000
                                 + 0.888991 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.11414 w_k1_ADM_s000
                                 + 1.00269 w_k1_AXP_s000
                                 + 0.887311 w_k1_DIS_s000
                                 + 1.09751 w_k1_FDX_s000
                                 + 0.78595 w_k1_IBM_s000
                                 + 1.10725 w_k1_INTC_s000
                                 + 0.949226 w_k1_KO_s000
                                 + 0.721698 w_k1_MAT_s000
                                 + 1.08655 w_k1_MSFT_s000 + 1.073 w_k1_PG_s000
                                 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.824854 w_k1_ADM_s000
                                 + 0.627782 w_k1_AXP_s000
                                 + 0.86515 w_k1_DIS_s000
                                 + 0.815453 w_k1_FDX_s000
                                 + 0.981408 w_k1_IBM_s000
                                 + 0.977849 w_k1_INTC_s000
                                 + 0.972848 w_k1_KO_s000
                                 + 0.96215 w_k1_MAT_s000
                                 + 0.972933 w_k1_MSFT_s000
                                 + 0.984052 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.952341 w_k1_ADM_s000
                                 + 1.27523 w_k1_AXP_s000 + 1.399 w_k1_DIS_s000
                                 + 1.26843 w_k1_FDX_s000
                                 + 1.15049 w_k1_IBM_s000
                                 + 1.43599 w_k1_INTC_s000
                                 + 1.18108 w_k1_KO_s000 + 1.01976 w_k1_MAT_s000
                                 + 1.20454 w_k1_MSFT_s000
                                 + 1.05897 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.16927 w_k1_ADM_s001
                                 + 0.953808 w_k1_AXP_s001
                                 + 1.02601 w_k1_DIS_s001
                                 + 1.05539 w_k1_FDX_s001
                                 + 1.13965 w_k1_IBM_s001
                                 + 1.0243 w_k1_INTC_s001 + 1.09076 w_k1_KO_s001
                                 + 1.17886 w_k1_MAT_s001
                                 + 0.930501 w_k1_MSFT_s001
                                 + 1.02229 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.682289 w_k1_ADM_s001
                                 + 0.656632 w_k1_AXP_s001
                                 + 0.606961 w_k1_DIS_s001
                                 + 0.730955 w_k1_FDX_s001
                                 + 0.702498 w_k1_IBM_s001
                                 + 0.861204 w_k1_INTC_s001
                                 + 0.705494 w_k1_KO_s001
                                 + 0.666836 w_k1_MAT_s001
                                 + 0.798536 w_k1_MSFT_s001
                                 + 0.801694 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.42661 w_k1_ADM_s001
                                 + 1.2372 w_k1_AXP_s001 + 1.58708 w_k1_DIS_s001
                                 + 1.3695 w_k1_FDX_s001
                                 + 0.894778 w_k1_IBM_s001
                                 + 1.15246 w_k1_INTC_s001
                                 + 1.16094 w_k1_KO_s001
                                 + 0.948966 w_k1_MAT_s001
                                 + 1.24861 w_k1_MSFT_s001
                                 + 1.18959 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.51417 w_k1_ADM_s001
                                 + 1.48334 w_k1_AXP_s001
                                 + 1.25118 w_k1_DIS_s001
                                 + 1.30437 w_k1_FDX_s001
                                 + 1.28094 w_k1_IBM_s001
                                 + 1.27047 w_k1_INTC_s001
                                 + 1.28608 w_k1_KO_s001 + 1.63612 w_k1_MAT_s001
                                 + 1.32043 w_k1_MSFT_s001
                                 + 1.22634 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.972457 w_k1_ADM_s001
                                 + 1.20805 w_k1_AXP_s001 + 1.3049 w_k1_DIS_s001
                                 + 1.04955 w_k1_FDX_s001
                                 + 1.69197 w_k1_IBM_s001
                                 + 1.04711 w_k1_INTC_s001
                                 + 1.49377 w_k1_KO_s001 + 1.61906 w_k1_MAT_s001
                                 + 1.09236 w_k1_MSFT_s001
                                 + 1.13404 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.10198 w_k1_ADM_s001
                                 + 0.945675 w_k1_AXP_s001
                                 + 1.06779 w_k1_DIS_s001
                                 + 1.14524 w_k1_FDX_s001
                                 + 0.796658 w_k1_IBM_s001
                                 + 0.971098 w_k1_INTC_s001
                                 + 0.924158 w_k1_KO_s001
                                 + 0.944493 w_k1_MAT_s001
                                 + 1.10891 w_k1_MSFT_s001
                                 + 1.13552 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.916991 w_k1_ADM_s001
                                 + 0.72334 w_k1_AXP_s001
                                 + 0.797017 w_k1_DIS_s001
                                 + 0.728784 w_k1_FDX_s001
                                 + 1.31093 w_k1_IBM_s001
                                 + 0.893426 w_k1_INTC_s001
                                 + 1.02917 w_k1_KO_s001
                                 + 0.967021 w_k1_MAT_s001
                                 + 0.824985 w_k1_MSFT_s001
                                 + 0.952832 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.855621 w_k1_ADM_s001
                                 + 0.848767 w_k1_AXP_s001
                                 + 0.846853 w_k1_DIS_s001
                                 + 0.903419 w_k1_FDX_s001
                                 + 0.897635 w_k1_IBM_s001
                                 + 0.832193 w_k1_INTC_s001
                                 + 0.810385 w_k1_KO_s001
                                 + 0.798224 w_k1_MAT_s001
                                 + 0.898612 w_k1_MSFT_s001
                                 + 0.950805 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.891429 w_k1_ADM_s002
                                 + 0.753 w_k1_AXP_s002 + 0.695774 w_k1_DIS_s002
                                 + 0.84986 w_k1_FDX_s002
                                 + 0.752306 w_k1_IBM_s002
                                 + 0.892951 w_k1_INTC_s002
                                 + 0.64251 w_k1_KO_s002
                                 + 0.640075 w_k1_MAT_s002
                                 + 0.819599 w_k1_MSFT_s002
                                 + 0.79814 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.943299 w_k1_ADM_s002
                                 + 1.27287 w_k1_AXP_s002
                                 + 1.52312 w_k1_DIS_s002
                                 + 1.20505 w_k1_FDX_s002
                                 + 1.32092 w_k1_IBM_s002
                                 + 1.24461 w_k1_INTC_s002
                                 + 1.19208 w_k1_KO_s002 + 1.48159 w_k1_MAT_s002
                                 + 1.29381 w_k1_MSFT_s002
                                 + 1.08843 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.694545 w_k1_ADM_s002
                                 + 0.725649 w_k1_AXP_s002
                                 + 0.547543 w_k1_DIS_s002
                                 + 0.712166 w_k1_FDX_s002
                                 + 0.872536 w_k1_IBM_s002
                                 + 0.672218 w_k1_INTC_s002
                                 + 0.514589 w_k1_KO_s002
                                 + 0.684973 w_k1_MAT_s002
                                 + 0.663196 w_k1_MSFT_s002
                                 + 0.754628 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.0025 w_k1_ADM_s002
                                 + 1.43771 w_k1_AXP_s002
                                 + 1.88854 w_k1_DIS_s002
                                 + 1.42572 w_k1_FDX_s002
                                 + 1.01777 w_k1_IBM_s002
                                 + 1.24735 w_k1_INTC_s002
                                 + 1.26687 w_k1_KO_s002 + 1.15689 w_k1_MAT_s002
                                 + 1.41072 w_k1_MSFT_s002
                                 + 1.11168 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.0906 w_k1_ADM_s002
                                 + 1.31076 w_k1_AXP_s002
                                 + 1.16736 w_k1_DIS_s002
                                 + 1.12105 w_k1_FDX_s002
                                 + 1.01673 w_k1_IBM_s002
                                 + 1.10208 w_k1_INTC_s002
                                 + 0.973534 w_k1_KO_s002
                                 + 0.928851 w_k1_MAT_s002
                                 + 1.00198 w_k1_MSFT_s002
                                 + 0.985497 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.07023 w_k1_ADM_s002
                                 + 0.91631 w_k1_AXP_s002
                                 + 1.04923 w_k1_DIS_s002
                                 + 1.01717 w_k1_FDX_s002
                                 + 1.07807 w_k1_IBM_s002
                                 + 0.889016 w_k1_INTC_s002
                                 + 0.866375 w_k1_KO_s002
                                 + 0.793434 w_k1_MAT_s002
                                 + 0.977552 w_k1_MSFT_s002
                                 + 0.884827 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.972882 w_k1_ADM_s002
                                 + 1.2737 w_k1_AXP_s002
                                 + 0.798232 w_k1_DIS_s002
                                 + 0.898775 w_k1_FDX_s002
                                 + 1.30442 w_k1_IBM_s002
                                 + 0.92505 w_k1_INTC_s002
                                 + 0.893987 w_k1_KO_s002
                                 + 1.43425 w_k1_MAT_s002
                                 + 0.827786 w_k1_MSFT_s002
                                 + 0.921117 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.09309 w_k1_ADM_s002
                                 + 0.94704 w_k1_AXP_s002
                                 + 0.848437 w_k1_DIS_s002
                                 + 1.0351 w_k1_FDX_s002
                                 + 0.688822 w_k1_IBM_s002
                                 + 0.870808 w_k1_INTC_s002
                                 + 0.726614 w_k1_KO_s002
                                 + 0.953677 w_k1_MAT_s002
                                 + 0.987031 w_k1_MSFT_s002
                                 + 0.984524 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.981265 w_k1_ADM_s003
                                 + 0.770603 w_k1_AXP_s003
                                 + 1.04627 w_k1_DIS_s003
                                 + 1.00903 w_k1_FDX_s003
                                 + 0.957547 w_k1_IBM_s003
                                 + 1.15602 w_k1_INTC_s003
                                 + 0.887811 w_k1_KO_s003
                                 + 0.814311 w_k1_MAT_s003
                                 + 1.14593 w_k1_MSFT_s003
                                 + 0.999349 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.05558 w_k1_ADM_s003
                                 + 1.13178 w_k1_AXP_s003
                                 + 1.17574 w_k1_DIS_s003
                                 + 1.08846 w_k1_FDX_s003
                                 + 1.51266 w_k1_IBM_s003
                                 + 1.03883 w_k1_INTC_s003
                                 + 1.38494 w_k1_KO_s003 + 1.91414 w_k1_MAT_s003
                                 + 1.08537 w_k1_MSFT_s003
                                 + 1.08747 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.759189 w_k1_ADM_s003
                                 + 0.909114 w_k1_AXP_s003
                                 + 0.810032 w_k1_DIS_s003
                                 + 1.0041 w_k1_FDX_s003 + 0.74923 w_k1_IBM_s003
                                 + 0.962758 w_k1_INTC_s003
                                 + 0.812799 w_k1_KO_s003
                                 + 0.90982 w_k1_MAT_s003
                                 + 1.0397 w_k1_MSFT_s003
                                 + 0.834243 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.02789 w_k1_ADM_s003
                                 + 1.67224 w_k1_AXP_s003
                                 + 1.19238 w_k1_DIS_s003
                                 + 1.25665 w_k1_FDX_s003
                                 + 1.14838 w_k1_IBM_s003
                                 + 0.986994 w_k1_INTC_s003
                                 + 1.04389 w_k1_KO_s003 + 1.45977 w_k1_MAT_s003
                                 + 1.15537 w_k1_MSFT_s003
                                 + 1.04831 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.00547 w_k1_ADM_s003
                                 + 0.677588 w_k1_AXP_s003
                                 + 0.77396 w_k1_DIS_s003
                                 + 0.90328 w_k1_FDX_s003
                                 + 0.61435 w_k1_IBM_s003
                                 + 0.964463 w_k1_INTC_s003
                                 + 0.804904 w_k1_KO_s003
                                 + 0.571017 w_k1_MAT_s003
                                 + 0.943294 w_k1_MSFT_s003
                                 + 0.829001 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.24354 w_k1_ADM_s003
                                 + 1.48493 w_k1_AXP_s003
                                 + 1.66125 w_k1_DIS_s003
                                 + 1.44602 w_k1_FDX_s003
                                 + 1.15183 w_k1_IBM_s003
                                 + 1.17288 w_k1_INTC_s003
                                 + 1.44234 w_k1_KO_s003 + 1.80284 w_k1_MAT_s003
                                 + 1.54169 w_k1_MSFT_s003
                                 + 1.23568 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.747596 w_k1_ADM_s003
                                 + 0.895597 w_k1_AXP_s003
                                 + 0.651824 w_k1_DIS_s003
                                 + 0.742753 w_k1_FDX_s003
                                 + 1.00736 w_k1_IBM_s003
                                 + 0.794522 w_k1_INTC_s003
                                 + 0.718264 w_k1_KO_s003
                                 + 1.06014 w_k1_MAT_s003
                                 + 0.823982 w_k1_MSFT_s003
                                 + 0.799582 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.12877 w_k1_ADM_s003
                                 + 1.29742 w_k1_AXP_s003
                                 + 0.914387 w_k1_DIS_s003
                                 + 1.24073 w_k1_FDX_s003
                                 + 0.965323 w_k1_IBM_s003
                                 + 1.1565 w_k1_INTC_s003
                                 + 0.916226 w_k1_KO_s003
                                 + 1.29523 w_k1_MAT_s003
                                 + 1.2562 w_k1_MSFT_s003 + 1.0122 w_k1_PG_s003
                                 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.0248 w_k1_ADM_s004
                                 + 0.916307 w_k1_AXP_s004
                                 + 0.81133 w_k1_DIS_s004
                                 + 1.02505 w_k1_FDX_s004
                                 + 0.61693 w_k1_IBM_s004
                                 + 1.19782 w_k1_INTC_s004
                                 + 0.714039 w_k1_KO_s004
                                 + 0.588081 w_k1_MAT_s004
                                 + 0.84314 w_k1_MSFT_s004
                                 + 0.878581 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.782353 w_k1_ADM_s004
                                 + 0.710265 w_k1_AXP_s004
                                 + 0.653362 w_k1_DIS_s004
                                 + 0.732253 w_k1_FDX_s004
                                 + 0.752014 w_k1_IBM_s004
                                 + 0.861593 w_k1_INTC_s004
                                 + 0.71803 w_k1_KO_s004
                                 + 0.782463 w_k1_MAT_s004
                                 + 0.690026 w_k1_MSFT_s004
                                 + 0.776648 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.77653 w_k1_ADM_s004
                                 + 0.913234 w_k1_AXP_s004
                                 + 0.91156 w_k1_DIS_s004
                                 + 0.950192 w_k1_FDX_s004
                                 + 0.927616 w_k1_IBM_s004
                                 + 1.08742 w_k1_INTC_s004
                                 + 0.890184 w_k1_KO_s004 + 1.1026 w_k1_MAT_s004
                                 + 0.858204 w_k1_MSFT_s004
                                 + 0.863219 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.927573 w_k1_ADM_s004
                                 + 0.581562 w_k1_AXP_s004
                                 + 0.971032 w_k1_DIS_s004
                                 + 0.944542 w_k1_FDX_s004
                                 + 0.8366 w_k1_IBM_s004
                                 + 0.989052 w_k1_INTC_s004
                                 + 0.938187 w_k1_KO_s004
                                 + 0.534936 w_k1_MAT_s004
                                 + 0.890579 w_k1_MSFT_s004
                                 + 1.01307 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.80632 w_k1_ADM_s004
                                 + 0.852303 w_k1_AXP_s004
                                 + 0.972511 w_k1_DIS_s004
                                 + 0.955481 w_k1_FDX_s004
                                 + 1.21247 w_k1_IBM_s004
                                 + 1.02114 w_k1_INTC_s004
                                 + 0.943011 w_k1_KO_s004 + 1.1724 w_k1_MAT_s004
                                 + 0.843829 w_k1_MSFT_s004
                                 + 0.97156 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.10034 w_k1_ADM_s004
                                 + 1.31905 w_k1_AXP_s004
                                 + 1.43116 w_k1_DIS_s004
                                 + 1.22956 w_k1_FDX_s004
                                 + 1.35737 w_k1_IBM_s004
                                 + 1.18285 w_k1_INTC_s004
                                 + 1.16891 w_k1_KO_s004 + 1.49093 w_k1_MAT_s004
                                 + 0.992114 w_k1_MSFT_s004
                                 + 1.14848 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.09022 w_k1_ADM_s004
                                 + 1.12781 w_k1_AXP_s004
                                 + 1.38348 w_k1_DIS_s004
                                 + 1.27088 w_k1_FDX_s004
                                 + 0.838453 w_k1_IBM_s004
                                 + 1.1556 w_k1_INTC_s004
                                 + 0.969472 w_k1_KO_s004
                                 + 0.761903 w_k1_MAT_s004
                                 + 1.04176 w_k1_MSFT_s004 + 1.032 w_k1_PG_s004
                                 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.822998 w_k1_ADM_s004
                                 + 0.917636 w_k1_AXP_s004
                                 + 0.929977 w_k1_DIS_s004
                                 + 0.895878 w_k1_FDX_s004
                                 + 1.43167 w_k1_IBM_s004
                                 + 1.01554 w_k1_INTC_s004
                                 + 1.24602 w_k1_KO_s004 + 1.47141 w_k1_MAT_s004
                                 + 0.835007 w_k1_MSFT_s004
                                 + 1.01407 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.48339 w_k1_ADM_s005
                                 + 1.19909 w_k1_AXP_s005
                                 + 1.42175 w_k1_DIS_s005
                                 + 1.33287 w_k1_FDX_s005
                                 + 1.16877 w_k1_IBM_s005
                                 + 1.19848 w_k1_INTC_s005
                                 + 1.11071 w_k1_KO_s005
                                 + 0.771177 w_k1_MAT_s005
                                 + 1.37081 w_k1_MSFT_s005
                                 + 1.07758 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.99287 w_k1_ADM_s005
                                 + 0.99319 w_k1_AXP_s005 + 1.242 w_k1_DIS_s005
                                 + 1.08456 w_k1_FDX_s005
                                 + 1.09752 w_k1_IBM_s005
                                 + 1.21187 w_k1_INTC_s005
                                 + 1.23661 w_k1_KO_s005
                                 + 0.951264 w_k1_MAT_s005
                                 + 1.35485 w_k1_MSFT_s005
                                 + 1.15322 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.878756 w_k1_ADM_s005
                                 + 0.965552 w_k1_AXP_s005
                                 + 0.910311 w_k1_DIS_s005
                                 + 1.00603 w_k1_FDX_s005
                                 + 1.04084 w_k1_IBM_s005
                                 + 1.09618 w_k1_INTC_s005
                                 + 1.02973 w_k1_KO_s005
                                 + 0.905936 w_k1_MAT_s005
                                 + 1.09709 w_k1_MSFT_s005
                                 + 0.922335 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.758604 w_k1_ADM_s005
                                 + 0.745039 w_k1_AXP_s005
                                 + 0.655106 w_k1_DIS_s005
                                 + 0.774471 w_k1_FDX_s005
                                 + 0.71943 w_k1_IBM_s005
                                 + 0.716388 w_k1_INTC_s005
                                 + 0.713142 w_k1_KO_s005
                                 + 0.68014 w_k1_MAT_s005
                                 + 0.698334 w_k1_MSFT_s005
                                 + 0.746171 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.22169 w_k1_ADM_s005
                                 + 1.26845 w_k1_AXP_s005
                                 + 2.07031 w_k1_DIS_s005
                                 + 1.46152 w_k1_FDX_s005
                                 + 1.47302 w_k1_IBM_s005 + 1.378 w_k1_INTC_s005
                                 + 1.37459 w_k1_KO_s005 + 1.1376 w_k1_MAT_s005
                                 + 1.62239 w_k1_MSFT_s005
                                 + 1.30275 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.817196 w_k1_ADM_s005
                                 + 0.918548 w_k1_AXP_s005
                                 + 0.79229 w_k1_DIS_s005
                                 + 1.01358 w_k1_FDX_s005 + 0.6756 w_k1_IBM_s005
                                 + 1.10638 w_k1_INTC_s005
                                 + 0.670629 w_k1_KO_s005
                                 + 0.760001 w_k1_MAT_s005
                                 + 1.05713 w_k1_MSFT_s005
                                 + 0.853589 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.900169 w_k1_ADM_s005
                                 + 0.994178 w_k1_AXP_s005
                                 + 0.77049 w_k1_DIS_s005
                                 + 0.82318 w_k1_FDX_s005
                                 + 1.28796 w_k1_IBM_s005
                                 + 0.796461 w_k1_INTC_s005
                                 + 0.839177 w_k1_KO_s005
                                 + 1.31652 w_k1_MAT_s005
                                 + 0.833479 w_k1_MSFT_s005
                                 + 0.892855 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.886929 w_k1_ADM_s005
                                 + 1.54763 w_k1_AXP_s005
                                 + 1.11202 w_k1_DIS_s005
                                 + 1.22334 w_k1_FDX_s005 + 1.4514 w_k1_IBM_s005
                                 + 1.13243 w_k1_INTC_s005
                                 + 1.17768 w_k1_KO_s005 + 1.86505 w_k1_MAT_s005
                                 + 1.14611 w_k1_MSFT_s005
                                 + 1.18907 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.893038 w_k1_ADM_s006
                                 + 0.821998 w_k1_AXP_s006
                                 + 0.903619 w_k1_DIS_s006
                                 + 0.837094 w_k1_FDX_s006
                                 + 1.42171 w_k1_IBM_s006
                                 + 0.989923 w_k1_INTC_s006
                                 + 0.991159 w_k1_KO_s006
                                 + 1.10628 w_k1_MAT_s006
                                 + 0.835626 w_k1_MSFT_s006
                                 + 1.12368 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.111 w_k1_ADM_s006
                                 + 1.01353 w_k1_AXP_s006 + 1.5185 w_k1_DIS_s006
                                 + 1.18036 w_k1_FDX_s006 + 1.0745 w_k1_IBM_s006
                                 + 1.00595 w_k1_INTC_s006
                                 + 1.19601 w_k1_KO_s006 + 1.12139 w_k1_MAT_s006
                                 + 1.2062 w_k1_MSFT_s006 + 1.16581 w_k1_PG_s006
                                 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.947656 w_k1_ADM_s006
                                 + 0.941067 w_k1_AXP_s006
                                 + 1.02164 w_k1_DIS_s006 + 1.0148 w_k1_FDX_s006
                                 + 0.92876 w_k1_IBM_s006
                                 + 1.04512 w_k1_INTC_s006
                                 + 0.874396 w_k1_KO_s006
                                 + 0.926076 w_k1_MAT_s006
                                 + 0.982985 w_k1_MSFT_s006
                                 + 1.02378 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.840835 w_k1_ADM_s006
                                 + 0.763434 w_k1_AXP_s006
                                 + 0.758999 w_k1_DIS_s006
                                 + 0.874209 w_k1_FDX_s006
                                 + 0.707349 w_k1_IBM_s006
                                 + 0.869605 w_k1_INTC_s006
                                 + 0.68519 w_k1_KO_s006
                                 + 0.456591 w_k1_MAT_s006
                                 + 0.773103 w_k1_MSFT_s006
                                 + 0.894364 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.45401 w_k1_ADM_s006
                                 + 1.77612 w_k1_AXP_s006
                                 + 1.62151 w_k1_DIS_s006 + 1.3207 w_k1_FDX_s006
                                 + 1.30328 w_k1_IBM_s006
                                 + 1.06138 w_k1_INTC_s006
                                 + 1.24778 w_k1_KO_s006 + 1.57941 w_k1_MAT_s006
                                 + 1.08538 w_k1_MSFT_s006
                                 + 1.41109 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.838635 w_k1_ADM_s006
                                 + 0.776929 w_k1_AXP_s006
                                 + 0.684811 w_k1_DIS_s006
                                 + 0.706108 w_k1_FDX_s006
                                 + 0.898256 w_k1_IBM_s006
                                 + 0.795743 w_k1_INTC_s006
                                 + 0.701867 w_k1_KO_s006
                                 + 0.891248 w_k1_MAT_s006
                                 + 0.702952 w_k1_MSFT_s006
                                 + 0.819099 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.912368 w_k1_ADM_s006
                                 + 0.672432 w_k1_AXP_s006
                                 + 0.960009 w_k1_DIS_s006
                                 + 0.854358 w_k1_FDX_s006
                                 + 1.05894 w_k1_IBM_s006
                                 + 0.823926 w_k1_INTC_s006
                                 + 1.10603 w_k1_KO_s006
                                 + 0.911713 w_k1_MAT_s006
                                 + 0.926775 w_k1_MSFT_s006
                                 + 1.01548 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.897519 w_k1_ADM_s006
                                 + 1.07617 w_k1_AXP_s006
                                 + 0.859167 w_k1_DIS_s006
                                 + 0.937274 w_k1_FDX_s006
                                 + 1.11081 w_k1_IBM_s006
                                 + 0.841617 w_k1_INTC_s006
                                 + 0.96311 w_k1_KO_s006 + 1.33779 w_k1_MAT_s006
                                 + 0.876222 w_k1_MSFT_s006
                                 + 1.04426 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.566787 w_k1_ADM_s007
                                 + 0.866569 w_k1_AXP_s007
                                 + 0.999798 w_k1_DIS_s007
                                 + 0.879891 w_k1_FDX_s007
                                 + 1.41759 w_k1_IBM_s007
                                 + 0.960234 w_k1_INTC_s007
                                 + 1.1385 w_k1_KO_s007 + 1.36268 w_k1_MAT_s007
                                 + 0.960528 w_k1_MSFT_s007
                                 + 0.90481 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.522275 w_k1_ADM_s007
                                 + 0.487205 w_k1_AXP_s007
                                 + 0.511989 w_k1_DIS_s007
                                 + 0.607127 w_k1_FDX_s007
                                 + 0.898997 w_k1_IBM_s007
                                 + 0.615684 w_k1_INTC_s007
                                 + 0.715832 w_k1_KO_s007
                                 + 0.59071 w_k1_MAT_s007
                                 + 0.65477 w_k1_MSFT_s007
                                 + 0.65093 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.893041 w_k1_ADM_s007
                                 + 0.995626 w_k1_AXP_s007
                                 + 1.49399 w_k1_DIS_s007
                                 + 1.25772 w_k1_FDX_s007
                                 + 1.10217 w_k1_IBM_s007
                                 + 1.23982 w_k1_INTC_s007
                                 + 1.32994 w_k1_KO_s007 + 1.10552 w_k1_MAT_s007
                                 + 1.29057 w_k1_MSFT_s007
                                 + 0.963634 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.740924 w_k1_ADM_s007
                                 + 1.00789 w_k1_AXP_s007
                                 + 0.848421 w_k1_DIS_s007
                                 + 0.947327 w_k1_FDX_s007
                                 + 1.08201 w_k1_IBM_s007
                                 + 0.978702 w_k1_INTC_s007
                                 + 0.998805 w_k1_KO_s007
                                 + 1.04037 w_k1_MAT_s007
                                 + 1.02539 w_k1_MSFT_s007
                                 + 0.767936 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.594525 w_k1_ADM_s007
                                 + 0.738705 w_k1_AXP_s007
                                 + 0.769844 w_k1_DIS_s007
                                 + 0.817913 w_k1_FDX_s007
                                 + 0.699495 w_k1_IBM_s007
                                 + 1.00953 w_k1_INTC_s007
                                 + 0.797933 w_k1_KO_s007
                                 + 0.529945 w_k1_MAT_s007
                                 + 0.952009 w_k1_MSFT_s007
                                 + 0.739782 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.922927 w_k1_ADM_s007
                                 + 0.92989 w_k1_AXP_s007
                                 + 1.09795 w_k1_DIS_s007
                                 + 1.14898 w_k1_FDX_s007
                                 + 0.751005 w_k1_IBM_s007
                                 + 1.0724 w_k1_INTC_s007
                                 + 0.870559 w_k1_KO_s007
                                 + 0.633137 w_k1_MAT_s007
                                 + 1.06534 w_k1_MSFT_s007
                                 + 0.83202 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.30665 w_k1_ADM_s007
                                 + 1.53861 w_k1_AXP_s007
                                 + 1.84126 w_k1_DIS_s007
                                 + 1.43145 w_k1_FDX_s007
                                 + 1.26239 w_k1_IBM_s007
                                 + 1.47753 w_k1_INTC_s007
                                 + 1.43748 w_k1_KO_s007 + 1.3397 w_k1_MAT_s007
                                 + 1.59009 w_k1_MSFT_s007 + 1.1597 w_k1_PG_s007
                                 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.644913 w_k1_ADM_s007
                                 + 0.588815 w_k1_AXP_s007
                                 + 0.67413 w_k1_DIS_s007
                                 + 0.737576 w_k1_FDX_s007
                                 + 0.98052 w_k1_IBM_s007
                                 + 0.841711 w_k1_INTC_s007
                                 + 0.869604 w_k1_KO_s007
                                 + 0.692681 w_k1_MAT_s007
                                 + 0.972507 w_k1_MSFT_s007
                                 + 0.719986 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.944133 w_k1_ADM_s010
                                 + 0.899702 w_k1_AXP_s010
                                 + 0.841248 w_k1_DIS_s010
                                 + 0.918281 w_k1_FDX_s010
                                 + 0.979811 w_k1_IBM_s010
                                 + 0.892017 w_k1_INTC_s010
                                 + 0.903494 w_k1_KO_s010
                                 + 1.12573 w_k1_MAT_s010
                                 + 0.93132 w_k1_MSFT_s010
                                 + 0.98773 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.880041 w_k1_ADM_s010
                                 + 0.915382 w_k1_AXP_s010
                                 + 0.87005 w_k1_DIS_s010
                                 + 0.89394 w_k1_FDX_s010
                                 + 1.28032 w_k1_IBM_s010
                                 + 0.966535 w_k1_INTC_s010
                                 + 1.05928 w_k1_KO_s010 + 1.40531 w_k1_MAT_s010
                                 + 0.940445 w_k1_MSFT_s010
                                 + 0.865314 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.03602 w_k1_ADM_s010
                                 + 1.1371 w_k1_AXP_s010 + 1.20053 w_k1_DIS_s010
                                 + 1.15995 w_k1_FDX_s010
                                 + 0.986592 w_k1_IBM_s010
                                 + 1.2416 w_k1_INTC_s010
                                 + 0.995004 w_k1_KO_s010
                                 + 0.892552 w_k1_MAT_s010
                                 + 1.21011 w_k1_MSFT_s010
                                 + 0.998049 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.16506 w_k1_ADM_s010
                                 + 0.753281 w_k1_AXP_s010
                                 + 1.0637 w_k1_DIS_s010 + 1.10406 w_k1_FDX_s010
                                 + 0.780201 w_k1_IBM_s010
                                 + 0.957016 w_k1_INTC_s010
                                 + 0.900117 w_k1_KO_s010
                                 + 0.73004 w_k1_MAT_s010
                                 + 1.08069 w_k1_MSFT_s010
                                 + 0.958155 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.883543 w_k1_ADM_s010
                                 + 0.636197 w_k1_AXP_s010
                                 + 0.672274 w_k1_DIS_s010
                                 + 0.730632 w_k1_FDX_s010
                                 + 0.93065 w_k1_IBM_s010
                                 + 0.746157 w_k1_INTC_s010
                                 + 0.738877 w_k1_KO_s010
                                 + 0.923467 w_k1_MAT_s010
                                 + 0.778569 w_k1_MSFT_s010
                                 + 0.787529 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.811968 w_k1_ADM_s010
                                 + 0.953162 w_k1_AXP_s010
                                 + 0.579561 w_k1_DIS_s010
                                 + 0.763931 w_k1_FDX_s010
                                 + 0.811309 w_k1_IBM_s010
                                 + 0.833684 w_k1_INTC_s010
                                 + 0.719968 w_k1_KO_s010
                                 + 0.934663 w_k1_MAT_s010
                                 + 0.758054 w_k1_MSFT_s010
                                 + 0.768677 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.838471 w_k1_ADM_s010
                                 + 0.775179 w_k1_AXP_s010
                                 + 0.805496 w_k1_DIS_s010
                                 + 0.903754 w_k1_FDX_s010
                                 + 0.651307 w_k1_IBM_s010
                                 + 0.740169 w_k1_INTC_s010
                                 + 0.736231 w_k1_KO_s010
                                 + 0.75499 w_k1_MAT_s010
                                 + 0.830126 w_k1_MSFT_s010
                                 + 0.859522 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.54312 w_k1_ADM_s010
                                 + 1.3772 w_k1_AXP_s010 + 1.4416 w_k1_DIS_s010
                                 + 1.23841 w_k1_FDX_s010
                                 + 1.41303 w_k1_IBM_s010
                                 + 1.09108 w_k1_INTC_s010
                                 + 1.27852 w_k1_KO_s010 + 1.80145 w_k1_MAT_s010
                                 + 1.19466 w_k1_MSFT_s010
                                 + 1.20362 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.641216 w_k1_ADM_s011
                                 + 0.559127 w_k1_AXP_s011
                                 + 0.808121 w_k1_DIS_s011
                                 + 0.795635 w_k1_FDX_s011
                                 + 0.73236 w_k1_IBM_s011
                                 + 1.04338 w_k1_INTC_s011
                                 + 0.805488 w_k1_KO_s011
                                 + 0.527126 w_k1_MAT_s011
                                 + 0.995101 w_k1_MSFT_s011
                                 + 0.768977 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.900718 w_k1_ADM_s011
                                 + 0.920803 w_k1_AXP_s011
                                 + 0.926646 w_k1_DIS_s011
                                 + 0.837412 w_k1_FDX_s011
                                 + 0.964698 w_k1_IBM_s011
                                 + 1.0534 w_k1_INTC_s011
                                 + 0.885071 w_k1_KO_s011
                                 + 0.88484 w_k1_MAT_s011
                                 + 1.03482 w_k1_MSFT_s011
                                 + 0.897181 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.21961 w_k1_ADM_s011
                                 + 1.40037 w_k1_AXP_s011
                                 + 1.29172 w_k1_DIS_s011
                                 + 1.08606 w_k1_FDX_s011 + 1.6007 w_k1_IBM_s011
                                 + 1.30853 w_k1_INTC_s011
                                 + 1.27839 w_k1_KO_s011 + 1.59081 w_k1_MAT_s011
                                 + 1.1598 w_k1_MSFT_s011 + 1.13871 w_k1_PG_s011
                                 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.931191 w_k1_ADM_s011
                                 + 1.13664 w_k1_AXP_s011
                                 + 0.960124 w_k1_DIS_s011
                                 + 0.991574 w_k1_FDX_s011
                                 + 1.02914 w_k1_IBM_s011
                                 + 0.956743 w_k1_INTC_s011
                                 + 0.889397 w_k1_KO_s011
                                 + 1.28048 w_k1_MAT_s011
                                 + 0.932327 w_k1_MSFT_s011
                                 + 0.957926 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.832354 w_k1_ADM_s011
                                 + 1.04087 w_k1_AXP_s011
                                 + 0.828836 w_k1_DIS_s011
                                 + 0.947292 w_k1_FDX_s011
                                 + 0.697833 w_k1_IBM_s011
                                 + 0.930862 w_k1_INTC_s011
                                 + 0.664668 w_k1_KO_s011
                                 + 0.667669 w_k1_MAT_s011
                                 + 0.852637 w_k1_MSFT_s011
                                 + 0.859412 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.17988 w_k1_ADM_s011
                                 + 1.47247 w_k1_AXP_s011
                                 + 1.22425 w_k1_DIS_s011
                                 + 1.22846 w_k1_FDX_s011
                                 + 1.18105 w_k1_IBM_s011
                                 + 1.28877 w_k1_INTC_s011
                                 + 1.12515 w_k1_KO_s011 + 1.64617 w_k1_MAT_s011
                                 + 1.19903 w_k1_MSFT_s011
                                 + 1.09504 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.922878 w_k1_ADM_s011
                                 + 0.772695 w_k1_AXP_s011
                                 + 0.544252 w_k1_DIS_s011
                                 + 0.712308 w_k1_FDX_s011
                                 + 0.68846 w_k1_IBM_s011
                                 + 0.629255 w_k1_INTC_s011
                                 + 0.635305 w_k1_KO_s011
                                 + 0.733332 w_k1_MAT_s011
                                 + 0.685662 w_k1_MSFT_s011
                                 + 0.855005 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.02308 w_k1_ADM_s011
                                 + 1.20032 w_k1_AXP_s011
                                 + 1.50357 w_k1_DIS_s011
                                 + 1.15567 w_k1_FDX_s011
                                 + 0.865458 w_k1_IBM_s011
                                 + 1.24023 w_k1_INTC_s011
                                 + 1.11149 w_k1_KO_s011 + 1.04731 w_k1_MAT_s011
                                 + 1.28647 w_k1_MSFT_s011 + 1.1999 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.45921 w_k1_ADM_s012
                                 + 1.31104 w_k1_AXP_s012
                                 + 1.30531 w_k1_DIS_s012
                                 + 1.19818 w_k1_FDX_s012
                                 + 1.68944 w_k1_IBM_s012
                                 + 1.15167 w_k1_INTC_s012
                                 + 1.11456 w_k1_KO_s012 + 1.50494 w_k1_MAT_s012
                                 + 1.17083 w_k1_MSFT_s012 + 1.215 w_k1_PG_s012
                                 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.09291 w_k1_ADM_s012
                                 + 1.00506 w_k1_AXP_s012
                                 + 1.03427 w_k1_DIS_s012
                                 + 1.09966 w_k1_FDX_s012
                                 + 0.721339 w_k1_IBM_s012
                                 + 0.9788 w_k1_INTC_s012
                                 + 0.773697 w_k1_KO_s012
                                 + 0.713341 w_k1_MAT_s012
                                 + 1.01138 w_k1_MSFT_s012
                                 + 1.05446 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.08949 w_k1_ADM_s012
                                 + 1.15395 w_k1_AXP_s012 + 1.2206 w_k1_DIS_s012
                                 + 1.05406 w_k1_FDX_s012
                                 + 1.26899 w_k1_IBM_s012
                                 + 1.01557 w_k1_INTC_s012
                                 + 1.13226 w_k1_KO_s012 + 1.57288 w_k1_MAT_s012
                                 + 1.126 w_k1_MSFT_s012 + 1.09985 w_k1_PG_s012
                                 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.11555 w_k1_ADM_s012
                                 + 0.865685 w_k1_AXP_s012
                                 + 1.03351 w_k1_DIS_s012
                                 + 0.922612 w_k1_FDX_s012
                                 + 1.05365 w_k1_IBM_s012
                                 + 0.88402 w_k1_INTC_s012
                                 + 0.958603 w_k1_KO_s012
                                 + 1.01737 w_k1_MAT_s012
                                 + 0.964166 w_k1_MSFT_s012
                                 + 1.03154 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.11819 w_k1_ADM_s012
                                 + 1.01372 w_k1_AXP_s012
                                 + 0.890778 w_k1_DIS_s012
                                 + 1.03369 w_k1_FDX_s012
                                 + 1.06336 w_k1_IBM_s012
                                 + 1.04896 w_k1_INTC_s012
                                 + 0.96698 w_k1_KO_s012 + 1.03792 w_k1_MAT_s012
                                 + 1.002 w_k1_MSFT_s012 + 1.09948 w_k1_PG_s012
                                 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.26409 w_k1_ADM_s012
                                 + 1.37358 w_k1_AXP_s012
                                 + 1.62863 w_k1_DIS_s012
                                 + 1.38967 w_k1_FDX_s012
                                 + 1.16688 w_k1_IBM_s012
                                 + 1.24287 w_k1_INTC_s012
                                 + 1.15917 w_k1_KO_s012 + 1.76937 w_k1_MAT_s012
                                 + 1.35065 w_k1_MSFT_s012
                                 + 1.29471 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.984079 w_k1_ADM_s012
                                 + 0.738048 w_k1_AXP_s012
                                 + 0.665751 w_k1_DIS_s012
                                 + 0.832365 w_k1_FDX_s012
                                 + 0.771067 w_k1_IBM_s012
                                 + 0.868371 w_k1_INTC_s012
                                 + 0.679277 w_k1_KO_s012
                                 + 0.730668 w_k1_MAT_s012
                                 + 0.879173 w_k1_MSFT_s012
                                 + 0.808523 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.914614 w_k1_ADM_s012
                                 + 0.674504 w_k1_AXP_s012
                                 + 0.650276 w_k1_DIS_s012
                                 + 0.741964 w_k1_FDX_s012
                                 + 1.11399 w_k1_IBM_s012
                                 + 0.744847 w_k1_INTC_s012
                                 + 0.957539 w_k1_KO_s012
                                 + 0.819714 w_k1_MAT_s012
                                 + 0.806626 w_k1_MSFT_s012
                                 + 0.890154 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.01803 w_k1_ADM_s013
                                 + 1.03206 w_k1_AXP_s013
                                 + 1.55015 w_k1_DIS_s013
                                 + 1.14014 w_k1_FDX_s013
                                 + 1.10095 w_k1_IBM_s013
                                 + 1.08867 w_k1_INTC_s013
                                 + 1.21844 w_k1_KO_s013 + 1.12147 w_k1_MAT_s013
                                 + 1.24259 w_k1_MSFT_s013
                                 + 1.20981 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.00416 w_k1_ADM_s013
                                 + 1.03694 w_k1_AXP_s013 + 1.0666 w_k1_DIS_s013
                                 + 1.07705 w_k1_FDX_s013
                                 + 0.756211 w_k1_IBM_s013
                                 + 0.957047 w_k1_INTC_s013
                                 + 0.963913 w_k1_KO_s013
                                 + 0.750815 w_k1_MAT_s013
                                 + 1.02393 w_k1_MSFT_s013 + 1.0831 w_k1_PG_s013
                                 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.715794 w_k1_ADM_s013
                                 + 0.658085 w_k1_AXP_s013
                                 + 0.688675 w_k1_DIS_s013
                                 + 0.75908 w_k1_FDX_s013
                                 + 0.699828 w_k1_IBM_s013
                                 + 0.734576 w_k1_INTC_s013
                                 + 0.729351 w_k1_KO_s013
                                 + 0.648326 w_k1_MAT_s013
                                 + 0.736225 w_k1_MSFT_s013
                                 + 0.840492 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.02731 w_k1_ADM_s013
                                 + 1.09738 w_k1_AXP_s013 + 1.1404 w_k1_DIS_s013
                                 + 1.09443 w_k1_FDX_s013
                                 + 0.954992 w_k1_IBM_s013
                                 + 0.895942 w_k1_INTC_s013
                                 + 1.03795 w_k1_KO_s013
                                 + 0.995178 w_k1_MAT_s013
                                 + 0.967632 w_k1_MSFT_s013
                                 + 1.11811 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.940779 w_k1_ADM_s013
                                 + 0.921214 w_k1_AXP_s013
                                 + 0.85466 w_k1_DIS_s013
                                 + 0.738789 w_k1_FDX_s013
                                 + 1.38247 w_k1_IBM_s013
                                 + 0.779072 w_k1_INTC_s013
                                 + 0.993729 w_k1_KO_s013
                                 + 1.13985 w_k1_MAT_s013
                                 + 0.853428 w_k1_MSFT_s013
                                 + 1.00839 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.708501 w_k1_ADM_s013
                                 + 0.592516 w_k1_AXP_s013
                                 + 0.629414 w_k1_DIS_s013
                                 + 0.678981 w_k1_FDX_s013
                                 + 1.0016 w_k1_IBM_s013
                                 + 0.721016 w_k1_INTC_s013
                                 + 0.834623 w_k1_KO_s013
                                 + 0.996252 w_k1_MAT_s013
                                 + 0.808861 w_k1_MSFT_s013
                                 + 0.923158 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.824354 w_k1_ADM_s013
                                 + 0.660824 w_k1_AXP_s013
                                 + 0.923006 w_k1_DIS_s013
                                 + 0.956959 w_k1_FDX_s013
                                 + 0.552915 w_k1_IBM_s013
                                 + 0.704448 w_k1_INTC_s013
                                 + 0.725528 w_k1_KO_s013
                                 + 0.453034 w_k1_MAT_s013
                                 + 0.897376 w_k1_MSFT_s013
                                 + 0.930968 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.03913 w_k1_ADM_s013
                                 + 1.10764 w_k1_AXP_s013
                                 + 1.17541 w_k1_DIS_s013
                                 + 1.07406 w_k1_FDX_s013
                                 + 1.58859 w_k1_IBM_s013
                                 + 1.05498 w_k1_INTC_s013
                                 + 1.23465 w_k1_KO_s013 + 1.49885 w_k1_MAT_s013
                                 + 1.18377 w_k1_MSFT_s013
                                 + 1.13572 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.12245 w_k1_ADM_s014
                                 + 0.870088 w_k1_AXP_s014
                                 + 1.11332 w_k1_DIS_s014
                                 + 1.05527 w_k1_FDX_s014
                                 + 0.932349 w_k1_IBM_s014
                                 + 1.0875 w_k1_INTC_s014 + 1.01504 w_k1_KO_s014
                                 + 1.00258 w_k1_MAT_s014
                                 + 1.11729 w_k1_MSFT_s014
                                 + 0.902501 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.52078 w_k1_ADM_s014
                                 + 1.56951 w_k1_AXP_s014 + 1.4953 w_k1_DIS_s014
                                 + 1.31479 w_k1_FDX_s014
                                 + 1.27609 w_k1_IBM_s014
                                 + 1.48943 w_k1_INTC_s014
                                 + 1.30346 w_k1_KO_s014 + 1.14899 w_k1_MAT_s014
                                 + 1.25693 w_k1_MSFT_s014
                                 + 1.28113 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.10086 w_k1_ADM_s014
                                 + 1.10334 w_k1_AXP_s014
                                 + 1.33785 w_k1_DIS_s014
                                 + 1.18799 w_k1_FDX_s014
                                 + 1.04234 w_k1_IBM_s014
                                 + 1.26201 w_k1_INTC_s014
                                 + 1.04155 w_k1_KO_s014
                                 + 0.903734 w_k1_MAT_s014
                                 + 1.18802 w_k1_MSFT_s014
                                 + 1.13748 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.896876 w_k1_ADM_s014
                                 + 1.00008 w_k1_AXP_s014
                                 + 1.05984 w_k1_DIS_s014
                                 + 0.985118 w_k1_FDX_s014
                                 + 1.60883 w_k1_IBM_s014
                                 + 1.28809 w_k1_INTC_s014 + 1.2183 w_k1_KO_s014
                                 + 1.13906 w_k1_MAT_s014
                                 + 0.987656 w_k1_MSFT_s014
                                 + 1.0846 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.854624 w_k1_ADM_s014
                                 + 0.530571 w_k1_AXP_s014
                                 + 0.710386 w_k1_DIS_s014
                                 + 0.783859 w_k1_FDX_s014
                                 + 0.753862 w_k1_IBM_s014
                                 + 0.897916 w_k1_INTC_s014
                                 + 0.77562 w_k1_KO_s014
                                 + 0.397735 w_k1_MAT_s014
                                 + 0.781348 w_k1_MSFT_s014
                                 + 0.7673 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.56136 w_k1_ADM_s014
                                 + 1.14838 w_k1_AXP_s014
                                 + 1.73464 w_k1_DIS_s014
                                 + 1.51437 w_k1_FDX_s014
                                 + 0.921669 w_k1_IBM_s014
                                 + 1.49108 w_k1_INTC_s014
                                 + 1.20485 w_k1_KO_s014 + 0.59215 w_k1_MAT_s014
                                 + 1.28863 w_k1_MSFT_s014
                                 + 1.06426 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.0116 w_k1_ADM_s014
                                 + 0.865337 w_k1_AXP_s014
                                 + 0.758099 w_k1_DIS_s014
                                 + 0.906121 w_k1_FDX_s014
                                 + 0.878231 w_k1_IBM_s014
                                 + 1.08333 w_k1_INTC_s014
                                 + 0.722884 w_k1_KO_s014
                                 + 0.666542 w_k1_MAT_s014
                                 + 0.803133 w_k1_MSFT_s014
                                 + 0.874678 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.962878 w_k1_ADM_s014
                                 + 1.01234 w_k1_AXP_s014
                                 + 1.25484 w_k1_DIS_s014
                                 + 1.08168 w_k1_FDX_s014
                                 + 1.30145 w_k1_IBM_s014
                                 + 1.26428 w_k1_INTC_s014 + 1.2198 w_k1_KO_s014
                                 + 1.06602 w_k1_MAT_s014
                                 + 1.06747 w_k1_MSFT_s014
                                 + 1.06452 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.88035 w_k1_ADM_s015
                                 + 0.604975 w_k1_AXP_s015
                                 + 0.877046 w_k1_DIS_s015
                                 + 0.955266 w_k1_FDX_s015
                                 + 0.684347 w_k1_IBM_s015
                                 + 1.06262 w_k1_INTC_s015
                                 + 0.851465 w_k1_KO_s015
                                 + 0.478239 w_k1_MAT_s015
                                 + 1.00215 w_k1_MSFT_s015
                                 + 0.905054 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.36581 w_k1_ADM_s015
                                 + 1.16381 w_k1_AXP_s015
                                 + 1.43555 w_k1_DIS_s015
                                 + 1.29289 w_k1_FDX_s015
                                 + 1.02441 w_k1_IBM_s015
                                 + 1.41424 w_k1_INTC_s015
                                 + 1.27671 w_k1_KO_s015
                                 + 0.909558 w_k1_MAT_s015
                                 + 1.38925 w_k1_MSFT_s015
                                 + 1.09259 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.12211 w_k1_ADM_s015
                                 + 1.06686 w_k1_AXP_s015
                                 + 1.08832 w_k1_DIS_s015
                                 + 1.02021 w_k1_FDX_s015
                                 + 1.35695 w_k1_IBM_s015
                                 + 1.22981 w_k1_INTC_s015
                                 + 1.27044 w_k1_KO_s015 + 1.30767 w_k1_MAT_s015
                                 + 1.07587 w_k1_MSFT_s015 + 1.1427 w_k1_PG_s015
                                 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.0104 w_k1_ADM_s015
                                 + 1.41484 w_k1_AXP_s015
                                 + 0.869588 w_k1_DIS_s015
                                 + 0.969047 w_k1_FDX_s015
                                 + 1.09434 w_k1_IBM_s015
                                 + 1.0952 w_k1_INTC_s015 + 1.02314 w_k1_KO_s015
                                 + 1.5428 w_k1_MAT_s015
                                 + 0.891187 w_k1_MSFT_s015
                                 + 1.07085 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.939411 w_k1_ADM_s015
                                 + 1.37501 w_k1_AXP_s015 + 1.462 w_k1_DIS_s015
                                 + 1.17801 w_k1_FDX_s015 + 1.8237 w_k1_IBM_s015
                                 + 1.41256 w_k1_INTC_s015
                                 + 1.47064 w_k1_KO_s015 + 2.16522 w_k1_MAT_s015
                                 + 1.26133 w_k1_MSFT_s015
                                 + 1.17973 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.0687 w_k1_ADM_s015
                                 + 1.00159 w_k1_AXP_s015
                                 + 1.01629 w_k1_DIS_s015
                                 + 1.04506 w_k1_FDX_s015
                                 + 0.866208 w_k1_IBM_s015
                                 + 1.14357 w_k1_INTC_s015
                                 + 0.953382 w_k1_KO_s015
                                 + 0.827557 w_k1_MAT_s015
                                 + 0.988357 w_k1_MSFT_s015
                                 + 0.948895 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.979351 w_k1_ADM_s015
                                 + 0.796622 w_k1_AXP_s015
                                 + 0.646347 w_k1_DIS_s015
                                 + 0.888289 w_k1_FDX_s015
                                 + 0.582304 w_k1_IBM_s015
                                 + 0.830319 w_k1_INTC_s015
                                 + 0.657997 w_k1_KO_s015
                                 + 0.418255 w_k1_MAT_s015
                                 + 0.90913 w_k1_MSFT_s015
                                 + 0.842053 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.740065 w_k1_ADM_s015
                                 + 0.799905 w_k1_AXP_s015
                                 + 0.694304 w_k1_DIS_s015
                                 + 0.837501 w_k1_FDX_s015
                                 + 0.883511 w_k1_IBM_s015
                                 + 1.03377 w_k1_INTC_s015
                                 + 0.840874 w_k1_KO_s015
                                 + 0.803824 w_k1_MAT_s015
                                 + 0.804252 w_k1_MSFT_s015
                                 + 0.87176 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.30868 w_k1_ADM_s016
                                 + 1.45978 w_k1_AXP_s016
                                 + 1.53651 w_k1_DIS_s016
                                 + 1.28202 w_k1_FDX_s016 + 1.3551 w_k1_IBM_s016
                                 + 1.20996 w_k1_INTC_s016
                                 + 1.28372 w_k1_KO_s016 + 1.24773 w_k1_MAT_s016
                                 + 1.05913 w_k1_MSFT_s016
                                 + 1.23556 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.611332 w_k1_ADM_s016
                                 + 0.671454 w_k1_AXP_s016
                                 + 0.609632 w_k1_DIS_s016
                                 + 0.687034 w_k1_FDX_s016
                                 + 0.93584 w_k1_IBM_s016
                                 + 0.722607 w_k1_INTC_s016
                                 + 0.759018 w_k1_KO_s016
                                 + 1.08546 w_k1_MAT_s016
                                 + 0.639848 w_k1_MSFT_s016
                                 + 0.834103 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.30332 w_k1_ADM_s016
                                 + 1.25687 w_k1_AXP_s016
                                 + 1.06242 w_k1_DIS_s016
                                 + 1.22982 w_k1_FDX_s016
                                 + 0.771117 w_k1_IBM_s016
                                 + 1.16409 w_k1_INTC_s016
                                 + 0.896968 w_k1_KO_s016
                                 + 0.84032 w_k1_MAT_s016
                                 + 1.00587 w_k1_MSFT_s016
                                 + 1.09476 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.11969 w_k1_ADM_s016
                                 + 0.874559 w_k1_AXP_s016
                                 + 1.22636 w_k1_DIS_s016
                                 + 1.06775 w_k1_FDX_s016
                                 + 1.00808 w_k1_IBM_s016
                                 + 0.929108 w_k1_INTC_s016
                                 + 1.1993 w_k1_KO_s016 + 1.29332 w_k1_MAT_s016
                                 + 1.10285 w_k1_MSFT_s016
                                 + 1.17685 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.865005 w_k1_ADM_s016
                                 + 0.954365 w_k1_AXP_s016
                                 + 1.05384 w_k1_DIS_s016
                                 + 0.944874 w_k1_FDX_s016
                                 + 1.28108 w_k1_IBM_s016
                                 + 0.882334 w_k1_INTC_s016
                                 + 1.03895 w_k1_KO_s016 + 1.18694 w_k1_MAT_s016
                                 + 0.839453 w_k1_MSFT_s016
                                 + 1.0997 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.892923 w_k1_ADM_s016
                                 + 1.33266 w_k1_AXP_s016
                                 + 1.16325 w_k1_DIS_s016
                                 + 1.01939 w_k1_FDX_s016
                                 + 1.69472 w_k1_IBM_s016
                                 + 0.93724 w_k1_INTC_s016
                                 + 1.15945 w_k1_KO_s016 + 1.78644 w_k1_MAT_s016
                                 + 0.945623 w_k1_MSFT_s016
                                 + 1.08722 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.992057 w_k1_ADM_s016
                                 + 0.938357 w_k1_AXP_s016
                                 + 0.716175 w_k1_DIS_s016
                                 + 0.869282 w_k1_FDX_s016
                                 + 0.852315 w_k1_IBM_s016
                                 + 0.801951 w_k1_INTC_s016
                                 + 0.867229 w_k1_KO_s016
                                 + 1.08703 w_k1_MAT_s016
                                 + 0.79152 w_k1_MSFT_s016
                                 + 0.997615 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.922737 w_k1_ADM_s016
                                 + 0.721491 w_k1_AXP_s016
                                 + 0.857727 w_k1_DIS_s016
                                 + 0.953382 w_k1_FDX_s016
                                 + 0.6193 w_k1_IBM_s016
                                 + 0.956701 w_k1_INTC_s016
                                 + 0.653104 w_k1_KO_s016
                                 + 0.474402 w_k1_MAT_s016
                                 + 0.864718 w_k1_MSFT_s016
                                 + 0.890152 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.04801 w_k1_ADM_s017
                                 + 1.55174 w_k1_AXP_s017
                                 + 1.73229 w_k1_DIS_s017
                                 + 1.36745 w_k1_FDX_s017 + 1.1552 w_k1_IBM_s017
                                 + 1.20081 w_k1_INTC_s017
                                 + 1.32783 w_k1_KO_s017 + 1.50027 w_k1_MAT_s017
                                 + 1.3321 w_k1_MSFT_s017 + 1.25988 w_k1_PG_s017
                                 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.675242 w_k1_ADM_s017
                                 + 0.655206 w_k1_AXP_s017
                                 + 0.630709 w_k1_DIS_s017
                                 + 0.77418 w_k1_FDX_s017
                                 + 0.707028 w_k1_IBM_s017
                                 + 0.748209 w_k1_INTC_s017
                                 + 0.722817 w_k1_KO_s017
                                 + 0.566189 w_k1_MAT_s017
                                 + 0.735969 w_k1_MSFT_s017
                                 + 0.761298 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.845026 w_k1_ADM_s017
                                 + 1.18124 w_k1_AXP_s017
                                 + 1.14331 w_k1_DIS_s017
                                 + 1.15911 w_k1_FDX_s017
                                 + 1.59124 w_k1_IBM_s017
                                 + 1.00236 w_k1_INTC_s017
                                 + 1.19766 w_k1_KO_s017 + 1.84168 w_k1_MAT_s017
                                 + 1.01356 w_k1_MSFT_s017
                                 + 1.09865 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.860858 w_k1_ADM_s017
                                 + 0.798799 w_k1_AXP_s017
                                 + 0.729412 w_k1_DIS_s017
                                 + 0.829204 w_k1_FDX_s017
                                 + 1.00022 w_k1_IBM_s017
                                 + 0.696992 w_k1_INTC_s017
                                 + 0.792908 w_k1_KO_s017
                                 + 0.904105 w_k1_MAT_s017
                                 + 0.703301 w_k1_MSFT_s017
                                 + 0.864773 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.956348 w_k1_ADM_s017
                                 + 0.852673 w_k1_AXP_s017
                                 + 0.907873 w_k1_DIS_s017
                                 + 0.891673 w_k1_FDX_s017
                                 + 1.22865 w_k1_IBM_s017
                                 + 0.792384 w_k1_INTC_s017
                                 + 1.02512 w_k1_KO_s017
                                 + 0.994887 w_k1_MAT_s017
                                 + 0.886848 w_k1_MSFT_s017
                                 + 0.982832 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.809272 w_k1_ADM_s017
                                 + 0.959564 w_k1_AXP_s017
                                 + 0.913357 w_k1_DIS_s017
                                 + 0.994866 w_k1_FDX_s017
                                 + 0.931798 w_k1_IBM_s017
                                 + 1.01133 w_k1_INTC_s017
                                 + 1.01626 w_k1_KO_s017
                                 + 0.724417 w_k1_MAT_s017
                                 + 0.874142 w_k1_MSFT_s017
                                 + 1.00165 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.760185 w_k1_ADM_s017
                                 + 0.731888 w_k1_AXP_s017
                                 + 0.918017 w_k1_DIS_s017
                                 + 1.0176 w_k1_FDX_s017
                                 + 0.689144 w_k1_IBM_s017
                                 + 0.849087 w_k1_INTC_s017
                                 + 0.813556 w_k1_KO_s017
                                 + 0.69991 w_k1_MAT_s017
                                 + 0.912412 w_k1_MSFT_s017
                                 + 0.896434 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.06178 w_k1_ADM_s017
                                 + 1.03195 w_k1_AXP_s017
                                 + 1.19971 w_k1_DIS_s017
                                 + 1.10949 w_k1_FDX_s017
                                 + 1.06414 w_k1_IBM_s017
                                 + 1.02475 w_k1_INTC_s017
                                 + 1.13569 w_k1_KO_s017
                                 + 0.969504 w_k1_MAT_s017
                                 + 1.04929 w_k1_MSFT_s017
                                 + 1.10057 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.02131 w_k1_ADM_s100
                                 + 0.769144 w_k1_AXP_s100
                                 + 0.897343 w_k1_DIS_s100
                                 + 1.04122 w_k1_FDX_s100
                                 + 0.677302 w_k1_IBM_s100
                                 + 0.993557 w_k1_INTC_s100
                                 + 0.769572 w_k1_KO_s100
                                 + 0.626836 w_k1_MAT_s100
                                 + 1.00279 w_k1_MSFT_s100
                                 + 0.943578 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 0.957976 w_k1_ADM_s100
                                 + 1.55452 w_k1_AXP_s100
                                 + 1.48135 w_k1_DIS_s100
                                 + 1.07707 w_k1_FDX_s100
                                 + 1.81879 w_k1_IBM_s100
                                 + 1.12828 w_k1_INTC_s100
                                 + 1.60155 w_k1_KO_s100 + 1.92523 w_k1_MAT_s100
                                 + 1.13815 w_k1_MSFT_s100
                                 + 1.20558 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.800815 w_k1_ADM_s100
                                 + 0.812196 w_k1_AXP_s100
                                 + 0.718718 w_k1_DIS_s100
                                 + 0.878488 w_k1_FDX_s100
                                 + 1.13281 w_k1_IBM_s100
                                 + 0.639374 w_k1_INTC_s100
                                 + 0.961641 w_k1_KO_s100
                                 + 1.03211 w_k1_MAT_s100
                                 + 0.759228 w_k1_MSFT_s100
                                 + 1.04543 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.34827 w_k1_ADM_s100
                                 + 0.956134 w_k1_AXP_s100
                                 + 0.812729 w_k1_DIS_s100
                                 + 1.13901 w_k1_FDX_s100
                                 + 0.800936 w_k1_IBM_s100
                                 + 0.812161 w_k1_INTC_s100
                                 + 0.864217 w_k1_KO_s100
                                 + 0.672557 w_k1_MAT_s100
                                 + 0.896511 w_k1_MSFT_s100
                                 + 1.08069 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.10346 w_k1_ADM_s100
                                 + 1.15468 w_k1_AXP_s100
                                 + 1.09714 w_k1_DIS_s100
                                 + 1.06867 w_k1_FDX_s100
                                 + 1.36755 w_k1_IBM_s100
                                 + 0.745838 w_k1_INTC_s100
                                 + 1.20031 w_k1_KO_s100 + 1.10649 w_k1_MAT_s100
                                 + 0.910143 w_k1_MSFT_s100
                                 + 1.26775 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.860427 w_k1_ADM_s100
                                 + 0.745593 w_k1_AXP_s100
                                 + 0.72124 w_k1_DIS_s100
                                 + 0.78036 w_k1_FDX_s100
                                 + 0.829026 w_k1_IBM_s100
                                 + 0.586841 w_k1_INTC_s100
                                 + 0.764564 w_k1_KO_s100
                                 + 0.860412 w_k1_MAT_s100
                                 + 0.661078 w_k1_MSFT_s100
                                 + 0.849838 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.571507 w_k1_ADM_s100
                                 + 0.802916 w_k1_AXP_s100
                                 + 1.09259 w_k1_DIS_s100
                                 + 0.967408 w_k1_FDX_s100
                                 + 0.813317 w_k1_IBM_s100
                                 + 1.26818 w_k1_INTC_s100
                                 + 1.04933 w_k1_KO_s100
                                 + 0.862329 w_k1_MAT_s100
                                 + 1.23325 w_k1_MSFT_s100
                                 + 0.99966 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.887034 w_k1_ADM_s100
                                 + 0.979236 w_k1_AXP_s100
                                 + 1.23038 w_k1_DIS_s100
                                 + 1.19203 w_k1_FDX_s100
                                 + 0.982456 w_k1_IBM_s100
                                 + 0.954176 w_k1_INTC_s100
                                 + 1.06238 w_k1_KO_s100 + 1.18823 w_k1_MAT_s100
                                 + 1.13923 w_k1_MSFT_s100
                                 + 1.16764 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.892458 w_k1_ADM_s101
                                 + 1.20654 w_k1_AXP_s101
                                 + 0.950915 w_k1_DIS_s101
                                 + 0.969318 w_k1_FDX_s101
                                 + 1.44636 w_k1_IBM_s101
                                 + 1.21211 w_k1_INTC_s101
                                 + 1.15876 w_k1_KO_s101 + 1.46516 w_k1_MAT_s101
                                 + 0.921075 w_k1_MSFT_s101
                                 + 1.05845 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.56246 w_k1_ADM_s101
                                 + 1.05331 w_k1_AXP_s101
                                 + 0.843266 w_k1_DIS_s101
                                 + 1.01863 w_k1_FDX_s101
                                 + 0.821567 w_k1_IBM_s101
                                 + 1.20194 w_k1_INTC_s101
                                 + 0.980775 w_k1_KO_s101
                                 + 0.531893 w_k1_MAT_s101
                                 + 0.991865 w_k1_MSFT_s101
                                 + 1.01274 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.35517 w_k1_ADM_s101
                                 + 1.24073 w_k1_AXP_s101
                                 + 1.08392 w_k1_DIS_s101 + 1.1032 w_k1_FDX_s101
                                 + 1.28488 w_k1_IBM_s101
                                 + 1.13203 w_k1_INTC_s101
                                 + 1.22645 w_k1_KO_s101 + 1.10036 w_k1_MAT_s101
                                 + 1.06993 w_k1_MSFT_s101
                                 + 1.16355 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.08216 w_k1_ADM_s101
                                 + 0.791324 w_k1_AXP_s101
                                 + 0.73352 w_k1_DIS_s101
                                 + 0.901498 w_k1_FDX_s101
                                 + 0.893062 w_k1_IBM_s101
                                 + 0.973472 w_k1_INTC_s101
                                 + 0.892852 w_k1_KO_s101
                                 + 0.792788 w_k1_MAT_s101
                                 + 0.847444 w_k1_MSFT_s101
                                 + 0.982102 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.15487 w_k1_ADM_s101
                                 + 0.767911 w_k1_AXP_s101
                                 + 0.717396 w_k1_DIS_s101
                                 + 0.918482 w_k1_FDX_s101
                                 + 0.542006 w_k1_IBM_s101
                                 + 0.90706 w_k1_INTC_s101
                                 + 0.680969 w_k1_KO_s101
                                 + 0.272515 w_k1_MAT_s101
                                 + 0.880977 w_k1_MSFT_s101
                                 + 0.772888 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.34553 w_k1_ADM_s101
                                 + 1.03272 w_k1_AXP_s101
                                 + 1.21728 w_k1_DIS_s101 + 1.0463 w_k1_FDX_s101
                                 + 1.27754 w_k1_IBM_s101
                                 + 1.23548 w_k1_INTC_s101
                                 + 1.59313 w_k1_KO_s101 + 1.11413 w_k1_MAT_s101
                                 + 1.1818 w_k1_MSFT_s101 + 1.18489 w_k1_PG_s101
                                 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.25727 w_k1_ADM_s101
                                 + 0.873712 w_k1_AXP_s101
                                 + 0.772371 w_k1_DIS_s101
                                 + 0.747531 w_k1_FDX_s101
                                 + 1.51863 w_k1_IBM_s101
                                 + 1.32382 w_k1_INTC_s101 + 1.2571 w_k1_KO_s101
                                 + 0.95077 w_k1_MAT_s101
                                 + 1.0648 w_k1_MSFT_s101 + 0.95079 w_k1_PG_s101
                                 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.89964 w_k1_ADM_s101
                                 + 1.36459 w_k1_AXP_s101
                                 + 1.52289 w_k1_DIS_s101
                                 + 1.44671 w_k1_FDX_s101
                                 + 1.01315 w_k1_IBM_s101
                                 + 1.40959 w_k1_INTC_s101
                                 + 1.23254 w_k1_KO_s101
                                 + 0.593741 w_k1_MAT_s101
                                 + 1.35934 w_k1_MSFT_s101
                                 + 1.09224 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.6503 w_k1_ADM_s102
                                 + 1.42904 w_k1_AXP_s102
                                 + 1.45974 w_k1_DIS_s102
                                 + 1.33596 w_k1_FDX_s102
                                 + 0.953337 w_k1_IBM_s102
                                 + 0.988455 w_k1_INTC_s102
                                 + 1.15027 w_k1_KO_s102
                                 + 0.922642 w_k1_MAT_s102
                                 + 1.1447 w_k1_MSFT_s102 + 1.09657 w_k1_PG_s102
                                 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.996138 w_k1_ADM_s102
                                 + 0.942818 w_k1_AXP_s102
                                 + 0.636423 w_k1_DIS_s102
                                 + 0.70992 w_k1_FDX_s102
                                 + 0.966944 w_k1_IBM_s102
                                 + 0.786977 w_k1_INTC_s102
                                 + 0.862165 w_k1_KO_s102
                                 + 0.918946 w_k1_MAT_s102
                                 + 0.874057 w_k1_MSFT_s102
                                 + 0.847934 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.10916 w_k1_ADM_s102
                                 + 1.34291 w_k1_AXP_s102
                                 + 1.61467 w_k1_DIS_s102
                                 + 1.17686 w_k1_FDX_s102
                                 + 1.55504 w_k1_IBM_s102
                                 + 1.25167 w_k1_INTC_s102
                                 + 1.37123 w_k1_KO_s102 + 1.29224 w_k1_MAT_s102
                                 + 1.18183 w_k1_MSFT_s102
                                 + 1.12098 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.811353 w_k1_ADM_s102
                                 + 0.774998 w_k1_AXP_s102
                                 + 1.07162 w_k1_DIS_s102
                                 + 0.899596 w_k1_FDX_s102
                                 + 0.99017 w_k1_IBM_s102
                                 + 1.07528 w_k1_INTC_s102
                                 + 1.01588 w_k1_KO_s102
                                 + 0.870331 w_k1_MAT_s102
                                 + 1.09177 w_k1_MSFT_s102
                                 + 0.99738 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.03706 w_k1_ADM_s102
                                 + 1.32061 w_k1_AXP_s102
                                 + 0.883018 w_k1_DIS_s102
                                 + 0.938851 w_k1_FDX_s102
                                 + 0.94648 w_k1_IBM_s102
                                 + 0.988833 w_k1_INTC_s102
                                 + 0.925209 w_k1_KO_s102
                                 + 0.759596 w_k1_MAT_s102
                                 + 0.784423 w_k1_MSFT_s102
                                 + 0.966901 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.877249 w_k1_ADM_s102
                                 + 1.10829 w_k1_AXP_s102
                                 + 1.00626 w_k1_DIS_s102
                                 + 1.03878 w_k1_FDX_s102
                                 + 1.19329 w_k1_IBM_s102
                                 + 1.23289 w_k1_INTC_s102
                                 + 1.13397 w_k1_KO_s102 + 1.13626 w_k1_MAT_s102
                                 + 1.03381 w_k1_MSFT_s102
                                 + 1.11405 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.776049 w_k1_ADM_s102
                                 + 1.4092 w_k1_AXP_s102 + 1.07325 w_k1_DIS_s102
                                 + 0.920414 w_k1_FDX_s102
                                 + 1.3958 w_k1_IBM_s102 + 1.0982 w_k1_INTC_s102
                                 + 1.18621 w_k1_KO_s102 + 1.71957 w_k1_MAT_s102
                                 + 1.01012 w_k1_MSFT_s102
                                 + 1.03807 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.00411 w_k1_ADM_s102
                                 + 1.01731 w_k1_AXP_s102
                                 + 0.852908 w_k1_DIS_s102
                                 + 0.98503 w_k1_FDX_s102
                                 + 0.799943 w_k1_IBM_s102
                                 + 1.01352 w_k1_INTC_s102
                                 + 0.869267 w_k1_KO_s102
                                 + 0.821344 w_k1_MAT_s102
                                 + 1.03691 w_k1_MSFT_s102
                                 + 0.918293 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.946541 w_k1_ADM_s103
                                 + 0.829547 w_k1_AXP_s103
                                 + 0.953089 w_k1_DIS_s103
                                 + 0.958094 w_k1_FDX_s103
                                 + 0.884637 w_k1_IBM_s103
                                 + 1.17203 w_k1_INTC_s103
                                 + 0.930051 w_k1_KO_s103
                                 + 0.651556 w_k1_MAT_s103
                                 + 0.986942 w_k1_MSFT_s103
                                 + 0.788431 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.983243 w_k1_ADM_s103
                                 + 0.633364 w_k1_AXP_s103
                                 + 0.645602 w_k1_DIS_s103
                                 + 0.727109 w_k1_FDX_s103
                                 + 0.708451 w_k1_IBM_s103
                                 + 0.914795 w_k1_INTC_s103
                                 + 0.836344 w_k1_KO_s103
                                 + 0.553235 w_k1_MAT_s103
                                 + 0.873076 w_k1_MSFT_s103
                                 + 0.80789 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.791695 w_k1_ADM_s103
                                 + 0.771223 w_k1_AXP_s103
                                 + 0.900897 w_k1_DIS_s103
                                 + 0.72386 w_k1_FDX_s103
                                 + 1.14012 w_k1_IBM_s103
                                 + 1.01448 w_k1_INTC_s103
                                 + 0.986205 w_k1_KO_s103
                                 + 1.03853 w_k1_MAT_s103
                                 + 0.822245 w_k1_MSFT_s103
                                 + 0.884481 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.903073 w_k1_ADM_s103
                                 + 1.00449 w_k1_AXP_s103
                                 + 1.14622 w_k1_DIS_s103
                                 + 1.01124 w_k1_FDX_s103
                                 + 1.42091 w_k1_IBM_s103
                                 + 1.0323 w_k1_INTC_s103 + 1.02317 w_k1_KO_s103
                                 + 1.11891 w_k1_MAT_s103
                                 + 0.955861 w_k1_MSFT_s103
                                 + 0.949851 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.83974 w_k1_ADM_s103
                                 + 0.781009 w_k1_AXP_s103
                                 + 0.875429 w_k1_DIS_s103
                                 + 0.928876 w_k1_FDX_s103
                                 + 0.683997 w_k1_IBM_s103
                                 + 1.01596 w_k1_INTC_s103
                                 + 0.786867 w_k1_KO_s103
                                 + 0.629419 w_k1_MAT_s103
                                 + 1.05635 w_k1_MSFT_s103
                                 + 0.85945 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.72679 w_k1_ADM_s103
                                 + 1.59445 w_k1_AXP_s103
                                 + 1.61993 w_k1_DIS_s103
                                 + 1.32438 w_k1_FDX_s103
                                 + 1.46221 w_k1_IBM_s103
                                 + 1.29659 w_k1_INTC_s103
                                 + 1.38846 w_k1_KO_s103 + 1.51953 w_k1_MAT_s103
                                 + 1.1376 w_k1_MSFT_s103 + 1.1387 w_k1_PG_s103
                                 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.02343 w_k1_ADM_s103
                                 + 1.09889 w_k1_AXP_s103
                                 + 1.31672 w_k1_DIS_s103 + 1.2503 w_k1_FDX_s103
                                 + 0.772279 w_k1_IBM_s103
                                 + 1.79041 w_k1_INTC_s103
                                 + 1.05873 w_k1_KO_s103
                                 + 0.792634 w_k1_MAT_s103
                                 + 1.51053 w_k1_MSFT_s103
                                 + 1.15325 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.84222 w_k1_ADM_s103
                                 + 1.10671 w_k1_AXP_s103
                                 + 1.31685 w_k1_DIS_s103
                                 + 1.16514 w_k1_FDX_s103
                                 + 1.04859 w_k1_IBM_s103
                                 + 1.29396 w_k1_INTC_s103
                                 + 1.21424 w_k1_KO_s103 + 1.01038 w_k1_MAT_s103
                                 + 1.2277 w_k1_MSFT_s103
                                 + 0.974627 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.10838 w_k1_ADM_s104
                                 + 1.29791 w_k1_AXP_s104
                                 + 1.29059 w_k1_DIS_s104
                                 + 1.28665 w_k1_FDX_s104
                                 + 0.79578 w_k1_IBM_s104
                                 + 1.38361 w_k1_INTC_s104
                                 + 1.00233 w_k1_KO_s104
                                 + 0.667197 w_k1_MAT_s104
                                 + 1.2833 w_k1_MSFT_s104 + 1.14507 w_k1_PG_s104
                                 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.10046 w_k1_ADM_s104
                                 + 0.67807 w_k1_AXP_s104
                                 + 0.596632 w_k1_DIS_s104
                                 + 0.693893 w_k1_FDX_s104
                                 + 0.944785 w_k1_IBM_s104
                                 + 0.76794 w_k1_INTC_s104
                                 + 0.910499 w_k1_KO_s104
                                 + 0.659686 w_k1_MAT_s104
                                 + 0.764274 w_k1_MSFT_s104
                                 + 0.882478 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.52944 w_k1_ADM_s104
                                 + 1.47008 w_k1_AXP_s104
                                 + 1.14859 w_k1_DIS_s104
                                 + 1.17771 w_k1_FDX_s104
                                 + 1.14402 w_k1_IBM_s104
                                 + 1.06235 w_k1_INTC_s104
                                 + 0.975833 w_k1_KO_s104 + 1.1798 w_k1_MAT_s104
                                 + 1.05287 w_k1_MSFT_s104 + 1.1067 w_k1_PG_s104
                                 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.980949 w_k1_ADM_s104
                                 + 0.918611 w_k1_AXP_s104
                                 + 0.761228 w_k1_DIS_s104
                                 + 0.886312 w_k1_FDX_s104
                                 + 0.903282 w_k1_IBM_s104
                                 + 0.964075 w_k1_INTC_s104
                                 + 0.84803 w_k1_KO_s104
                                 + 0.890404 w_k1_MAT_s104
                                 + 0.829268 w_k1_MSFT_s104
                                 + 0.912861 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.13015 w_k1_ADM_s104
                                 + 1.1722 w_k1_AXP_s104
                                 + 0.969863 w_k1_DIS_s104
                                 + 0.981429 w_k1_FDX_s104
                                 + 1.2857 w_k1_IBM_s104
                                 + 0.947076 w_k1_INTC_s104
                                 + 1.11986 w_k1_KO_s104 + 1.39698 w_k1_MAT_s104
                                 + 1.1016 w_k1_MSFT_s104 + 1.04837 w_k1_PG_s104
                                 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.771087 w_k1_ADM_s104
                                 + 0.733841 w_k1_AXP_s104
                                 + 0.652793 w_k1_DIS_s104
                                 + 0.847596 w_k1_FDX_s104
                                 + 0.557825 w_k1_IBM_s104
                                 + 0.725791 w_k1_INTC_s104
                                 + 0.659284 w_k1_KO_s104
                                 + 0.544179 w_k1_MAT_s104
                                 + 0.877251 w_k1_MSFT_s104
                                 + 0.744043 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.33903 w_k1_ADM_s104
                                 + 1.44373 w_k1_AXP_s104
                                 + 1.72103 w_k1_DIS_s104
                                 + 1.30156 w_k1_FDX_s104
                                 + 1.13721 w_k1_IBM_s104
                                 + 1.53547 w_k1_INTC_s104
                                 + 1.39248 w_k1_KO_s104
                                 + 0.998602 w_k1_MAT_s104
                                 + 1.58779 w_k1_MSFT_s104
                                 + 1.17562 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.40523 w_k1_ADM_s104
                                 + 0.98603 w_k1_AXP_s104
                                 + 0.979466 w_k1_DIS_s104
                                 + 1.08954 w_k1_FDX_s104
                                 + 0.826467 w_k1_IBM_s104
                                 + 1.1049 w_k1_INTC_s104
                                 + 0.864831 w_k1_KO_s104
                                 + 0.688115 w_k1_MAT_s104
                                 + 1.1538 w_k1_MSFT_s104 + 1.02217 w_k1_PG_s104
                                 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.695852 w_k1_ADM_s105
                                 + 0.582579 w_k1_AXP_s105
                                 + 0.718708 w_k1_DIS_s105
                                 + 0.634587 w_k1_FDX_s105
                                 + 0.993153 w_k1_IBM_s105
                                 + 0.813094 w_k1_INTC_s105
                                 + 0.908683 w_k1_KO_s105
                                 + 0.722151 w_k1_MAT_s105
                                 + 0.750504 w_k1_MSFT_s105
                                 + 0.850271 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.960624 w_k1_ADM_s105
                                 + 1.13857 w_k1_AXP_s105
                                 + 1.43834 w_k1_DIS_s105
                                 + 1.33007 w_k1_FDX_s105
                                 + 0.982883 w_k1_IBM_s105
                                 + 1.41683 w_k1_INTC_s105
                                 + 1.36143 w_k1_KO_s105 + 1.04524 w_k1_MAT_s105
                                 + 1.3573 w_k1_MSFT_s105 + 1.14973 w_k1_PG_s105
                                 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.25065 w_k1_ADM_s105
                                 + 0.946567 w_k1_AXP_s105
                                 + 1.29631 w_k1_DIS_s105
                                 + 1.00901 w_k1_FDX_s105
                                 + 1.60402 w_k1_IBM_s105
                                 + 0.985083 w_k1_INTC_s105
                                 + 1.44381 w_k1_KO_s105
                                 + 0.991223 w_k1_MAT_s105
                                 + 1.03358 w_k1_MSFT_s105 + 1.1535 w_k1_PG_s105
                                 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.53751 w_k1_ADM_s105
                                 + 0.944499 w_k1_AXP_s105
                                 + 1.19311 w_k1_DIS_s105
                                 + 1.27751 w_k1_FDX_s105
                                 + 0.784123 w_k1_IBM_s105
                                 + 1.3549 w_k1_INTC_s105 + 1.08286 w_k1_KO_s105
                                 + 0.563087 w_k1_MAT_s105
                                 + 1.02854 w_k1_MSFT_s105
                                 + 1.00684 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.932614 w_k1_ADM_s105
                                 + 0.72168 w_k1_AXP_s105
                                 + 0.995863 w_k1_DIS_s105
                                 + 0.851132 w_k1_FDX_s105
                                 + 1.2423 w_k1_IBM_s105 + 1.1144 w_k1_INTC_s105
                                 + 1.17906 w_k1_KO_s105 + 1.04837 w_k1_MAT_s105
                                 + 0.969587 w_k1_MSFT_s105
                                 + 0.943262 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.08676 w_k1_ADM_s105
                                 + 0.735169 w_k1_AXP_s105
                                 + 0.814439 w_k1_DIS_s105
                                 + 0.890598 w_k1_FDX_s105
                                 + 0.878367 w_k1_IBM_s105
                                 + 0.991669 w_k1_INTC_s105
                                 + 0.977698 w_k1_KO_s105
                                 + 0.835704 w_k1_MAT_s105
                                 + 0.87091 w_k1_MSFT_s105
                                 + 0.974052 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.48028 w_k1_ADM_s105
                                 + 1.29998 w_k1_AXP_s105
                                 + 1.60921 w_k1_DIS_s105
                                 + 1.20852 w_k1_FDX_s105 + 1.2558 w_k1_IBM_s105
                                 + 1.35026 w_k1_INTC_s105
                                 + 1.57767 w_k1_KO_s105 + 1.1337 w_k1_MAT_s105
                                 + 1.23288 w_k1_MSFT_s105
                                 + 1.17009 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.10936 w_k1_ADM_s105
                                 + 0.795395 w_k1_AXP_s105
                                 + 0.785523 w_k1_DIS_s105
                                 + 0.901043 w_k1_FDX_s105
                                 + 0.743616 w_k1_IBM_s105
                                 + 1.01657 w_k1_INTC_s105
                                 + 0.831959 w_k1_KO_s105
                                 + 0.547774 w_k1_MAT_s105
                                 + 0.826071 w_k1_MSFT_s105
                                 + 0.825452 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.865472 w_k1_ADM_s106
                                 + 0.832897 w_k1_AXP_s106
                                 + 0.998145 w_k1_DIS_s106
                                 + 0.912704 w_k1_FDX_s106
                                 + 0.963934 w_k1_IBM_s106
                                 + 1.19408 w_k1_INTC_s106
                                 + 1.02342 w_k1_KO_s106
                                 + 0.688816 w_k1_MAT_s106
                                 + 1.08066 w_k1_MSFT_s106
                                 + 0.937186 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.989198 w_k1_ADM_s106
                                 + 0.84523 w_k1_AXP_s106
                                 + 0.682597 w_k1_DIS_s106
                                 + 0.835058 w_k1_FDX_s106
                                 + 0.764781 w_k1_IBM_s106
                                 + 0.968038 w_k1_INTC_s106
                                 + 0.823591 w_k1_KO_s106
                                 + 0.778985 w_k1_MAT_s106
                                 + 0.765075 w_k1_MSFT_s106
                                 + 0.814885 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.998794 w_k1_ADM_s106
                                 + 1.11742 w_k1_AXP_s106
                                 + 0.661203 w_k1_DIS_s106
                                 + 0.878044 w_k1_FDX_s106
                                 + 1.03015 w_k1_IBM_s106
                                 + 1.20735 w_k1_INTC_s106
                                 + 1.03732 w_k1_KO_s106 + 1.31087 w_k1_MAT_s106
                                 + 1.06473 w_k1_MSFT_s106
                                 + 1.06383 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.886468 w_k1_ADM_s106
                                 + 0.658191 w_k1_AXP_s106
                                 + 0.788658 w_k1_DIS_s106
                                 + 0.909798 w_k1_FDX_s106
                                 + 0.650487 w_k1_IBM_s106
                                 + 0.931634 w_k1_INTC_s106
                                 + 0.692878 w_k1_KO_s106
                                 + 0.373171 w_k1_MAT_s106
                                 + 0.933241 w_k1_MSFT_s106
                                 + 0.863529 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.21389 w_k1_ADM_s106
                                 + 1.06159 w_k1_AXP_s106
                                 + 0.93949 w_k1_DIS_s106
                                 + 0.93109 w_k1_FDX_s106
                                 + 1.41928 w_k1_IBM_s106
                                 + 1.11383 w_k1_INTC_s106
                                 + 1.12029 w_k1_KO_s106 + 1.22831 w_k1_MAT_s106
                                 + 1.08841 w_k1_MSFT_s106 + 1.1698 w_k1_PG_s106
                                 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.09562 w_k1_ADM_s106
                                 + 1.07344 w_k1_AXP_s106
                                 + 1.03249 w_k1_DIS_s106
                                 + 1.14283 w_k1_FDX_s106
                                 + 0.83722 w_k1_IBM_s106
                                 + 1.49409 w_k1_INTC_s106
                                 + 0.954502 w_k1_KO_s106
                                 + 0.497036 w_k1_MAT_s106
                                 + 1.17165 w_k1_MSFT_s106
                                 + 1.02901 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.18891 w_k1_ADM_s106
                                 + 1.14239 w_k1_AXP_s106
                                 + 1.11884 w_k1_DIS_s106
                                 + 1.04736 w_k1_FDX_s106
                                 + 1.03903 w_k1_IBM_s106
                                 + 1.44075 w_k1_INTC_s106
                                 + 1.11906 w_k1_KO_s106
                                 + 0.912583 w_k1_MAT_s106
                                 + 1.11524 w_k1_MSFT_s106
                                 + 0.979834 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.43792 w_k1_ADM_s106
                                 + 1.28309 w_k1_AXP_s106
                                 + 1.57082 w_k1_DIS_s106
                                 + 1.36796 w_k1_FDX_s106 + 1.2395 w_k1_IBM_s106
                                 + 1.3398 w_k1_INTC_s106 + 1.2607 w_k1_KO_s106
                                 + 1.39714 w_k1_MAT_s106
                                 + 1.34803 w_k1_MSFT_s106 + 1.1561 w_k1_PG_s106
                                 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.966019 w_k1_ADM_s107
                                 + 1.04042 w_k1_AXP_s107
                                 + 0.936374 w_k1_DIS_s107
                                 + 0.945125 w_k1_FDX_s107
                                 + 1.24864 w_k1_IBM_s107
                                 + 1.11534 w_k1_INTC_s107
                                 + 1.06889 w_k1_KO_s107 + 1.13633 w_k1_MAT_s107
                                 + 1.03534 w_k1_MSFT_s107
                                 + 1.00343 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.745558 w_k1_ADM_s107
                                 + 0.903718 w_k1_AXP_s107
                                 + 0.916749 w_k1_DIS_s107
                                 + 0.932805 w_k1_FDX_s107
                                 + 0.839664 w_k1_IBM_s107
                                 + 0.895743 w_k1_INTC_s107
                                 + 0.788402 w_k1_KO_s107
                                 + 0.674932 w_k1_MAT_s107
                                 + 0.948309 w_k1_MSFT_s107
                                 + 0.869072 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.06554 w_k1_ADM_s107
                                 + 1.33731 w_k1_AXP_s107 + 1.2957 w_k1_DIS_s107
                                 + 1.19076 w_k1_FDX_s107
                                 + 1.08149 w_k1_IBM_s107
                                 + 1.06776 w_k1_INTC_s107
                                 + 1.08717 w_k1_KO_s107
                                 + 0.886714 w_k1_MAT_s107
                                 + 1.05113 w_k1_MSFT_s107 + 1.1061 w_k1_PG_s107
                                 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.945758 w_k1_ADM_s107
                                 + 1.12851 w_k1_AXP_s107
                                 + 1.28092 w_k1_DIS_s107
                                 + 1.12905 w_k1_FDX_s107
                                 + 0.732722 w_k1_IBM_s107
                                 + 1.05598 w_k1_INTC_s107
                                 + 0.775925 w_k1_KO_s107
                                 + 0.518752 w_k1_MAT_s107
                                 + 1.04295 w_k1_MSFT_s107
                                 + 0.869472 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.00365 w_k1_ADM_s107
                                 + 0.820996 w_k1_AXP_s107
                                 + 0.741429 w_k1_DIS_s107
                                 + 0.882287 w_k1_FDX_s107
                                 + 0.688557 w_k1_IBM_s107
                                 + 0.886668 w_k1_INTC_s107
                                 + 0.656968 w_k1_KO_s107
                                 + 0.542752 w_k1_MAT_s107
                                 + 0.938381 w_k1_MSFT_s107
                                 + 0.752837 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.598075 w_k1_ADM_s107
                                 + 0.495094 w_k1_AXP_s107
                                 + 0.577104 w_k1_DIS_s107
                                 + 0.693751 w_k1_FDX_s107
                                 + 0.636792 w_k1_IBM_s107
                                 + 0.85139 w_k1_INTC_s107
                                 + 0.565316 w_k1_KO_s107
                                 + 0.420914 w_k1_MAT_s107
                                 + 0.652088 w_k1_MSFT_s107
                                 + 0.749096 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.56859 w_k1_ADM_s107
                                 + 1.2305 w_k1_AXP_s107 + 1.7443 w_k1_DIS_s107
                                 + 1.44351 w_k1_FDX_s107 + 1.351 w_k1_IBM_s107
                                 + 1.02711 w_k1_INTC_s107
                                 + 1.30395 w_k1_KO_s107 + 1.33692 w_k1_MAT_s107
                                 + 1.15008 w_k1_MSFT_s107
                                 + 1.20118 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.756093 w_k1_ADM_s107
                                 + 1.254 w_k1_AXP_s107 + 1.88018 w_k1_DIS_s107
                                 + 1.20302 w_k1_FDX_s107
                                 + 1.82607 w_k1_IBM_s107
                                 + 0.979803 w_k1_INTC_s107
                                 + 1.56106 w_k1_KO_s107 + 2.07871 w_k1_MAT_s107
                                 + 1.16921 w_k1_MSFT_s107
                                 + 1.17467 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.765806 w_k1_ADM_s110
                                 + 0.930487 w_k1_AXP_s110
                                 + 0.83426 w_k1_DIS_s110 + 0.8019 w_k1_FDX_s110
                                 + 1.10552 w_k1_IBM_s110
                                 + 1.03181 w_k1_INTC_s110
                                 + 0.919698 w_k1_KO_s110
                                 + 1.03431 w_k1_MAT_s110
                                 + 1.00943 w_k1_MSFT_s110
                                 + 0.859384 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.908912 w_k1_ADM_s110
                                 + 1.18131 w_k1_AXP_s110
                                 + 1.07942 w_k1_DIS_s110
                                 + 1.07533 w_k1_FDX_s110
                                 + 0.932213 w_k1_IBM_s110
                                 + 1.09407 w_k1_INTC_s110
                                 + 1.06602 w_k1_KO_s110
                                 + 0.752268 w_k1_MAT_s110
                                 + 1.17536 w_k1_MSFT_s110
                                 + 1.07855 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.827137 w_k1_ADM_s110
                                 + 0.663829 w_k1_AXP_s110
                                 + 0.666524 w_k1_DIS_s110
                                 + 0.785349 w_k1_FDX_s110
                                 + 0.569223 w_k1_IBM_s110
                                 + 0.905945 w_k1_INTC_s110
                                 + 0.637195 w_k1_KO_s110
                                 + 0.324809 w_k1_MAT_s110
                                 + 0.921423 w_k1_MSFT_s110
                                 + 0.831082 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.07063 w_k1_ADM_s110
                                 + 1.17999 w_k1_AXP_s110
                                 + 1.50367 w_k1_DIS_s110
                                 + 1.16394 w_k1_FDX_s110 + 1.444 w_k1_IBM_s110
                                 + 1.11766 w_k1_INTC_s110 + 1.5073 w_k1_KO_s110
                                 + 1.04114 w_k1_MAT_s110
                                 + 1.41143 w_k1_MSFT_s110
                                 + 1.15692 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.056 w_k1_ADM_s110
                                 + 0.763885 w_k1_AXP_s110
                                 + 0.998955 w_k1_DIS_s110
                                 + 0.956661 w_k1_FDX_s110
                                 + 0.746401 w_k1_IBM_s110
                                 + 0.91767 w_k1_INTC_s110
                                 + 0.875922 w_k1_KO_s110
                                 + 0.449475 w_k1_MAT_s110
                                 + 1.09916 w_k1_MSFT_s110
                                 + 0.929992 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.98809 w_k1_ADM_s110
                                 + 1.20345 w_k1_AXP_s110
                                 + 0.945549 w_k1_DIS_s110
                                 + 1.03372 w_k1_FDX_s110
                                 + 1.38392 w_k1_IBM_s110
                                 + 1.00235 w_k1_INTC_s110
                                 + 1.02125 w_k1_KO_s110 + 1.25821 w_k1_MAT_s110
                                 + 1.08437 w_k1_MSFT_s110
                                 + 0.968813 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.21466 w_k1_ADM_s110
                                 + 1.41796 w_k1_AXP_s110
                                 + 1.08443 w_k1_DIS_s110
                                 + 1.23019 w_k1_FDX_s110
                                 + 1.12829 w_k1_IBM_s110
                                 + 1.2231 w_k1_INTC_s110 + 1.03341 w_k1_KO_s110
                                 + 1.16227 w_k1_MAT_s110
                                 + 1.22569 w_k1_MSFT_s110
                                 + 1.03446 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.790934 w_k1_ADM_s110
                                 + 0.670814 w_k1_AXP_s110
                                 + 0.558688 w_k1_DIS_s110
                                 + 0.734784 w_k1_FDX_s110
                                 + 0.813226 w_k1_IBM_s110
                                 + 0.825912 w_k1_INTC_s110
                                 + 0.74597 w_k1_KO_s110
                                 + 0.645471 w_k1_MAT_s110
                                 + 0.826973 w_k1_MSFT_s110
                                 + 0.829085 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.17867 w_k1_ADM_s111
                                 + 1.11129 w_k1_AXP_s111
                                 + 1.41652 w_k1_DIS_s111
                                 + 1.14385 w_k1_FDX_s111 + 1.0959 w_k1_IBM_s111
                                 + 1.09376 w_k1_INTC_s111
                                 + 1.13939 w_k1_KO_s111 + 1.00344 w_k1_MAT_s111
                                 + 1.07216 w_k1_MSFT_s111
                                 + 0.936624 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.742714 w_k1_ADM_s111
                                 + 0.787897 w_k1_AXP_s111
                                 + 0.713924 w_k1_DIS_s111
                                 + 0.722247 w_k1_FDX_s111
                                 + 0.91064 w_k1_IBM_s111
                                 + 0.812223 w_k1_INTC_s111
                                 + 0.759238 w_k1_KO_s111
                                 + 0.832999 w_k1_MAT_s111
                                 + 0.79083 w_k1_MSFT_s111
                                 + 0.805231 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.797233 w_k1_ADM_s111
                                 + 1.21311 w_k1_AXP_s111
                                 + 1.09823 w_k1_DIS_s111
                                 + 0.921908 w_k1_FDX_s111
                                 + 1.09906 w_k1_IBM_s111
                                 + 0.989101 w_k1_INTC_s111
                                 + 1.08115 w_k1_KO_s111 + 1.24601 w_k1_MAT_s111
                                 + 1.09159 w_k1_MSFT_s111
                                 + 0.940575 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.913837 w_k1_ADM_s111
                                 + 1.58388 w_k1_AXP_s111
                                 + 1.25313 w_k1_DIS_s111
                                 + 1.23381 w_k1_FDX_s111
                                 + 0.971261 w_k1_IBM_s111
                                 + 1.20635 w_k1_INTC_s111
                                 + 1.07888 w_k1_KO_s111 + 1.4973 w_k1_MAT_s111
                                 + 1.30186 w_k1_MSFT_s111
                                 + 0.958034 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.99207 w_k1_ADM_s111
                                 + 1.19222 w_k1_AXP_s111
                                 + 1.35433 w_k1_DIS_s111
                                 + 1.04979 w_k1_FDX_s111
                                 + 1.70163 w_k1_IBM_s111
                                 + 1.02883 w_k1_INTC_s111
                                 + 1.03297 w_k1_KO_s111 + 1.60441 w_k1_MAT_s111
                                 + 1.09869 w_k1_MSFT_s111
                                 + 0.877825 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.0411 w_k1_ADM_s111
                                 + 0.920003 w_k1_AXP_s111
                                 + 1.10721 w_k1_DIS_s111
                                 + 1.03782 w_k1_FDX_s111
                                 + 0.941288 w_k1_IBM_s111
                                 + 1.20501 w_k1_INTC_s111
                                 + 1.09962 w_k1_KO_s111
                                 + 0.921301 w_k1_MAT_s111
                                 + 1.11968 w_k1_MSFT_s111
                                 + 0.915568 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.97033 w_k1_ADM_s111
                                 + 1.04538 w_k1_AXP_s111
                                 + 0.974401 w_k1_DIS_s111
                                 + 0.938928 w_k1_FDX_s111
                                 + 0.740692 w_k1_IBM_s111
                                 + 0.928034 w_k1_INTC_s111
                                 + 0.946491 w_k1_KO_s111
                                 + 0.924641 w_k1_MAT_s111
                                 + 0.955202 w_k1_MSFT_s111
                                 + 0.947313 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.36256 w_k1_ADM_s111
                                 + 1.35821 w_k1_AXP_s111 + 1.433 w_k1_DIS_s111
                                 + 1.20524 w_k1_FDX_s111
                                 + 1.29563 w_k1_IBM_s111
                                 + 1.09829 w_k1_INTC_s111
                                 + 1.31774 w_k1_KO_s111 + 1.61045 w_k1_MAT_s111
                                 + 1.23932 w_k1_MSFT_s111
                                 + 1.08457 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.855352 w_k1_ADM_s112
                                 + 0.841192 w_k1_AXP_s112
                                 + 0.7972 w_k1_DIS_s112
                                 + 0.792478 w_k1_FDX_s112
                                 + 1.16694 w_k1_IBM_s112
                                 + 0.985907 w_k1_INTC_s112
                                 + 1.07309 w_k1_KO_s112 + 1.07347 w_k1_MAT_s112
                                 + 0.799912 w_k1_MSFT_s112
                                 + 0.813767 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.975807 w_k1_ADM_s112
                                 + 1.0102 w_k1_AXP_s112 + 0.8265 w_k1_DIS_s112
                                 + 0.967306 w_k1_FDX_s112
                                 + 1.54688 w_k1_IBM_s112
                                 + 0.830829 w_k1_INTC_s112
                                 + 1.19494 w_k1_KO_s112 + 1.93253 w_k1_MAT_s112
                                 + 0.961123 w_k1_MSFT_s112
                                 + 0.965319 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.16069 w_k1_ADM_s112
                                 + 0.913085 w_k1_AXP_s112
                                 + 0.976315 w_k1_DIS_s112
                                 + 1.10952 w_k1_FDX_s112
                                 + 0.640219 w_k1_IBM_s112
                                 + 0.998063 w_k1_INTC_s112
                                 + 0.721543 w_k1_KO_s112
                                 + 0.801629 w_k1_MAT_s112
                                 + 1.01773 w_k1_MSFT_s112
                                 + 0.883265 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.2234 w_k1_ADM_s112
                                 + 0.968289 w_k1_AXP_s112
                                 + 1.16474 w_k1_DIS_s112
                                 + 1.05915 w_k1_FDX_s112
                                 + 1.09804 w_k1_IBM_s112
                                 + 0.79462 w_k1_INTC_s112
                                 + 1.13049 w_k1_KO_s112 + 1.06292 w_k1_MAT_s112
                                 + 1.1908 w_k1_MSFT_s112
                                 + 0.960797 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.06144 w_k1_ADM_s112
                                 + 0.915083 w_k1_AXP_s112
                                 + 0.647698 w_k1_DIS_s112
                                 + 0.783742 w_k1_FDX_s112
                                 + 0.815453 w_k1_IBM_s112
                                 + 0.848386 w_k1_INTC_s112
                                 + 0.793164 w_k1_KO_s112
                                 + 0.822634 w_k1_MAT_s112
                                 + 0.840875 w_k1_MSFT_s112
                                 + 0.827615 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.29905 w_k1_ADM_s112
                                 + 1.70585 w_k1_AXP_s112
                                 + 1.31956 w_k1_DIS_s112
                                 + 1.22413 w_k1_FDX_s112
                                 + 1.07913 w_k1_IBM_s112
                                 + 1.18266 w_k1_INTC_s112
                                 + 0.990486 w_k1_KO_s112
                                 + 1.49232 w_k1_MAT_s112
                                 + 1.21093 w_k1_MSFT_s112
                                 + 1.10717 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.39106 w_k1_ADM_s112
                                 + 1.42199 w_k1_AXP_s112
                                 + 1.68466 w_k1_DIS_s112
                                 + 1.22894 w_k1_FDX_s112
                                 + 1.41234 w_k1_IBM_s112
                                 + 0.961857 w_k1_INTC_s112
                                 + 1.26064 w_k1_KO_s112 + 1.64749 w_k1_MAT_s112
                                 + 1.11998 w_k1_MSFT_s112
                                 + 1.08991 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.00185 w_k1_ADM_s112
                                 + 1.10995 w_k1_AXP_s112
                                 + 1.13278 w_k1_DIS_s112
                                 + 1.19324 w_k1_FDX_s112
                                 + 0.957281 w_k1_IBM_s112
                                 + 1.19076 w_k1_INTC_s112
                                 + 0.966397 w_k1_KO_s112
                                 + 0.854976 w_k1_MAT_s112
                                 + 1.16847 w_k1_MSFT_s112
                                 + 0.946366 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.17651 w_k1_ADM_s113
                                 + 1.19727 w_k1_AXP_s113 + 1.1431 w_k1_DIS_s113
                                 + 1.17198 w_k1_FDX_s113
                                 + 0.853595 w_k1_IBM_s113
                                 + 1.17563 w_k1_INTC_s113
                                 + 1.11625 w_k1_KO_s113
                                 + 0.669737 w_k1_MAT_s113
                                 + 1.03798 w_k1_MSFT_s113
                                 + 0.999247 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.674175 w_k1_ADM_s113
                                 + 0.901976 w_k1_AXP_s113
                                 + 0.675867 w_k1_DIS_s113
                                 + 0.795013 w_k1_FDX_s113
                                 + 0.980649 w_k1_IBM_s113
                                 + 1.00343 w_k1_INTC_s113
                                 + 0.857913 w_k1_KO_s113
                                 + 1.02839 w_k1_MAT_s113
                                 + 0.897217 w_k1_MSFT_s113
                                 + 0.796024 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.975045 w_k1_ADM_s113
                                 + 1.42756 w_k1_AXP_s113
                                 + 1.27778 w_k1_DIS_s113 + 1.2292 w_k1_FDX_s113
                                 + 1.15226 w_k1_IBM_s113
                                 + 1.27213 w_k1_INTC_s113
                                 + 1.20174 w_k1_KO_s113 + 0.9919 w_k1_MAT_s113
                                 + 1.08326 w_k1_MSFT_s113
                                 + 1.05183 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.881606 w_k1_ADM_s113
                                 + 0.698587 w_k1_AXP_s113
                                 + 0.753358 w_k1_DIS_s113
                                 + 0.906619 w_k1_FDX_s113
                                 + 0.716432 w_k1_IBM_s113
                                 + 0.930481 w_k1_INTC_s113
                                 + 0.85953 w_k1_KO_s113
                                 + 0.632436 w_k1_MAT_s113
                                 + 0.862726 w_k1_MSFT_s113
                                 + 0.891092 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.660288 w_k1_ADM_s113
                                 + 0.813876 w_k1_AXP_s113
                                 + 0.988684 w_k1_DIS_s113
                                 + 0.849502 w_k1_FDX_s113
                                 + 1.10856 w_k1_IBM_s113
                                 + 0.860216 w_k1_INTC_s113 + 1.009 w_k1_KO_s113
                                 + 0.824288 w_k1_MAT_s113
                                 + 0.861852 w_k1_MSFT_s113
                                 + 0.978561 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.19104 w_k1_ADM_s113
                                 + 1.10999 w_k1_AXP_s113
                                 + 1.51805 w_k1_DIS_s113
                                 + 1.34096 w_k1_FDX_s113
                                 + 1.53955 w_k1_IBM_s113
                                 + 1.34856 w_k1_INTC_s113
                                 + 1.79659 w_k1_KO_s113 + 1.34893 w_k1_MAT_s113
                                 + 1.30154 w_k1_MSFT_s113
                                 + 1.21325 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.994916 w_k1_ADM_s113
                                 + 1.07192 w_k1_AXP_s113
                                 + 0.924596 w_k1_DIS_s113
                                 + 0.938133 w_k1_FDX_s113
                                 + 1.34716 w_k1_IBM_s113
                                 + 1.0518 w_k1_INTC_s113 + 1.23608 w_k1_KO_s113
                                 + 1.20215 w_k1_MAT_s113
                                 + 0.980469 w_k1_MSFT_s113
                                 + 1.16834 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.709041 w_k1_ADM_s113
                                 + 1.20322 w_k1_AXP_s113
                                 + 1.13931 w_k1_DIS_s113
                                 + 1.04455 w_k1_FDX_s113
                                 + 0.982664 w_k1_IBM_s113
                                 + 1.15547 w_k1_INTC_s113
                                 + 1.09432 w_k1_KO_s113
                                 + 0.973386 w_k1_MAT_s113
                                 + 1.00878 w_k1_MSFT_s113
                                 + 1.06015 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.874963 w_k1_ADM_s114
                                 + 0.713071 w_k1_AXP_s114
                                 + 0.597078 w_k1_DIS_s114
                                 + 0.759498 w_k1_FDX_s114
                                 + 0.859688 w_k1_IBM_s114
                                 + 0.785385 w_k1_INTC_s114
                                 + 0.765716 w_k1_KO_s114
                                 + 0.814199 w_k1_MAT_s114
                                 + 0.792575 w_k1_MSFT_s114
                                 + 0.846623 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 1.02098 w_k1_ADM_s114
                                 + 1.09327 w_k1_AXP_s114
                                 + 0.680187 w_k1_DIS_s114
                                 + 0.938688 w_k1_FDX_s114
                                 + 0.792173 w_k1_IBM_s114
                                 + 0.864713 w_k1_INTC_s114
                                 + 0.757915 w_k1_KO_s114
                                 + 1.00495 w_k1_MAT_s114
                                 + 0.792797 w_k1_MSFT_s114
                                 + 1.03523 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.880533 w_k1_ADM_s114
                                 + 0.922131 w_k1_AXP_s114
                                 + 0.754048 w_k1_DIS_s114
                                 + 0.862222 w_k1_FDX_s114
                                 + 1.10132 w_k1_IBM_s114
                                 + 1.04003 w_k1_INTC_s114
                                 + 0.933392 w_k1_KO_s114
                                 + 0.829955 w_k1_MAT_s114
                                 + 0.813438 w_k1_MSFT_s114
                                 + 1.00058 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.71686 w_k1_ADM_s114
                                 + 1.38774 w_k1_AXP_s114
                                 + 1.46103 w_k1_DIS_s114
                                 + 1.29763 w_k1_FDX_s114
                                 + 1.16902 w_k1_IBM_s114
                                 + 1.17673 w_k1_INTC_s114
                                 + 1.28486 w_k1_KO_s114 + 1.02107 w_k1_MAT_s114
                                 + 1.07968 w_k1_MSFT_s114
                                 + 1.39105 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.899979 w_k1_ADM_s114
                                 + 0.784717 w_k1_AXP_s114
                                 + 0.788405 w_k1_DIS_s114
                                 + 0.856692 w_k1_FDX_s114
                                 + 0.665778 w_k1_IBM_s114
                                 + 0.963264 w_k1_INTC_s114
                                 + 0.780598 w_k1_KO_s114
                                 + 0.659003 w_k1_MAT_s114
                                 + 0.912151 w_k1_MSFT_s114
                                 + 0.989551 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.838399 w_k1_ADM_s114
                                 + 1.23359 w_k1_AXP_s114
                                 + 1.34204 w_k1_DIS_s114
                                 + 1.11554 w_k1_FDX_s114
                                 + 1.50645 w_k1_IBM_s114
                                 + 1.30198 w_k1_INTC_s114
                                 + 1.57766 w_k1_KO_s114 + 1.5553 w_k1_MAT_s114
                                 + 1.12927 w_k1_MSFT_s114
                                 + 1.24367 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.01804 w_k1_ADM_s114
                                 + 1.0458 w_k1_AXP_s114 + 1.0824 w_k1_DIS_s114
                                 + 1.04865 w_k1_FDX_s114
                                 + 0.986551 w_k1_IBM_s114
                                 + 1.27004 w_k1_INTC_s114
                                 + 1.15294 w_k1_KO_s114
                                 + 0.771703 w_k1_MAT_s114
                                 + 0.987347 w_k1_MSFT_s114
                                 + 1.15337 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.38374 w_k1_ADM_s114
                                 + 1.2053 w_k1_AXP_s114 + 1.31856 w_k1_DIS_s114
                                 + 1.38315 w_k1_FDX_s114
                                 + 0.710868 w_k1_IBM_s114
                                 + 1.33599 w_k1_INTC_s114
                                 + 1.01805 w_k1_KO_s114
                                 + 0.751264 w_k1_MAT_s114
                                 + 1.21281 w_k1_MSFT_s114
                                 + 1.17035 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.799541 w_k1_ADM_s115
                                 + 0.681697 w_k1_AXP_s115
                                 + 0.705855 w_k1_DIS_s115
                                 + 0.775708 w_k1_FDX_s115
                                 + 0.964634 w_k1_IBM_s115
                                 + 0.795418 w_k1_INTC_s115
                                 + 0.871359 w_k1_KO_s115
                                 + 0.714376 w_k1_MAT_s115
                                 + 0.690659 w_k1_MSFT_s115
                                 + 0.926852 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.791185 w_k1_ADM_s115
                                 + 0.75566 w_k1_AXP_s115
                                 + 0.849649 w_k1_DIS_s115
                                 + 0.891045 w_k1_FDX_s115
                                 + 0.634046 w_k1_IBM_s115
                                 + 0.849725 w_k1_INTC_s115
                                 + 0.688467 w_k1_KO_s115
                                 + 0.696958 w_k1_MAT_s115
                                 + 0.700068 w_k1_MSFT_s115
                                 + 0.872881 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.30979 w_k1_ADM_s115
                                 + 1.30214 w_k1_AXP_s115
                                 + 1.13713 w_k1_DIS_s115
                                 + 1.25182 w_k1_FDX_s115
                                 + 0.783869 w_k1_IBM_s115
                                 + 1.32943 w_k1_INTC_s115
                                 + 1.16032 w_k1_KO_s115
                                 + 0.994544 w_k1_MAT_s115
                                 + 1.11908 w_k1_MSFT_s115
                                 + 1.15871 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.02122 w_k1_ADM_s115
                                 + 0.87529 w_k1_AXP_s115
                                 + 0.653049 w_k1_DIS_s115
                                 + 0.822144 w_k1_FDX_s115
                                 + 0.786514 w_k1_IBM_s115
                                 + 0.90729 w_k1_INTC_s115
                                 + 0.741319 w_k1_KO_s115
                                 + 0.651107 w_k1_MAT_s115
                                 + 0.762678 w_k1_MSFT_s115
                                 + 0.844716 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.13836 w_k1_ADM_s115
                                 + 1.27476 w_k1_AXP_s115
                                 + 1.57585 w_k1_DIS_s115
                                 + 1.25479 w_k1_FDX_s115
                                 + 1.29171 w_k1_IBM_s115
                                 + 1.30947 w_k1_INTC_s115
                                 + 1.68073 w_k1_KO_s115 + 1.45568 w_k1_MAT_s115
                                 + 1.1891 w_k1_MSFT_s115 + 1.3146 w_k1_PG_s115
                                 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.847373 w_k1_ADM_s115
                                 + 0.97439 w_k1_AXP_s115
                                 + 0.609895 w_k1_DIS_s115
                                 + 0.763423 w_k1_FDX_s115
                                 + 1.15057 w_k1_IBM_s115
                                 + 0.87631 w_k1_INTC_s115
                                 + 0.907042 w_k1_KO_s115 + 1.0243 w_k1_MAT_s115
                                 + 0.692448 w_k1_MSFT_s115
                                 + 0.949563 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.06929 w_k1_ADM_s115
                                 + 1.23859 w_k1_AXP_s115
                                 + 1.06463 w_k1_DIS_s115
                                 + 0.931556 w_k1_FDX_s115
                                 + 1.48421 w_k1_IBM_s115
                                 + 1.18922 w_k1_INTC_s115
                                 + 1.45916 w_k1_KO_s115 + 1.79412 w_k1_MAT_s115
                                 + 1.05446 w_k1_MSFT_s115
                                 + 1.26683 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.24329 w_k1_ADM_s115
                                 + 1.22752 w_k1_AXP_s115
                                 + 1.20847 w_k1_DIS_s115
                                 + 1.06592 w_k1_FDX_s115
                                 + 1.02351 w_k1_IBM_s115
                                 + 1.13187 w_k1_INTC_s115
                                 + 1.37983 w_k1_KO_s115 + 1.17457 w_k1_MAT_s115
                                 + 0.96648 w_k1_MSFT_s115
                                 + 1.13936 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.06915 w_k1_ADM_s116
                                 + 1.5577 w_k1_AXP_s116 + 1.48309 w_k1_DIS_s116
                                 + 1.4667 w_k1_FDX_s116 + 1.04183 w_k1_IBM_s116
                                 + 1.19784 w_k1_INTC_s116 + 1.1929 w_k1_KO_s116
                                 + 1.32286 w_k1_MAT_s116
                                 + 1.32909 w_k1_MSFT_s116
                                 + 0.984079 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.918548 w_k1_ADM_s116
                                 + 1.18811 w_k1_AXP_s116
                                 + 1.03909 w_k1_DIS_s116
                                 + 1.11138 w_k1_FDX_s116
                                 + 0.789989 w_k1_IBM_s116
                                 + 0.927239 w_k1_INTC_s116
                                 + 0.947398 w_k1_KO_s116
                                 + 0.970074 w_k1_MAT_s116
                                 + 1.05526 w_k1_MSFT_s116
                                 + 0.945251 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.850389 w_k1_ADM_s116
                                 + 1.02114 w_k1_AXP_s116
                                 + 1.06326 w_k1_DIS_s116
                                 + 0.945732 w_k1_FDX_s116
                                 + 1.23007 w_k1_IBM_s116
                                 + 0.893198 w_k1_INTC_s116
                                 + 1.02753 w_k1_KO_s116 + 1.43016 w_k1_MAT_s116
                                 + 0.922338 w_k1_MSFT_s116
                                 + 0.89457 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.531337 w_k1_ADM_s116
                                 + 0.623055 w_k1_AXP_s116
                                 + 0.703877 w_k1_DIS_s116
                                 + 0.689951 w_k1_FDX_s116
                                 + 0.849252 w_k1_IBM_s116
                                 + 0.715674 w_k1_INTC_s116
                                 + 0.72826 w_k1_KO_s116
                                 + 0.707435 w_k1_MAT_s116
                                 + 0.904832 w_k1_MSFT_s116
                                 + 0.67508 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.05698 w_k1_ADM_s116
                                 + 1.38468 w_k1_AXP_s116 + 1.3999 w_k1_DIS_s116
                                 + 1.17755 w_k1_FDX_s116
                                 + 1.44858 w_k1_IBM_s116
                                 + 1.12534 w_k1_INTC_s116
                                 + 1.32128 w_k1_KO_s116 + 1.53489 w_k1_MAT_s116
                                 + 1.20448 w_k1_MSFT_s116
                                 + 1.05913 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.856389 w_k1_ADM_s116
                                 + 0.702885 w_k1_AXP_s116
                                 + 0.741758 w_k1_DIS_s116
                                 + 0.950424 w_k1_FDX_s116
                                 + 0.778706 w_k1_IBM_s116
                                 + 0.886657 w_k1_INTC_s116
                                 + 0.820554 w_k1_KO_s116
                                 + 0.598091 w_k1_MAT_s116
                                 + 0.918268 w_k1_MSFT_s116
                                 + 0.867784 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.7427 w_k1_ADM_s116
                                 + 0.891496 w_k1_AXP_s116
                                 + 1.0269 w_k1_DIS_s116 + 1.08178 w_k1_FDX_s116
                                 + 0.64271 w_k1_IBM_s116
                                 + 0.927589 w_k1_INTC_s116
                                 + 0.750018 w_k1_KO_s116
                                 + 0.613973 w_k1_MAT_s116
                                 + 0.988261 w_k1_MSFT_s116
                                 + 0.815287 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.930792 w_k1_ADM_s116
                                 + 1.22853 w_k1_AXP_s116
                                 + 1.35478 w_k1_DIS_s116
                                 + 1.22479 w_k1_FDX_s116
                                 + 1.09519 w_k1_IBM_s116
                                 + 0.925607 w_k1_INTC_s116
                                 + 1.10873 w_k1_KO_s116 + 1.34577 w_k1_MAT_s116
                                 + 1.11683 w_k1_MSFT_s116 + 1.0937 w_k1_PG_s116
                                 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.859223 w_k1_ADM_s117
                                 + 0.852565 w_k1_AXP_s117
                                 + 0.993425 w_k1_DIS_s117
                                 + 0.876359 w_k1_FDX_s117
                                 + 1.11338 w_k1_IBM_s117
                                 + 1.04377 w_k1_INTC_s117
                                 + 1.10311 w_k1_KO_s117 + 1.11068 w_k1_MAT_s117
                                 + 1.05324 w_k1_MSFT_s117
                                 + 0.934096 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.2614 w_k1_ADM_s117
                                 + 0.974297 w_k1_AXP_s117
                                 + 0.932262 w_k1_DIS_s117
                                 + 0.927315 w_k1_FDX_s117
                                 + 0.793905 w_k1_IBM_s117
                                 + 0.905733 w_k1_INTC_s117
                                 + 1.01631 w_k1_KO_s117
                                 + 0.816929 w_k1_MAT_s117
                                 + 1.07272 w_k1_MSFT_s117
                                 + 1.01611 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.695142 w_k1_ADM_s117
                                 + 0.686505 w_k1_AXP_s117
                                 + 0.629955 w_k1_DIS_s117
                                 + 0.65127 w_k1_FDX_s117
                                 + 0.820534 w_k1_IBM_s117
                                 + 0.852803 w_k1_INTC_s117
                                 + 0.836864 w_k1_KO_s117
                                 + 0.579275 w_k1_MAT_s117
                                 + 0.724764 w_k1_MSFT_s117
                                 + 0.757132 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.18156 w_k1_ADM_s117
                                 + 1.42085 w_k1_AXP_s117
                                 + 1.65088 w_k1_DIS_s117
                                 + 1.32525 w_k1_FDX_s117
                                 + 1.23516 w_k1_IBM_s117
                                 + 1.21296 w_k1_INTC_s117
                                 + 1.24377 w_k1_KO_s117 + 1.43043 w_k1_MAT_s117
                                 + 1.36451 w_k1_MSFT_s117
                                 + 1.06399 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.842397 w_k1_ADM_s117
                                 + 0.663092 w_k1_AXP_s117
                                 + 0.986584 w_k1_DIS_s117
                                 + 0.822354 w_k1_FDX_s117
                                 + 0.945614 w_k1_IBM_s117
                                 + 0.957823 w_k1_INTC_s117
                                 + 0.975148 w_k1_KO_s117
                                 + 0.88215 w_k1_MAT_s117
                                 + 1.00665 w_k1_MSFT_s117
                                 + 0.887227 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.881966 w_k1_ADM_s117
                                 + 0.614602 w_k1_AXP_s117
                                 + 0.822253 w_k1_DIS_s117
                                 + 0.91043 w_k1_FDX_s117
                                 + 0.554751 w_k1_IBM_s117
                                 + 0.908042 w_k1_INTC_s117
                                 + 0.741773 w_k1_KO_s117
                                 + 0.51827 w_k1_MAT_s117
                                 + 0.896386 w_k1_MSFT_s117
                                 + 0.90186 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.930929 w_k1_ADM_s117
                                 + 1.11338 w_k1_AXP_s117 + 1.0175 w_k1_DIS_s117
                                 + 1.05314 w_k1_FDX_s117
                                 + 0.809326 w_k1_IBM_s117
                                 + 1.07988 w_k1_INTC_s117
                                 + 0.96996 w_k1_KO_s117
                                 + 0.673056 w_k1_MAT_s117
                                 + 0.91569 w_k1_MSFT_s117
                                 + 0.850392 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.843308 w_k1_ADM_s117
                                 + 0.999491 w_k1_AXP_s117 + 1.115 w_k1_DIS_s117
                                 + 0.954519 w_k1_FDX_s117
                                 + 1.42916 w_k1_IBM_s117
                                 + 1.1733 w_k1_INTC_s117 + 1.14484 w_k1_KO_s117
                                 + 1.73174 w_k1_MAT_s117
                                 + 1.07667 w_k1_MSFT_s117
                                 + 1.02899 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0437880425223074 w_k1_ADM_s000
                                 + 0.0417016030249102 w_k1_ADM_s001
                                 + 0.0241394847441823 w_k1_ADM_s002
                                 + 0.0753278128064741 w_k1_ADM_s003
                                 + 0.0565924585245183 w_k1_ADM_s004
                                 + 0.0211644925558497 w_k1_ADM_s005
                                 + 0.0507974139425992 w_k1_ADM_s006
                                 + 0.0250112590954986 w_k1_ADM_s007
                                 + 0.0924220154809976 w_k1_ADM_s010
                                 + 0.0191575210790461 w_k1_ADM_s011
                                 + 0.0679451253044169 w_k1_ADM_s012
                                 + 0.0674091402624177 w_k1_ADM_s013
                                 + 0.061050591474035 w_k1_ADM_s014
                                 + 0.0529411977403864 w_k1_ADM_s015
                                 + 0.0420642610150925 w_k1_ADM_s016
                                 + 0.0359954579091322 w_k1_ADM_s017
                                 + 0.00434066561629914 w_k1_ADM_s100
                                 + 0.00697651190030326 w_k1_ADM_s101
                                 + 0.0140787595824135 w_k1_ADM_s102
                                 + 0.00482950915262787 w_k1_ADM_s103
                                 + 0.00632855317828141 w_k1_ADM_s104
                                 + 0.00735471645921553 w_k1_ADM_s105
                                 + 0.00981159318922322 w_k1_ADM_s106
                                 + 0.0030439652895604 w_k1_ADM_s107
                                 + 0.0276383500078316 w_k1_ADM_s110
                                 + 0.0343750085982844 w_k1_ADM_s111
                                 + 0.0225423987643871 w_k1_ADM_s112
                                 + 0.013050903161036 w_k1_ADM_s113
                                 + 0.0221399092049291 w_k1_ADM_s114
                                 + 0.0155093946298415 w_k1_ADM_s115
                                 + 0.00797662145450021 w_k1_ADM_s116
                                 + 0.0189847433219585 w_k1_ADM_s117
                                 + 0.047147381254201 w_k1_AXP_s000
                                 + 0.0387488114104194 w_k1_AXP_s001
                                 + 0.0261815976891929 w_k1_AXP_s002
                                 + 0.0827243487133386 w_k1_AXP_s003
                                 + 0.0591787610194746 w_k1_AXP_s004
                                 + 0.0226629080901677 w_k1_AXP_s005
                                 + 0.0508830063396453 w_k1_AXP_s006
                                 + 0.0298832032539699 w_k1_AXP_s007
                                 + 0.0892907270766846 w_k1_AXP_s010
                                 + 0.021691184616556 w_k1_AXP_s011
                                 + 0.0618403729527607 w_k1_AXP_s012
                                 + 0.0676941893928572 w_k1_AXP_s013
                                 + 0.0534887227906126 w_k1_AXP_s014
                                 + 0.0530152029925051 w_k1_AXP_s015
                                 + 0.0412099539874524 w_k1_AXP_s016
                                 + 0.0400427118010096 w_k1_AXP_s017
                                 + 0.00439757784986172 w_k1_AXP_s100
                                 + 0.00555665808071493 w_k1_AXP_s101
                                 + 0.01479227795825 w_k1_AXP_s102
                                 + 0.00481043423527282 w_k1_AXP_s103
                                 + 0.00594774225809195 w_k1_AXP_s104
                                 + 0.00569454609044723 w_k1_AXP_s105
                                 + 0.0091195147131659 w_k1_AXP_s106
                                 + 0.00347045857737671 w_k1_AXP_s107
                                 + 0.0294731788695134 w_k1_AXP_s110
                                 + 0.037981328154869 w_k1_AXP_s111
                                 + 0.0218411855202908 w_k1_AXP_s112
                                 + 0.0145791377948738 w_k1_AXP_s113
                                 + 0.0220387536888125 w_k1_AXP_s114
                                 + 0.0157443912710073 w_k1_AXP_s115
                                 + 0.00930477229535982 w_k1_AXP_s116
                                 + 0.0194948723068648 w_k1_AXP_s117
                                 + 0.0477093431366951 w_k1_DIS_s000
                                 + 0.0401078515151354 w_k1_DIS_s001
                                 + 0.0257966636716655 w_k1_DIS_s002
                                 + 0.0779499470369976 w_k1_DIS_s003
                                 + 0.0647143626344361 w_k1_DIS_s004
                                 + 0.022736016349217 w_k1_DIS_s005
                                 + 0.0560606657330575 w_k1_DIS_s006
                                 + 0.0341966566693915 w_k1_DIS_s007
                                 + 0.0899466062879311 w_k1_DIS_s010
                                 + 0.0201008565005454 w_k1_DIS_s011
                                 + 0.0647488860706299 w_k1_DIS_s012
                                 + 0.0774715363277771 w_k1_DIS_s013
                                 + 0.0606039108009881 w_k1_DIS_s014
                                 + 0.0533659462973541 w_k1_DIS_s015
                                 + 0.0422282637064244 w_k1_DIS_s016
                                 + 0.0427614038842788 w_k1_DIS_s017
                                 + 0.00460182744999492 w_k1_DIS_s100
                                 + 0.00540285279186408 w_k1_DIS_s101
                                 + 0.013880724739168 w_k1_DIS_s102
                                 + 0.00547642440644661 w_k1_DIS_s103
                                 + 0.00548022770397167 w_k1_DIS_s104
                                 + 0.00692023755997288 w_k1_DIS_s105
                                 + 0.00912000418564171 w_k1_DIS_s106
                                 + 0.00370954516792302 w_k1_DIS_s107
                                 + 0.0291820042761833 w_k1_DIS_s110
                                 + 0.0385151497055388 w_k1_DIS_s111
                                 + 0.0205610501800657 w_k1_DIS_s112
                                 + 0.0143957885623251 w_k1_DIS_s113
                                 + 0.021917166606223 w_k1_DIS_s114
                                 + 0.0155631367763188 w_k1_DIS_s115
                                 + 0.00946188139688527 w_k1_DIS_s116
                                 + 0.0215787897713561 w_k1_DIS_s117
                                 + 0.0483237068403152 w_k1_FDX_s000
                                 + 0.0401542921590329 w_k1_FDX_s001
                                 + 0.0254779501868655 w_k1_FDX_s002
                                 + 0.081318027409585 w_k1_FDX_s003
                                 + 0.0634829733777936 w_k1_FDX_s004
                                 + 0.0227964120764878 w_k1_FDX_s005
                                 + 0.0511605116140996 w_k1_FDX_s006
                                 + 0.0322969409080835 w_k1_FDX_s007
                                 + 0.0898437888239313 w_k1_FDX_s010
                                 + 0.0191921714060064 w_k1_FDX_s011
                                 + 0.0636842731559762 w_k1_FDX_s012
                                 + 0.071221745689222 w_k1_FDX_s013
                                 + 0.0585303428528455 w_k1_FDX_s014
                                 + 0.0532271509170884 w_k1_FDX_s015
                                 + 0.0413231939314164 w_k1_FDX_s016
                                 + 0.0419022432137887 w_k1_FDX_s017
                                 + 0.00465971442437445 w_k1_FDX_s100
                                 + 0.00546836928087112 w_k1_FDX_s101
                                 + 0.0130742566800243 w_k1_FDX_s102
                                 + 0.00508987534861106 w_k1_FDX_s103
                                 + 0.00548475688387476 w_k1_FDX_s104
                                 + 0.00654179416267124 w_k1_FDX_s105
                                 + 0.00911288863835524 w_k1_FDX_s106
                                 + 0.00340988634284523 w_k1_FDX_s107
                                 + 0.0286807354027403 w_k1_FDX_s110
                                 + 0.034413809804908 w_k1_FDX_s111
                                 + 0.020550817260392 w_k1_FDX_s112
                                 + 0.0145163295325664 w_k1_FDX_s113
                                 + 0.0216855517642773 w_k1_FDX_s114
                                 + 0.0148146867712399 w_k1_FDX_s115
                                 + 0.0095091068375208 w_k1_FDX_s116
                                 + 0.0193762916297308 w_k1_FDX_s117
                                 + 0.0467931234503535 w_k1_IBM_s000
                                 + 0.0418868964930166 w_k1_IBM_s001
                                 + 0.0261294707488917 w_k1_IBM_s002
                                 + 0.0836625329327752 w_k1_IBM_s003
                                 + 0.0636033748056109 w_k1_IBM_s004
                                 + 0.0239674776349574 w_k1_IBM_s005
                                 + 0.053533365960696 w_k1_IBM_s006
                                 + 0.0339611004166865 w_k1_IBM_s007
                                 + 0.0906465851190012 w_k1_IBM_s010
                                 + 0.020313375646663 w_k1_IBM_s011
                                 + 0.0630125098751402 w_k1_IBM_s012
                                 + 0.0738023066256722 w_k1_IBM_s013
                                 + 0.0578150794589322 w_k1_IBM_s014
                                 + 0.0540769269616198 w_k1_IBM_s015
                                 + 0.0418827515283956 w_k1_IBM_s016
                                 + 0.0424231280699328 w_k1_IBM_s017
                                 + 0.00478556101188886 w_k1_IBM_s100
                                 + 0.00590795924230267 w_k1_IBM_s101
                                 + 0.0137800193215853 w_k1_IBM_s102
                                 + 0.00529427296835325 w_k1_IBM_s103
                                 + 0.00540143012422686 w_k1_IBM_s104
                                 + 0.00659224623655044 w_k1_IBM_s105
                                 + 0.00924200484189793 w_k1_IBM_s106
                                 + 0.00337484905184067 w_k1_IBM_s107
                                 + 0.0294623892222972 w_k1_IBM_s110
                                 + 0.035107151950868 w_k1_IBM_s111
                                 + 0.0210958844057765 w_k1_IBM_s112
                                 + 0.0150457951665562 w_k1_IBM_s113
                                 + 0.0211185943715413 w_k1_IBM_s114
                                 + 0.0148966027299358 w_k1_IBM_s115
                                 + 0.00895395481514008 w_k1_IBM_s116
                                 + 0.0200194301694396 w_k1_IBM_s117
                                 + 0.0532152946789433 w_k1_INTC_s000
                                 + 0.0386149335680769 w_k1_INTC_s001
                                 + 0.0247028243982814 w_k1_INTC_s002
                                 + 0.0793777070928429 w_k1_INTC_s003
                                 + 0.067006600905678 w_k1_INTC_s004
                                 + 0.023005699850053 w_k1_INTC_s005
                                 + 0.0483309939999315 w_k1_INTC_s006
                                 + 0.034024609148748 w_k1_INTC_s007
                                 + 0.0882931125903852 w_k1_INTC_s010
                                 + 0.0210411261744015 w_k1_INTC_s011
                                 + 0.0608149752947301 w_k1_INTC_s012
                                 + 0.0657910679973842 w_k1_INTC_s013
                                 + 0.065709903709039 w_k1_INTC_s014
                                 + 0.0607400468477259 w_k1_INTC_s015
                                 + 0.038632995880441 w_k1_INTC_s016
                                 + 0.0378357191555228 w_k1_INTC_s017
                                 + 0.00390059506812417 w_k1_INTC_s100
                                 + 0.00626231490663334 w_k1_INTC_s101
                                 + 0.0134389047844826 w_k1_INTC_s102
                                 + 0.00592091026065497 w_k1_INTC_s103
                                 + 0.00560879365973629 w_k1_INTC_s104
                                 + 0.00729696928213219 w_k1_INTC_s105
                                 + 0.0110246542890077 w_k1_INTC_s106
                                 + 0.00323264550300852 w_k1_INTC_s107
                                 + 0.0296176089969894 w_k1_INTC_s110
                                 + 0.0347578292716237 w_k1_INTC_s111
                                 + 0.0198646192979154 w_k1_INTC_s112
                                 + 0.0154422013562004 w_k1_INTC_s113
                                 + 0.0236537326523723 w_k1_INTC_s114
                                 + 0.0158294321346513 w_k1_INTC_s115
                                 + 0.00846152684452921 w_k1_INTC_s116
                                 + 0.0205906696974968 w_k1_INTC_s117
                                 + 0.0472189156571525 w_k1_KO_s000
                                 + 0.0403882030404504 w_k1_KO_s001
                                 + 0.0217948142544397 w_k1_KO_s002
                                 + 0.0781271784409531 w_k1_KO_s003
                                 + 0.059670493210509 w_k1_KO_s004
                                 + 0.0217889030643315 w_k1_KO_s005
                                 + 0.0504068761398982 w_k1_KO_s006
                                 + 0.0339879810155032 w_k1_KO_s007
                                 + 0.0852957271477344 w_k1_KO_s010
                                 + 0.0186229251848977 w_k1_KO_s011
                                 + 0.057330831877209 w_k1_KO_s012
                                 + 0.0728900227484607 w_k1_KO_s013
                                 + 0.0558219813884051 w_k1_KO_s014
                                 + 0.0548692453472236 w_k1_KO_s015
                                 + 0.0405927979153248 w_k1_KO_s016
                                 + 0.0415245450938948 w_k1_KO_s017
                                 + 0.00466538669945472 w_k1_KO_s100
                                 + 0.00611113190517455 w_k1_KO_s101
                                 + 0.0136591145568294 w_k1_KO_s102
                                 + 0.00520355445751561 w_k1_KO_s103
                                 + 0.00532425861500701 w_k1_KO_s104
                                 + 0.00732255498432885 w_k1_KO_s105
                                 + 0.00924620538499836 w_k1_KO_s106
                                 + 0.00316286897223668 w_k1_KO_s107
                                 + 0.029281386808567 w_k1_KO_s110
                                 + 0.0354916305120199 w_k1_KO_s111
                                 + 0.0200159179470953 w_k1_KO_s112
                                 + 0.0158395643355824 w_k1_KO_s113
                                 + 0.0227072450283701 w_k1_KO_s114
                                 + 0.01732468447566 w_k1_KO_s115
                                 + 0.00885782394992233 w_k1_KO_s116
                                 + 0.0206415363631465 w_k1_KO_s117
                                 + 0.0440182476016047 w_k1_MAT_s000
                                 + 0.0423226691637861 w_k1_MAT_s001
                                 + 0.024927601959863 w_k1_MAT_s002
                                 + 0.0987748779677033 w_k1_MAT_s003
                                 + 0.0648750705868516 w_k1_MAT_s004
                                 + 0.0223686984478919 w_k1_MAT_s005
                                 + 0.0537432340541222 w_k1_MAT_s006
                                 + 0.0310754289934273 w_k1_MAT_s007
                                 + 0.0989022135429959 w_k1_MAT_s010
                                 + 0.0222226433077759 w_k1_MAT_s011
                                 + 0.0688862916615145 w_k1_MAT_s012
                                 + 0.0716645832506904 w_k1_MAT_s013
                                 + 0.047130605604522 w_k1_MAT_s014
                                 + 0.0537862095105966 w_k1_MAT_s015
                                 + 0.0450378946319406 w_k1_MAT_s016
                                 + 0.0416322505265584 w_k1_MAT_s017
                                 + 0.00470434755333947 w_k1_MAT_s100
                                 + 0.00461411762295688 w_k1_MAT_s101
                                 + 0.0131458020547939 w_k1_MAT_s102
                                 + 0.00456083358114836 w_k1_MAT_s103
                                 + 0.00521332333788158 w_k1_MAT_s104
                                 + 0.0053816963753132 w_k1_MAT_s105
                                 + 0.00824802399153411 w_k1_MAT_s106
                                 + 0.00296504781875645 w_k1_MAT_s107
                                 + 0.0237497855316518 w_k1_MAT_s110
                                 + 0.0388889584152088 w_k1_MAT_s111
                                 + 0.022486514574207 w_k1_MAT_s112
                                 + 0.0132523611886242 w_k1_MAT_s113
                                 + 0.0193115047118033 w_k1_MAT_s114
                                 + 0.0160636944769374 w_k1_MAT_s115
                                 + 0.00955475700676227 w_k1_MAT_s116
                                 + 0.0206551986410132 w_k1_MAT_s117
                                 + 0.0491573196193622 w_k1_MSFT_s000
                                 + 0.0391637788896993 w_k1_MSFT_s001
                                 + 0.0248038596078707 w_k1_MSFT_s002
                                 + 0.0854228000787544 w_k1_MSFT_s003
                                 + 0.0552424558514297 w_k1_MSFT_s004
                                 + 0.0242178037047997 w_k1_MSFT_s005
                                 + 0.0491319062762731 w_k1_MSFT_s006
                                 + 0.0351792771754196 w_k1_MSFT_s007
                                 + 0.090370502708315 w_k1_MSFT_s010
                                 + 0.0201406496324644 w_k1_MSFT_s011
                                 + 0.0637539393610466 w_k1_MSFT_s012
                                 + 0.0722207486710087 w_k1_MSFT_s013
                                 + 0.0569267945043052 w_k1_MSFT_s014
                                 + 0.0539849266630395 w_k1_MSFT_s015
                                 + 0.0374545626161183 w_k1_MSFT_s016
                                 + 0.0387090576747024 w_k1_MSFT_s017
                                 + 0.00434101358774017 w_k1_MSFT_s100
                                 + 0.00558821734445318 w_k1_MSFT_s101
                                 + 0.013535375317531 w_k1_MSFT_s102
                                 + 0.00536041519612675 w_k1_MSFT_s103
                                 + 0.00581626526793991 w_k1_MSFT_s104
                                 + 0.00642630036588961 w_k1_MSFT_s105
                                 + 0.00979250483613655 w_k1_MSFT_s106
                                 + 0.00326782357290559 w_k1_MSFT_s107
                                 + 0.0324502302205849 w_k1_MSFT_s110
                                 + 0.0359117610649499 w_k1_MSFT_s111
                                 + 0.0205953089056192 w_k1_MSFT_s112
                                 + 0.014128183578546 w_k1_MSFT_s113
                                 + 0.0203667438177796 w_k1_MSFT_s114
                                 + 0.0136346891741326 w_k1_MSFT_s115
                                 + 0.00922954784327987 w_k1_MSFT_s116
                                 + 0.020832800455535 w_k1_MSFT_s117
                                 + 0.0459736057328165 w_k1_PG_s000
                                 + 0.0405015457563394 w_k1_PG_s001
                                 + 0.0235761327974534 w_k1_PG_s002
                                 + 0.0762096291828458 w_k1_PG_s003
                                 + 0.060080368011835 w_k1_PG_s004
                                 + 0.021573080415972 w_k1_PG_s005
                                 + 0.0547840528732345 w_k1_PG_s006
                                 + 0.0276006838706697 w_k1_PG_s007
                                 + 0.0850419869475693 w_k1_PG_s010
                                 + 0.0191727638952004 w_k1_PG_s011
                                 + 0.0642335086425978 w_k1_PG_s012
                                 + 0.0763490447886639 w_k1_PG_s013
                                 + 0.0548444253326436 w_k1_PG_s014
                                 + 0.0519123184315638 w_k1_PG_s015
                                 + 0.043555747118438 w_k1_PG_s016
                                 + 0.0409151073432383 w_k1_PG_s017
                                 + 0.00493149453973939 w_k1_PG_s100
                                 + 0.00544009906918243 w_k1_PG_s101
                                 + 0.0131399093818502 w_k1_PG_s102
                                 + 0.00469906330057399 w_k1_PG_s103
                                 + 0.00539038418130464 w_k1_PG_s104
                                 + 0.00641968770154387 w_k1_PG_s105
                                 + 0.00908955883189775 w_k1_PG_s106
                                 + 0.00316559787902801 w_k1_PG_s107
                                 + 0.028423363149715 w_k1_PG_s110
                                 + 0.032067154084746 w_k1_PG_s111
                                 + 0.0189174482648149 w_k1_PG_s112
                                 + 0.0145312496338207 w_k1_PG_s113
                                 + 0.0233445171969855 w_k1_PG_s114
                                 + 0.0159406566924217 w_k1_PG_s115
                                 + 0.00834755679731038 w_k1_PG_s116
                                 + 0.0185778029285497 w_k1_PG_s117 >= 1.06
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
