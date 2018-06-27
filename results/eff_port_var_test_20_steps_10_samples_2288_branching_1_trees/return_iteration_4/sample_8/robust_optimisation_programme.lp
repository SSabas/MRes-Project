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
 w_k1_ADM_s0_return_constraint:  1.18937 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.901932 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.07812 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.840898 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.95606 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.0101 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.85863 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.45647 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.59957 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.05703 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.790639 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.1787 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.962692 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.27316 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.877345 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.34229 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.935237 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.858562 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.28415 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.95546 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.921394 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.33994 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.12049 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.84508 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.863506 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.2472 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.812883 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.25828 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.972398 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.41437 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.804595 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.46049 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.0547 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.616393 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.881193 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.852523 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.08079 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.888856 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.28152 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.872972 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.17648 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.935137 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.04101 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.12599 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.850013 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.38579 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.51638 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.07809 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.793948 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.27825 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.693669 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.08634 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.649619 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.39673 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.737233 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.16223 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.07537 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.867617 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.98067 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.63088 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.792774 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.523459 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.722733 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.01916 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.676001 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.19592 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.751322 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.45166 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.750581 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.8183 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.24002 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.511707 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.700227 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.89355 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.07183 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.783787 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.31592 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.894127 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.27083 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.869593 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.958555 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.06182 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.835772 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.45984 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.46717 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.890021 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.625508 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.39115 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.791631 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.07082 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.675731 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.69079 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.734885 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.07051 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.10355 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.796871 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.11871 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.53143 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.91951 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.527124 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.773348 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.11241 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.581931 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.07855 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.673291 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.68068 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.669971 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 2.08587 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.36485 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.498527 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.70504 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.871058 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.1512 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.793555 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.17465 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.912717 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.17383 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.865182 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.02676 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.98196 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.842353 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.37383 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.637329 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.969488 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.821318 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.18318 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.879311 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.20589 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.724509 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.35724 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.868197 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.94947 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.17847 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.895447 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.04446 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.28249 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.04507 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.658314 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.847261 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.04006 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.703208 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.982036 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.720306 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.46195 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.707124 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.59206 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.14858 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.597789 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.783082 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.900783 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.10757 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.84642 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.23471 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.915464 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.919558 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.09239 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.852394 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.40958 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.11187 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.15977 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.560321 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.30765 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.775153 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.55188 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.669725 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.783311 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.08672 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.32588 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.671628 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.50086 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.853483 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.03026 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.169 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.85465 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.628807 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.601221 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.908293 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.03149 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.809073 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.58072 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.12609 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.2684 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.23351 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.30169 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.46662 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.656369 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.733203 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.05658 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.892566 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.826261 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.11171 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.88929 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.13015 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.893009 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.0683 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.10353 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.01099 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.0325 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.743532 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.00532 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.806988 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.05579 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.839162 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.22253 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.930202 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.39173 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.838967 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.23808 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.23152 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.946207 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.17133 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.16641 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.06942 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.720172 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.874566 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.10143 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.741653 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.09073 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.807579 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.14121 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.752475 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.41828 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.19796 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.699126 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.7812 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.995692 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.966017 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.938651 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.21266 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.894027 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.13961 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.922886 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.950024 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.15068 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.95134 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.990799 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.547188 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.988747 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.678224 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.16962 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.79822 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.955224 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.97382 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.39891 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.788765 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.27135 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.05598 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.962307 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.14015 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.4843 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.863017 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.6513 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.883977 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.997688 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.692003 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.23182 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.929078 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.23722 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.02107 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.45996 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.38928 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.612014 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.780385 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.13238 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.12806 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.905061 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.40765 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.881852 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.972768 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.00383 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.987472 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.25611 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.928269 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.25786 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.518747 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.2211 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.786263 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.64173 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.532903 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.746226 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.980137 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.30654 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.786159 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.72945 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.910714 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.961963 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.02862 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 2.13624 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.55544 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.490323 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 0.761461 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.821952 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.636241 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.55686 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.845882 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.21337 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.03059 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.32713 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.55055 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.687096 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.780478 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.994382 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.22212 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.90428 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.13272 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.920654 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.17891 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.881802 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.05289 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.02413 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.90605 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.15128 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.709854 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.954725 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.731721 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.13227 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.969586 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.15993 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.855069 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.39538 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.90717 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.0436 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.2053 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.862932 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.09599 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.20594 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.03898 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.694479 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.86159 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.99086 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.653437 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.983829 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.771312 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.3092 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.740385 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.30891 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.12807 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.708674 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.835216 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.94574 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.07531 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.899749 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.15301 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.920961 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.12395 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.923786 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.994831 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.12214 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.877037 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.08671 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.678822 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.01757 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.765399 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.06859 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.826725 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.985749 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.940262 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.32302 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.919596 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.04502 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.03587 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.949047 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.07347 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.39 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.916063 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.74258 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.94157 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.05542 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.786259 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.12574 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.889794 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.20912 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.85427 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.30401 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.03242 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.712846 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.767237 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.906501 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.03286 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.922496 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.326998500190249 z_k1_s0000
                                 - 0.153999643846535 z_k1_s0001
                                 - 0.183999597702296 z_k1_s0002
                                 - 0.527999290101148 z_k1_s0003
                                 - 0.151999578762985 z_k1_s0004
                                 - 0.554998251733784 z_k1_s0005
                                 - 0.13499966455124 z_k1_s0006
                                 - 0.520998423310295 z_k1_s0007
                                 - 0.385999016312537 z_k1_s0010
                                 - 0.180000116057752 z_k1_s0011
                                 - 0.190999949633476 z_k1_s0012
                                 - 0.199000122324959 z_k1_s0013
                                 - 0.173000045727718 z_k1_s0014
                                 - 0.427000412931468 z_k1_s0015
                                 - 0.248000830035234 z_k1_s0016
                                 - 0.349999208501021 z_k1_s0017
                                 - 0.113999926310597 z_k1_s0020
                                 - 0.0650000788649213 z_k1_s0021
                                 - 0.125000067086312 z_k1_s0022
                                 - 0.0759997065401806 z_k1_s0023
                                 - 0.0440000500024052 z_k1_s0024
                                 - 0.10500014431258 z_k1_s0025
                                 - 0.159000302301982 z_k1_s0026
                                 - 0.0450000094910593 z_k1_s0027
                                 - 0.209999770623346 z_k1_s0030
                                 - 0.068000021925646 z_k1_s0031
                                 - 0.121999866743087 z_k1_s0032
                                 - 0.0770000120952179 z_k1_s0033
                                 - 0.0840000044493295 z_k1_s0034
                                 - 0.082000006633869 z_k1_s0035
                                 - 0.0999998907730218 z_k1_s0036
                                 - 0.219000241792873 z_k1_s0037
                                 - 0.189000187193209 z_k1_s0040
                                 - 0.529999330948541 z_k1_s0041
                                 - 0.123000025014574 z_k1_s0042
                                 - 0.11700002875289 z_k1_s0043
                                 - 0.245001033269418 z_k1_s0044
                                 - 0.114000132272105 z_k1_s0045
                                 - 0.184000020891711 z_k1_s0046
                                 - 0.530999567508955 z_k1_s0047
                                 - 0.251000615470668 z_k1_s0050
                                 - 0.372001910185345 z_k1_s0051
                                 - 0.205000469311417 z_k1_s0052
                                 - 0.915003817550436 z_k1_s0053
                                 - 0.287000729856168 z_k1_s0054
                                 - 0.577001651294922 z_k1_s0055
                                 - 0.229000545568418 z_k1_s0056
                                 - 0.805001647534584 z_k1_s0057
                                 - 0.160999649296645 z_k1_s0060
                                 - 0.290999328391524 z_k1_s0061
                                 - 0.408997812413862 z_k1_s0062
                                 - 0.144999654363495 z_k1_s0063
                                 - 0.66899972819867 z_k1_s0064
                                 - 0.314999480640939 z_k1_s0065
                                 - 0.619999404035308 z_k1_s0066
                                 - 0.587000033903239 z_k1_s0067
                                 - 0.262000894789353 z_k1_s0070
                                 - 0.357001835115295 z_k1_s0071
                                 - 0.172000513513434 z_k1_s0072
                                 - 0.374001154118841 z_k1_s0073
                                 - 0.194000391567003 z_k1_s0074
                                 - 0.261000460258532 z_k1_s0075
                                 - 0.0800002716784245 z_k1_s0076
                                 - 0.317000589923276 z_k1_s0077
                                 - 0.292000115827835 z_k1_s0100
                                 - 0.305999348778432 z_k1_s0101
                                 - 0.202999405784374 z_k1_s0102
                                 - 0.107999991333579 z_k1_s0103
                                 - 0.264000145926538 z_k1_s0104
                                 - 0.0929999967150268 z_k1_s0105
                                 - 0.152000266321353 z_k1_s0106
                                 - 0.0860000794397078 z_k1_s0107
                                 - 0.161999930681185 z_k1_s0110
                                 - 0.363999335032002 z_k1_s0111
                                 - 0.161999930681185 z_k1_s0112
                                 - 0.243000689221462 z_k1_s0113
                                 - 0.176999931607218 z_k1_s0114
                                 - 0.137999889539549 z_k1_s0115
                                 - 0.472998852621369 z_k1_s0116
                                 - 0.264000650857924 z_k1_s0117
                                 - 1.44600062277995 z_k1_s0120
                                 - 0.667000238720173 z_k1_s0121
                                 - 0.331000534001941 z_k1_s0122
                                 - 0.482000207593316 z_k1_s0123
                                 - 1.12900376344832 z_k1_s0124
                                 - 1.21899761194845 z_k1_s0125
                                 - 1.58600080445445 z_k1_s0126
                                 - 0.453000483375272 z_k1_s0127
                                 - 0.234999429334881 z_k1_s0130
                                 - 0.344999899448545 z_k1_s0131
                                 - 0.175000522908266 z_k1_s0132
                                 - 0.122999985672944 z_k1_s0133
                                 - 0.343000100900872 z_k1_s0134
                                 - 0.102999855197765 z_k1_s0135
                                 - 0.174000623634429 z_k1_s0136
                                 - 0.151999869612175 z_k1_s0137
                                 - 0.679002304541728 z_k1_s0140
                                 - 0.308000225224266 z_k1_s0141
                                 - 0.241000200700182 z_k1_s0142
                                 - 1.15099911520643 z_k1_s0143
                                 - 0.564002043724043 z_k1_s0144
                                 - 0.244000189299755 z_k1_s0145
                                 - 0.516002226130876 z_k1_s0146
                                 - 0.484000952067322 z_k1_s0147
                                 - 0.347000460356133 z_k1_s0150
                                 - 0.107000029875819 z_k1_s0151
                                 - 0.125000053241844 z_k1_s0152
                                 - 0.180000233660242 z_k1_s0153
                                 - 0.213999702951667 z_k1_s0154
                                 - 0.388999385010279 z_k1_s0155
                                 - 0.247999172243092 z_k1_s0156
                                 - 0.173999928538258 z_k1_s0157
                                 - 0.259000700315057 z_k1_s0160
                                 - 0.185999841686074 z_k1_s0161
                                 - 0.320000051416726 z_k1_s0162
                                 - 0.340999396091304 z_k1_s0163
                                 - 0.420999408945485 z_k1_s0164
                                 - 0.112000017995854 z_k1_s0165
                                 - 0.134999927591482 z_k1_s0166
                                 - 0.108000017353145 z_k1_s0167
                                 - 0.212999952076958 z_k1_s0170
                                 - 0.36299923849783 z_k1_s0171
                                 - 0.305999158618146 z_k1_s0172
                                 - 0.195999755042789 z_k1_s0173
                                 - 0.507000133140755 z_k1_s0174
                                 - 0.21199992758084 z_k1_s0175
                                 - 0.223000197038144 z_k1_s0176
                                 - 0.173999874327717 z_k1_s0177
                                 - 0.404000237173295 z_k1_s1000
                                 - 0.651999081527885 z_k1_s1001
                                 - 0.317999892978144 z_k1_s1002
                                 - 0.584998651823608 z_k1_s1003
                                 - 0.352999854040219 z_k1_s1004
                                 - 0.596999625501514 z_k1_s1005
                                 - 0.298000022514046 z_k1_s1006
                                 - 0.548000656314095 z_k1_s1007
                                 - 0.624001212916279 z_k1_s1010
                                 - 0.281000163778294 z_k1_s1011
                                 - 0.199000042518786 z_k1_s1012
                                 - 0.369001070517828 z_k1_s1013
                                 - 0.619998356596075 z_k1_s1014
                                 - 0.449999427997201 z_k1_s1015
                                 - 0.620999070676126 z_k1_s1016
                                 - 0.334999885197941 z_k1_s1017
                                 - 0.499999047837773 z_k1_s1020
                                 - 0.530999824323728 z_k1_s1021
                                 - 0.345001133408091 z_k1_s1022
                                 - 0.492998583728036 z_k1_s1023
                                 - 0.466998991320478 z_k1_s1024
                                 - 0.355000517564837 z_k1_s1025
                                 - 0.116999956234042 z_k1_s1026
                                 - 0.1759999035589 z_k1_s1027
                                 - 1.25500444313072 z_k1_s1030
                                 - 1.73599733543192 z_k1_s1031
                                 - 1.99200171609377 z_k1_s1032
                                 - 1.12199996526385 z_k1_s1033
                                 - 1.94500213515171 z_k1_s1034
                                 - 1.21200380141797 z_k1_s1035
                                 - 2.17700083544003 z_k1_s1036
                                 - 0.605999491709449 z_k1_s1037
                                 - 0.381998410352071 z_k1_s1040
                                 - 0.42199958319478 z_k1_s1041
                                 - 0.707998233123347 z_k1_s1042
                                 - 0.420999267523806 z_k1_s1043
                                 - 0.609001343685666 z_k1_s1044
                                 - 0.242000032400709 z_k1_s1045
                                 - 0.452001417326497 z_k1_s1046
                                 - 0.582000456566869 z_k1_s1047
                                 - 0.779000228478913 z_k1_s1050
                                 - 0.243000187903054 z_k1_s1051
                                 - 0.366000293957699 z_k1_s1052
                                 - 0.929001438790682 z_k1_s1053
                                 - 0.47500013931641 z_k1_s1054
                                 - 0.513000150461723 z_k1_s1055
                                 - 0.601999243511528 z_k1_s1056
                                 - 0.525002020088677 z_k1_s1057
                                 - 0.3429996918902 z_k1_s1060
                                 - 0.398999860974233 z_k1_s1061
                                 - 0.283000094299948 z_k1_s1062
                                 - 1.03900001193527 z_k1_s1063
                                 - 0.906001061610149 z_k1_s1064
                                 - 0.322999969360062 z_k1_s1065
                                 - 0.238999930733934 z_k1_s1066
                                 - 0.768001213152857 z_k1_s1067
                                 - 0.467998946834811 z_k1_s1070
                                 - 2.30399671095037 z_k1_s1071
                                 - 1.73599686573538 z_k1_s1072
                                 - 0.547998022543824 z_k1_s1073
                                 - 0.538998516737903 z_k1_s1074
                                 - 0.761998137782758 z_k1_s1075
                                 - 1.07999820154 z_k1_s1076
                                 - 0.777997624325535 z_k1_s1077
                                 - 0.187000191916326 z_k1_s1100
                                 - 0.255000261704081 z_k1_s1101
                                 - 0.0520000591494649 z_k1_s1102
                                 - 0.153999868930829 z_k1_s1103
                                 - 0.0570000931927085 z_k1_s1104
                                 - 0.0680000697877549 z_k1_s1105
                                 - 0.0600000153185653 z_k1_s1106
                                 - 0.150000038296413 z_k1_s1107
                                 - 0.208000712565136 z_k1_s1110
                                 - 0.137000144770071 z_k1_s1111
                                 - 0.0990001024630423 z_k1_s1112
                                 - 0.209999458011964 z_k1_s1113
                                 - 0.230999551213268 z_k1_s1114
                                 - 0.293999830817177 z_k1_s1115
                                 - 0.0740000125662681 z_k1_s1116
                                 - 0.220999927974857 z_k1_s1117
                                 - 0.465999372509857 z_k1_s1120
                                 - 0.361998785983377 z_k1_s1121
                                 - 0.347000932063359 z_k1_s1122
                                 - 0.23500013161161 z_k1_s1123
                                 - 0.417999342959398 z_k1_s1124
                                 - 0.365999049696158 z_k1_s1125
                                 - 0.490000954786541 z_k1_s1126
                                 - 0.451001518589839 z_k1_s1127
                                 - 0.120000003860264 z_k1_s1130
                                 - 0.225000684738672 z_k1_s1131
                                 - 0.0970001024871462 z_k1_s1132
                                 - 0.0680000383208525 z_k1_s1133
                                 - 0.192000439776856 z_k1_s1134
                                 - 0.231000630731631 z_k1_s1135
                                 - 0.0730001017150934 z_k1_s1136
                                 - 0.0780000567092259 z_k1_s1137
                                 - 0.15900001339192 z_k1_s1140
                                 - 0.387000934718499 z_k1_s1141
                                 - 0.336000910375798 z_k1_s1142
                                 - 0.299000392715169 z_k1_s1143
                                 - 0.122000133231594 z_k1_s1144
                                 - 0.465999972425342 z_k1_s1145
                                 - 0.115999980367676 z_k1_s1146
                                 - 0.239999739553707 z_k1_s1147
                                 - 0.450999782102578 z_k1_s1150
                                 - 0.1899997582588 z_k1_s1151
                                 - 0.69999993389927 z_k1_s1152
                                 - 0.3189983174901 z_k1_s1153
                                 - 0.174999591825563 z_k1_s1154
                                 - 0.154999631014085 z_k1_s1155
                                 - 0.309998217630158 z_k1_s1156
                                 - 0.310998228725707 z_k1_s1157
                                 - 0.338000908025211 z_k1_s1160
                                 - 0.300000828131305 z_k1_s1161
                                 - 0.293000690045295 z_k1_s1162
                                 - 0.90200145230168 z_k1_s1163
                                 - 0.577000398892398 z_k1_s1164
                                 - 1.32900424993505 z_k1_s1165
                                 - 0.377000940674104 z_k1_s1166
                                 - 0.572002979122985 z_k1_s1167
                                 - 0.19500039258737 z_k1_s1170
                                 - 0.0800001174922097 z_k1_s1171
                                 - 0.132000363782321 z_k1_s1172
                                 - 0.0750001101489466 z_k1_s1173
                                 - 0.083000058178102 z_k1_s1174
                                 - 0.150000220297893 z_k1_s1175
                                 - 0.185000590301063 z_k1_s1176
                                 - 0.161999770641244 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.33804 w_k1_ADM_s000
                                 + 1.33703 w_k1_AXP_s000 + 1.4294 w_k1_DIS_s000
                                 + 1.25852 w_k1_FDX_s000
                                 + 1.07996 w_k1_IBM_s000
                                 + 1.19735 w_k1_INTC_s000
                                 + 1.27284 w_k1_KO_s000 + 1.11973 w_k1_MAT_s000
                                 + 1.32184 w_k1_MSFT_s000
                                 + 1.22078 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.02926 w_k1_ADM_s000
                                 + 0.738747 w_k1_AXP_s000
                                 + 0.707196 w_k1_DIS_s000
                                 + 0.750849 w_k1_FDX_s000
                                 + 1.27945 w_k1_IBM_s000
                                 + 0.646222 w_k1_INTC_s000
                                 + 0.990624 w_k1_KO_s000
                                 + 1.31226 w_k1_MAT_s000
                                 + 0.82412 w_k1_MSFT_s000
                                 + 0.967649 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.05476 w_k1_ADM_s000
                                 + 1.05987 w_k1_AXP_s000
                                 + 1.48138 w_k1_DIS_s000
                                 + 1.21324 w_k1_FDX_s000
                                 + 1.65442 w_k1_IBM_s000
                                 + 1.02385 w_k1_INTC_s000
                                 + 1.33835 w_k1_KO_s000 + 1.80615 w_k1_MAT_s000
                                 + 1.15066 w_k1_MSFT_s000
                                 + 1.24119 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.08631 w_k1_ADM_s000
                                 + 0.872707 w_k1_AXP_s000
                                 + 1.00655 w_k1_DIS_s000
                                 + 1.08013 w_k1_FDX_s000
                                 + 0.827837 w_k1_IBM_s000
                                 + 1.01557 w_k1_INTC_s000
                                 + 0.900203 w_k1_KO_s000
                                 + 0.925678 w_k1_MAT_s000
                                 + 1.1364 w_k1_MSFT_s000
                                 + 0.990203 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.29778 w_k1_ADM_s000
                                 + 0.961014 w_k1_AXP_s000
                                 + 1.67747 w_k1_DIS_s000
                                 + 1.31256 w_k1_FDX_s000
                                 + 0.936701 w_k1_IBM_s000
                                 + 1.07005 w_k1_INTC_s000
                                 + 1.12985 w_k1_KO_s000
                                 + 0.924843 w_k1_MAT_s000
                                 + 1.29466 w_k1_MSFT_s000
                                 + 1.05386 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.67621 w_k1_ADM_s000
                                 + 0.716448 w_k1_AXP_s000
                                 + 0.724886 w_k1_DIS_s000
                                 + 0.789768 w_k1_FDX_s000
                                 + 0.817375 w_k1_IBM_s000
                                 + 0.772962 w_k1_INTC_s000
                                 + 0.779409 w_k1_KO_s000
                                 + 0.976036 w_k1_MAT_s000
                                 + 0.782808 w_k1_MSFT_s000
                                 + 0.794999 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.26998 w_k1_ADM_s000
                                 + 1.44563 w_k1_AXP_s000
                                 + 1.12279 w_k1_DIS_s000
                                 + 1.10354 w_k1_FDX_s000
                                 + 1.22721 w_k1_IBM_s000
                                 + 0.917786 w_k1_INTC_s000
                                 + 1.08058 w_k1_KO_s000 + 1.40828 w_k1_MAT_s000
                                 + 1.03106 w_k1_MSFT_s000
                                 + 1.11883 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.933378 w_k1_ADM_s000
                                 + 0.965492 w_k1_AXP_s000
                                 + 1.03949 w_k1_DIS_s000
                                 + 0.945649 w_k1_FDX_s000
                                 + 1.11571 w_k1_IBM_s000
                                 + 0.883937 w_k1_INTC_s000
                                 + 1.18088 w_k1_KO_s000 + 1.14908 w_k1_MAT_s000
                                 + 1.05914 w_k1_MSFT_s000
                                 + 1.08442 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.88625 w_k1_ADM_s001
                                 + 0.706195 w_k1_AXP_s001
                                 + 0.942586 w_k1_DIS_s001
                                 + 0.846029 w_k1_FDX_s001
                                 + 0.836323 w_k1_IBM_s001
                                 + 0.979536 w_k1_INTC_s001
                                 + 1.08011 w_k1_KO_s001
                                 + 0.633732 w_k1_MAT_s001
                                 + 1.12428 w_k1_MSFT_s001
                                 + 1.02146 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.08064 w_k1_ADM_s001
                                 + 1.25602 w_k1_AXP_s001
                                 + 1.45295 w_k1_DIS_s001
                                 + 1.37034 w_k1_FDX_s001
                                 + 1.02754 w_k1_IBM_s001
                                 + 1.66969 w_k1_INTC_s001
                                 + 1.34944 w_k1_KO_s001
                                 + 0.917461 w_k1_MAT_s001
                                 + 1.27585 w_k1_MSFT_s001
                                 + 1.12627 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.15663 w_k1_ADM_s001
                                 + 1.28196 w_k1_AXP_s001
                                 + 1.00195 w_k1_DIS_s001 + 1.051 w_k1_FDX_s001
                                 + 1.17398 w_k1_IBM_s001
                                 + 0.928208 w_k1_INTC_s001
                                 + 1.15993 w_k1_KO_s001 + 1.29907 w_k1_MAT_s001
                                 + 1.05934 w_k1_MSFT_s001 + 1.1441 w_k1_PG_s001
                                 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.895461 w_k1_ADM_s001
                                 + 1.1202 w_k1_AXP_s001 + 1.00606 w_k1_DIS_s001
                                 + 1.10667 w_k1_FDX_s001
                                 + 0.875242 w_k1_IBM_s001
                                 + 1.16725 w_k1_INTC_s001
                                 + 0.984753 w_k1_KO_s001
                                 + 0.780163 w_k1_MAT_s001
                                 + 1.02962 w_k1_MSFT_s001
                                 + 0.994447 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.23613 w_k1_ADM_s001
                                 + 1.50854 w_k1_AXP_s001
                                 + 1.90102 w_k1_DIS_s001 + 1.3424 w_k1_FDX_s001
                                 + 1.45278 w_k1_IBM_s001
                                 + 1.77568 w_k1_INTC_s001
                                 + 2.09654 w_k1_KO_s001 + 1.76731 w_k1_MAT_s001
                                 + 1.70596 w_k1_MSFT_s001 + 1.4154 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.702974 w_k1_ADM_s001
                                 + 0.947673 w_k1_AXP_s001
                                 + 0.667748 w_k1_DIS_s001
                                 + 0.82068 w_k1_FDX_s001
                                 + 0.758083 w_k1_IBM_s001
                                 + 0.930647 w_k1_INTC_s001
                                 + 0.937954 w_k1_KO_s001
                                 + 0.965157 w_k1_MAT_s001
                                 + 0.96735 w_k1_MSFT_s001
                                 + 0.954794 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.04077 w_k1_ADM_s001
                                 + 1.1975 w_k1_AXP_s001 + 1.23876 w_k1_DIS_s001
                                 + 1.12361 w_k1_FDX_s001
                                 + 1.11957 w_k1_IBM_s001
                                 + 1.38026 w_k1_INTC_s001
                                 + 1.28469 w_k1_KO_s001 + 1.27579 w_k1_MAT_s001
                                 + 1.34701 w_k1_MSFT_s001
                                 + 1.11569 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.816019 w_k1_ADM_s001
                                 + 0.907194 w_k1_AXP_s001
                                 + 0.720381 w_k1_DIS_s001
                                 + 0.851275 w_k1_FDX_s001
                                 + 1.12579 w_k1_IBM_s001
                                 + 0.880779 w_k1_INTC_s001
                                 + 0.978141 w_k1_KO_s001
                                 + 1.15467 w_k1_MAT_s001
                                 + 0.831816 w_k1_MSFT_s001
                                 + 0.909197 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.4877 w_k1_ADM_s002
                                 + 1.09101 w_k1_AXP_s002
                                 + 1.18533 w_k1_DIS_s002
                                 + 1.27487 w_k1_FDX_s002
                                 + 0.85823 w_k1_IBM_s002
                                 + 0.997289 w_k1_INTC_s002
                                 + 0.951343 w_k1_KO_s002
                                 + 0.836578 w_k1_MAT_s002
                                 + 1.11872 w_k1_MSFT_s002 + 1.0814 w_k1_PG_s002
                                 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.08161 w_k1_ADM_s002
                                 + 0.699216 w_k1_AXP_s002
                                 + 0.722896 w_k1_DIS_s002
                                 + 0.916201 w_k1_FDX_s002
                                 + 0.586007 w_k1_IBM_s002
                                 + 0.899561 w_k1_INTC_s002
                                 + 0.697234 w_k1_KO_s002
                                 + 0.687974 w_k1_MAT_s002
                                 + 0.924284 w_k1_MSFT_s002
                                 + 0.879435 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.13198 w_k1_ADM_s002
                                 + 0.904975 w_k1_AXP_s002
                                 + 0.857235 w_k1_DIS_s002
                                 + 0.88731 w_k1_FDX_s002
                                 + 0.939607 w_k1_IBM_s002
                                 + 0.759991 w_k1_INTC_s002
                                 + 0.854759 w_k1_KO_s002
                                 + 0.693465 w_k1_MAT_s002
                                 + 0.794029 w_k1_MSFT_s002
                                 + 0.931279 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.987792 w_k1_ADM_s002
                                 + 1.16696 w_k1_AXP_s002
                                 + 0.999904 w_k1_DIS_s002
                                 + 0.958237 w_k1_FDX_s002
                                 + 1.71131 w_k1_IBM_s002
                                 + 0.957915 w_k1_INTC_s002
                                 + 1.11848 w_k1_KO_s002 + 1.77648 w_k1_MAT_s002
                                 + 0.911363 w_k1_MSFT_s002
                                 + 1.07692 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.800338 w_k1_ADM_s002
                                 + 0.584086 w_k1_AXP_s002
                                 + 0.653026 w_k1_DIS_s002
                                 + 0.682161 w_k1_FDX_s002
                                 + 1.13438 w_k1_IBM_s002
                                 + 0.816709 w_k1_INTC_s002
                                 + 0.825239 w_k1_KO_s002
                                 + 0.795043 w_k1_MAT_s002
                                 + 0.712812 w_k1_MSFT_s002
                                 + 0.890099 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.13071 w_k1_ADM_s002
                                 + 0.947016 w_k1_AXP_s002
                                 + 0.998231 w_k1_DIS_s002
                                 + 0.973534 w_k1_FDX_s002
                                 + 1.02738 w_k1_IBM_s002
                                 + 0.925801 w_k1_INTC_s002
                                 + 0.930491 w_k1_KO_s002
                                 + 0.665677 w_k1_MAT_s002
                                 + 0.872363 w_k1_MSFT_s002
                                 + 0.994715 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.25516 w_k1_ADM_s002
                                 + 1.049 w_k1_AXP_s002 + 1.36658 w_k1_DIS_s002
                                 + 1.13972 w_k1_FDX_s002
                                 + 1.19726 w_k1_IBM_s002
                                 + 1.06499 w_k1_INTC_s002
                                 + 1.16554 w_k1_KO_s002 + 1.03838 w_k1_MAT_s002
                                 + 1.06175 w_k1_MSFT_s002 + 1.0773 w_k1_PG_s002
                                 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.864764 w_k1_ADM_s002
                                 + 0.621157 w_k1_AXP_s002
                                 + 0.629783 w_k1_DIS_s002
                                 + 0.781383 w_k1_FDX_s002
                                 + 0.542011 w_k1_IBM_s002
                                 + 0.802974 w_k1_INTC_s002
                                 + 0.637583 w_k1_KO_s002
                                 + 0.32993 w_k1_MAT_s002
                                 + 0.760586 w_k1_MSFT_s002
                                 + 0.724576 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.04325 w_k1_ADM_s003
                                 + 0.98446 w_k1_AXP_s003
                                 + 0.974088 w_k1_DIS_s003
                                 + 1.02354 w_k1_FDX_s003
                                 + 0.744049 w_k1_IBM_s003
                                 + 0.937079 w_k1_INTC_s003
                                 + 0.750589 w_k1_KO_s003
                                 + 0.712284 w_k1_MAT_s003
                                 + 0.930415 w_k1_MSFT_s003
                                 + 0.895425 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.10175 w_k1_ADM_s003
                                 + 1.64097 w_k1_AXP_s003
                                 + 1.19607 w_k1_DIS_s003
                                 + 1.17968 w_k1_FDX_s003
                                 + 1.22533 w_k1_IBM_s003
                                 + 1.27171 w_k1_INTC_s003 + 1.1962 w_k1_KO_s003
                                 + 1.68651 w_k1_MAT_s003
                                 + 1.11546 w_k1_MSFT_s003
                                 + 1.05574 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.03703 w_k1_ADM_s003
                                 + 1.17137 w_k1_AXP_s003
                                 + 1.13274 w_k1_DIS_s003
                                 + 1.08232 w_k1_FDX_s003
                                 + 1.57388 w_k1_IBM_s003
                                 + 1.08447 w_k1_INTC_s003
                                 + 1.31986 w_k1_KO_s003 + 1.54224 w_k1_MAT_s003
                                 + 0.974409 w_k1_MSFT_s003
                                 + 1.0991 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.10176 w_k1_ADM_s003
                                 + 1.0823 w_k1_AXP_s003 + 1.00124 w_k1_DIS_s003
                                 + 1.14494 w_k1_FDX_s003
                                 + 0.991361 w_k1_IBM_s003
                                 + 1.09526 w_k1_INTC_s003
                                 + 0.955381 w_k1_KO_s003
                                 + 0.688149 w_k1_MAT_s003
                                 + 0.951623 w_k1_MSFT_s003
                                 + 0.904082 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.834461 w_k1_ADM_s003
                                 + 0.660313 w_k1_AXP_s003
                                 + 0.665013 w_k1_DIS_s003
                                 + 0.715452 w_k1_FDX_s003
                                 + 1.00051 w_k1_IBM_s003
                                 + 0.732337 w_k1_INTC_s003
                                 + 0.925877 w_k1_KO_s003
                                 + 1.10258 w_k1_MAT_s003
                                 + 0.760086 w_k1_MSFT_s003
                                 + 0.810863 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.07445 w_k1_ADM_s003
                                 + 1.21922 w_k1_AXP_s003
                                 + 1.18042 w_k1_DIS_s003
                                 + 1.08103 w_k1_FDX_s003
                                 + 1.03896 w_k1_IBM_s003
                                 + 1.10275 w_k1_INTC_s003
                                 + 1.12791 w_k1_KO_s003 + 1.29938 w_k1_MAT_s003
                                 + 1.12511 w_k1_MSFT_s003
                                 + 1.01931 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.856331 w_k1_ADM_s003
                                 + 1.40682 w_k1_AXP_s003
                                 + 1.44797 w_k1_DIS_s003
                                 + 1.27258 w_k1_FDX_s003
                                 + 1.03357 w_k1_IBM_s003
                                 + 1.16762 w_k1_INTC_s003 + 1.1403 w_k1_KO_s003
                                 + 1.01785 w_k1_MAT_s003
                                 + 1.18987 w_k1_MSFT_s003
                                 + 1.10109 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.703383 w_k1_ADM_s003
                                 + 0.934938 w_k1_AXP_s003
                                 + 0.842958 w_k1_DIS_s003
                                 + 0.883569 w_k1_FDX_s003
                                 + 1.03623 w_k1_IBM_s003
                                 + 0.792794 w_k1_INTC_s003
                                 + 0.871162 w_k1_KO_s003
                                 + 1.14362 w_k1_MAT_s003
                                 + 0.809707 w_k1_MSFT_s003
                                 + 0.950903 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.19381 w_k1_ADM_s004
                                 + 1.15985 w_k1_AXP_s004
                                 + 1.25091 w_k1_DIS_s004
                                 + 1.19247 w_k1_FDX_s004
                                 + 1.07056 w_k1_IBM_s004
                                 + 1.15005 w_k1_INTC_s004
                                 + 0.982054 w_k1_KO_s004
                                 + 0.781762 w_k1_MAT_s004
                                 + 1.27112 w_k1_MSFT_s004
                                 + 1.15418 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.934086 w_k1_ADM_s004
                                 + 0.85573 w_k1_AXP_s004
                                 + 0.797995 w_k1_DIS_s004
                                 + 0.878647 w_k1_FDX_s004
                                 + 0.700447 w_k1_IBM_s004
                                 + 0.807289 w_k1_INTC_s004
                                 + 0.803888 w_k1_KO_s004
                                 + 0.928491 w_k1_MAT_s004
                                 + 0.881153 w_k1_MSFT_s004
                                 + 0.900486 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.25282 w_k1_ADM_s004
                                 + 1.50544 w_k1_AXP_s004
                                 + 1.59006 w_k1_DIS_s004 + 1.2316 w_k1_FDX_s004
                                 + 1.7724 w_k1_IBM_s004
                                 + 1.55555 w_k1_INTC_s004 + 1.6124 w_k1_KO_s004
                                 + 1.95835 w_k1_MAT_s004
                                 + 1.59685 w_k1_MSFT_s004
                                 + 1.31901 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.29091 w_k1_ADM_s004
                                 + 1.03011 w_k1_AXP_s004 + 1.2778 w_k1_DIS_s004
                                 + 1.14579 w_k1_FDX_s004
                                 + 1.24773 w_k1_IBM_s004
                                 + 1.06539 w_k1_INTC_s004
                                 + 1.12718 w_k1_KO_s004 + 1.1142 w_k1_MAT_s004
                                 + 1.13049 w_k1_MSFT_s004
                                 + 1.21199 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.953672 w_k1_ADM_s004
                                 + 0.786965 w_k1_AXP_s004
                                 + 0.759931 w_k1_DIS_s004
                                 + 0.790563 w_k1_FDX_s004
                                 + 1.16907 w_k1_IBM_s004
                                 + 0.741641 w_k1_INTC_s004
                                 + 0.863474 w_k1_KO_s004
                                 + 0.902458 w_k1_MAT_s004
                                 + 0.866422 w_k1_MSFT_s004
                                 + 0.912765 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.04624 w_k1_ADM_s004
                                 + 1.19363 w_k1_AXP_s004
                                 + 1.41398 w_k1_DIS_s004
                                 + 1.07144 w_k1_FDX_s004
                                 + 1.46602 w_k1_IBM_s004
                                 + 1.0484 w_k1_INTC_s004 + 1.37828 w_k1_KO_s004
                                 + 1.71641 w_k1_MAT_s004
                                 + 1.12175 w_k1_MSFT_s004
                                 + 1.08596 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.31406 w_k1_ADM_s004
                                 + 1.56511 w_k1_AXP_s004
                                 + 1.42401 w_k1_DIS_s004
                                 + 1.36161 w_k1_FDX_s004
                                 + 1.00765 w_k1_IBM_s004
                                 + 1.4789 w_k1_INTC_s004 + 1.28129 w_k1_KO_s004
                                 + 0.986965 w_k1_MAT_s004
                                 + 1.35046 w_k1_MSFT_s004
                                 + 1.33866 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.13056 w_k1_ADM_s004
                                 + 0.989038 w_k1_AXP_s004
                                 + 0.972423 w_k1_DIS_s004
                                 + 0.928672 w_k1_FDX_s004
                                 + 1.09915 w_k1_IBM_s004
                                 + 1.1357 w_k1_INTC_s004 + 1.1871 w_k1_KO_s004
                                 + 1.06385 w_k1_MAT_s004
                                 + 1.18784 w_k1_MSFT_s004
                                 + 1.08348 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.744058 w_k1_ADM_s005
                                 + 0.647551 w_k1_AXP_s005
                                 + 0.727572 w_k1_DIS_s005
                                 + 0.671432 w_k1_FDX_s005
                                 + 1.16136 w_k1_IBM_s005
                                 + 0.875316 w_k1_INTC_s005
                                 + 1.07927 w_k1_KO_s005
                                 + 0.849131 w_k1_MAT_s005
                                 + 0.853024 w_k1_MSFT_s005
                                 + 0.960187 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.24546 w_k1_ADM_s005
                                 + 1.44957 w_k1_AXP_s005
                                 + 1.67535 w_k1_DIS_s005
                                 + 1.33399 w_k1_FDX_s005 + 1.0274 w_k1_IBM_s005
                                 + 1.37837 w_k1_INTC_s005 + 1.2858 w_k1_KO_s005
                                 + 0.926706 w_k1_MAT_s005
                                 + 1.2492 w_k1_MSFT_s005 + 1.05473 w_k1_PG_s005
                                 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.34524 w_k1_ADM_s005
                                 + 1.4747 w_k1_AXP_s005 + 1.78929 w_k1_DIS_s005
                                 + 1.2641 w_k1_FDX_s005 + 1.94476 w_k1_IBM_s005
                                 + 1.07693 w_k1_INTC_s005
                                 + 1.80481 w_k1_KO_s005 + 1.50183 w_k1_MAT_s005
                                 + 1.1428 w_k1_MSFT_s005 + 1.21036 w_k1_PG_s005
                                 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.990043 w_k1_ADM_s005
                                 + 0.878562 w_k1_AXP_s005
                                 + 0.96283 w_k1_DIS_s005
                                 + 0.939838 w_k1_FDX_s005
                                 + 0.869066 w_k1_IBM_s005
                                 + 0.911741 w_k1_INTC_s005
                                 + 0.891276 w_k1_KO_s005
                                 + 0.790014 w_k1_MAT_s005
                                 + 0.956618 w_k1_MSFT_s005
                                 + 0.893968 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.09262 w_k1_ADM_s005
                                 + 0.982952 w_k1_AXP_s005
                                 + 1.00021 w_k1_DIS_s005
                                 + 0.856893 w_k1_FDX_s005
                                 + 1.30603 w_k1_IBM_s005
                                 + 0.879622 w_k1_INTC_s005
                                 + 1.19193 w_k1_KO_s005 + 1.10386 w_k1_MAT_s005
                                 + 0.898878 w_k1_MSFT_s005
                                 + 1.09922 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.970854 w_k1_ADM_s005
                                 + 1.05928 w_k1_AXP_s005
                                 + 1.32474 w_k1_DIS_s005
                                 + 1.10531 w_k1_FDX_s005 + 1.0583 w_k1_IBM_s005
                                 + 1.291 w_k1_INTC_s005 + 1.23939 w_k1_KO_s005
                                 + 1.00457 w_k1_MAT_s005
                                 + 1.31935 w_k1_MSFT_s005
                                 + 1.23932 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.02766 w_k1_ADM_s005
                                 + 1.15044 w_k1_AXP_s005
                                 + 1.14818 w_k1_DIS_s005
                                 + 1.02476 w_k1_FDX_s005
                                 + 1.62306 w_k1_IBM_s005
                                 + 1.02597 w_k1_INTC_s005
                                 + 1.32879 w_k1_KO_s005 + 1.38271 w_k1_MAT_s005
                                 + 1.05382 w_k1_MSFT_s005
                                 + 1.12324 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.802107 w_k1_ADM_s005
                                 + 0.799957 w_k1_AXP_s005
                                 + 0.706719 w_k1_DIS_s005
                                 + 0.83386 w_k1_FDX_s005
                                 + 0.704386 w_k1_IBM_s005
                                 + 0.915107 w_k1_INTC_s005
                                 + 0.759048 w_k1_KO_s005
                                 + 0.754048 w_k1_MAT_s005
                                 + 0.854817 w_k1_MSFT_s005
                                 + 0.792514 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.67668 w_k1_ADM_s006
                                 + 0.71546 w_k1_AXP_s006
                                 + 0.885818 w_k1_DIS_s006
                                 + 0.812337 w_k1_FDX_s006
                                 + 1.06671 w_k1_IBM_s006
                                 + 0.956853 w_k1_INTC_s006
                                 + 0.86831 w_k1_KO_s006 + 1.03535 w_k1_MAT_s006
                                 + 0.856963 w_k1_MSFT_s006
                                 + 0.889903 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.890609 w_k1_ADM_s006
                                 + 0.94087 w_k1_AXP_s006
                                 + 1.06324 w_k1_DIS_s006
                                 + 1.15622 w_k1_FDX_s006
                                 + 0.683707 w_k1_IBM_s006
                                 + 0.998781 w_k1_INTC_s006
                                 + 0.831198 w_k1_KO_s006
                                 + 0.806975 w_k1_MAT_s006
                                 + 1.05207 w_k1_MSFT_s006
                                 + 0.940702 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.582344 w_k1_ADM_s006
                                 + 0.815671 w_k1_AXP_s006
                                 + 0.644978 w_k1_DIS_s006
                                 + 0.732541 w_k1_FDX_s006
                                 + 1.04859 w_k1_IBM_s006
                                 + 0.824084 w_k1_INTC_s006
                                 + 0.726076 w_k1_KO_s006
                                 + 1.14142 w_k1_MAT_s006
                                 + 0.792071 w_k1_MSFT_s006
                                 + 0.901651 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.35869 w_k1_ADM_s006
                                 + 1.64072 w_k1_AXP_s006
                                 + 1.35819 w_k1_DIS_s006 + 1.4021 w_k1_FDX_s006
                                 + 0.935174 w_k1_IBM_s006
                                 + 1.04784 w_k1_INTC_s006
                                 + 0.833694 w_k1_KO_s006
                                 + 1.04696 w_k1_MAT_s006
                                 + 0.870099 w_k1_MSFT_s006
                                 + 1.03882 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.876426 w_k1_ADM_s006
                                 + 1.07602 w_k1_AXP_s006
                                 + 1.00333 w_k1_DIS_s006
                                 + 1.01688 w_k1_FDX_s006
                                 + 1.21241 w_k1_IBM_s006
                                 + 1.12006 w_k1_INTC_s006
                                 + 1.03851 w_k1_KO_s006 + 1.02721 w_k1_MAT_s006
                                 + 0.882468 w_k1_MSFT_s006
                                 + 0.940087 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.04018 w_k1_ADM_s006
                                 + 1.26711 w_k1_AXP_s006
                                 + 0.930015 w_k1_DIS_s006
                                 + 1.15515 w_k1_FDX_s006 + 0.9805 w_k1_IBM_s006
                                 + 1.2153 w_k1_INTC_s006
                                 + 0.848984 w_k1_KO_s006
                                 + 1.02536 w_k1_MAT_s006
                                 + 1.01998 w_k1_MSFT_s006
                                 + 1.03455 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.2827 w_k1_ADM_s006
                                 + 1.18467 w_k1_AXP_s006 + 1.4757 w_k1_DIS_s006
                                 + 1.25047 w_k1_FDX_s006
                                 + 1.38116 w_k1_IBM_s006
                                 + 1.18885 w_k1_INTC_s006
                                 + 1.24027 w_k1_KO_s006 + 1.37737 w_k1_MAT_s006
                                 + 1.08772 w_k1_MSFT_s006
                                 + 1.13618 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.909538 w_k1_ADM_s006
                                 + 0.893728 w_k1_AXP_s006
                                 + 0.706807 w_k1_DIS_s006 + 0.946 w_k1_FDX_s006
                                 + 0.730129 w_k1_IBM_s006
                                 + 0.943189 w_k1_INTC_s006
                                 + 0.67307 w_k1_KO_s006
                                 + 0.631886 w_k1_MAT_s006
                                 + 0.786229 w_k1_MSFT_s006
                                 + 0.865768 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.30038 w_k1_ADM_s007
                                 + 1.40127 w_k1_AXP_s007
                                 + 1.43232 w_k1_DIS_s007
                                 + 1.27084 w_k1_FDX_s007
                                 + 1.32845 w_k1_IBM_s007
                                 + 1.15029 w_k1_INTC_s007
                                 + 1.40474 w_k1_KO_s007 + 1.51244 w_k1_MAT_s007
                                 + 1.17693 w_k1_MSFT_s007 + 1.2693 w_k1_PG_s007
                                 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.989815 w_k1_ADM_s007
                                 + 0.825306 w_k1_AXP_s007
                                 + 0.732872 w_k1_DIS_s007
                                 + 0.8886 w_k1_FDX_s007
                                 + 0.703966 w_k1_IBM_s007
                                 + 0.896118 w_k1_INTC_s007
                                 + 0.733472 w_k1_KO_s007
                                 + 0.540785 w_k1_MAT_s007
                                 + 0.83608 w_k1_MSFT_s007
                                 + 0.860992 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.857231 w_k1_ADM_s007
                                 + 0.88522 w_k1_AXP_s007
                                 + 1.42414 w_k1_DIS_s007
                                 + 1.03246 w_k1_FDX_s007
                                 + 1.14067 w_k1_IBM_s007
                                 + 0.950208 w_k1_INTC_s007
                                 + 1.09586 w_k1_KO_s007 + 1.23083 w_k1_MAT_s007
                                 + 1.13947 w_k1_MSFT_s007
                                 + 1.02154 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.743872 w_k1_ADM_s007
                                 + 0.803834 w_k1_AXP_s007
                                 + 1.06468 w_k1_DIS_s007
                                 + 0.90722 w_k1_FDX_s007
                                 + 1.09073 w_k1_IBM_s007
                                 + 0.956001 w_k1_INTC_s007
                                 + 0.967811 w_k1_KO_s007
                                 + 0.858559 w_k1_MAT_s007
                                 + 0.97543 w_k1_MSFT_s007
                                 + 0.969563 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.15125 w_k1_ADM_s007
                                 + 0.968916 w_k1_AXP_s007
                                 + 1.11752 w_k1_DIS_s007
                                 + 1.01314 w_k1_FDX_s007
                                 + 0.988157 w_k1_IBM_s007
                                 + 0.893944 w_k1_INTC_s007
                                 + 0.897195 w_k1_KO_s007
                                 + 0.771072 w_k1_MAT_s007
                                 + 0.97945 w_k1_MSFT_s007
                                 + 1.06304 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.990246 w_k1_ADM_s007
                                 + 1.27331 w_k1_AXP_s007
                                 + 1.22781 w_k1_DIS_s007
                                 + 1.22099 w_k1_FDX_s007
                                 + 0.816611 w_k1_IBM_s007
                                 + 0.981278 w_k1_INTC_s007
                                 + 0.897359 w_k1_KO_s007
                                 + 1.11532 w_k1_MAT_s007
                                 + 1.0446 w_k1_MSFT_s007 + 1.04827 w_k1_PG_s007
                                 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.17881 w_k1_ADM_s007
                                 + 1.47732 w_k1_AXP_s007
                                 + 2.21358 w_k1_DIS_s007
                                 + 1.33401 w_k1_FDX_s007
                                 + 1.46122 w_k1_IBM_s007 + 1.208 w_k1_INTC_s007
                                 + 1.5964 w_k1_KO_s007 + 1.31583 w_k1_MAT_s007
                                 + 1.34954 w_k1_MSFT_s007
                                 + 1.33796 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.712281 w_k1_ADM_s007
                                 + 0.784411 w_k1_AXP_s007
                                 + 0.676985 w_k1_DIS_s007
                                 + 0.791912 w_k1_FDX_s007
                                 + 1.01196 w_k1_IBM_s007
                                 + 0.597766 w_k1_INTC_s007
                                 + 0.666097 w_k1_KO_s007
                                 + 1.05364 w_k1_MAT_s007
                                 + 0.691652 w_k1_MSFT_s007
                                 + 0.816085 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.14643 w_k1_ADM_s010
                                 + 1.10933 w_k1_AXP_s010
                                 + 1.19866 w_k1_DIS_s010
                                 + 1.09831 w_k1_FDX_s010
                                 + 0.819329 w_k1_IBM_s010
                                 + 1.19597 w_k1_INTC_s010
                                 + 0.983491 w_k1_KO_s010
                                 + 0.783382 w_k1_MAT_s010
                                 + 1.07439 w_k1_MSFT_s010
                                 + 0.929181 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.822052 w_k1_ADM_s010
                                 + 1.09691 w_k1_AXP_s010
                                 + 0.952012 w_k1_DIS_s010
                                 + 0.87764 w_k1_FDX_s010
                                 + 1.12128 w_k1_IBM_s010
                                 + 0.822968 w_k1_INTC_s010
                                 + 0.921913 w_k1_KO_s010
                                 + 1.14723 w_k1_MAT_s010
                                 + 0.839473 w_k1_MSFT_s010
                                 + 0.926195 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.787649 w_k1_ADM_s010
                                 + 0.741191 w_k1_AXP_s010
                                 + 0.598003 w_k1_DIS_s010
                                 + 0.801184 w_k1_FDX_s010
                                 + 0.771702 w_k1_IBM_s010
                                 + 0.687769 w_k1_INTC_s010
                                 + 0.621096 w_k1_KO_s010
                                 + 0.685126 w_k1_MAT_s010
                                 + 0.621261 w_k1_MSFT_s010
                                 + 0.727695 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.863541 w_k1_ADM_s010
                                 + 0.951073 w_k1_AXP_s010
                                 + 0.913152 w_k1_DIS_s010
                                 + 0.809877 w_k1_FDX_s010
                                 + 1.38659 w_k1_IBM_s010
                                 + 0.88003 w_k1_INTC_s010 + 1.0874 w_k1_KO_s010
                                 + 1.2217 w_k1_MAT_s010
                                 + 0.919497 w_k1_MSFT_s010
                                 + 0.894126 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.944905 w_k1_ADM_s010
                                 + 0.823348 w_k1_AXP_s010
                                 + 0.857376 w_k1_DIS_s010
                                 + 0.914007 w_k1_FDX_s010
                                 + 0.764282 w_k1_IBM_s010
                                 + 0.906808 w_k1_INTC_s010
                                 + 0.810742 w_k1_KO_s010
                                 + 0.458169 w_k1_MAT_s010
                                 + 0.877129 w_k1_MSFT_s010
                                 + 0.879196 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.06183 w_k1_ADM_s010
                                 + 1.73426 w_k1_AXP_s010
                                 + 1.72611 w_k1_DIS_s010
                                 + 1.22856 w_k1_FDX_s010
                                 + 1.67148 w_k1_IBM_s010
                                 + 1.09677 w_k1_INTC_s010
                                 + 1.55284 w_k1_KO_s010 + 1.66998 w_k1_MAT_s010
                                 + 1.27218 w_k1_MSFT_s010
                                 + 1.29484 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.34523 w_k1_ADM_s010
                                 + 1.00695 w_k1_AXP_s010 + 1.4051 w_k1_DIS_s010
                                 + 1.08357 w_k1_FDX_s010
                                 + 1.19591 w_k1_IBM_s010
                                 + 1.05201 w_k1_INTC_s010
                                 + 1.26439 w_k1_KO_s010
                                 + 0.845884 w_k1_MAT_s010
                                 + 1.06974 w_k1_MSFT_s010
                                 + 1.06636 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.941432 w_k1_ADM_s010
                                 + 1.44548 w_k1_AXP_s010
                                 + 1.20946 w_k1_DIS_s010
                                 + 1.16756 w_k1_FDX_s010
                                 + 1.13481 w_k1_IBM_s010
                                 + 0.995239 w_k1_INTC_s010
                                 + 1.1249 w_k1_KO_s010 + 1.47026 w_k1_MAT_s010
                                 + 1.09298 w_k1_MSFT_s010
                                 + 1.02279 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.819628 w_k1_ADM_s011
                                 + 0.514294 w_k1_AXP_s011
                                 + 0.721696 w_k1_DIS_s011
                                 + 0.722506 w_k1_FDX_s011
                                 + 0.846411 w_k1_IBM_s011
                                 + 0.704508 w_k1_INTC_s011
                                 + 0.83655 w_k1_KO_s011
                                 + 0.661442 w_k1_MAT_s011
                                 + 0.828325 w_k1_MSFT_s011
                                 + 0.791291 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.926852 w_k1_ADM_s011
                                 + 0.908436 w_k1_AXP_s011
                                 + 0.788366 w_k1_DIS_s011
                                 + 0.90729 w_k1_FDX_s011
                                 + 0.692884 w_k1_IBM_s011
                                 + 0.77236 w_k1_INTC_s011
                                 + 0.775899 w_k1_KO_s011
                                 + 0.637591 w_k1_MAT_s011
                                 + 0.783571 w_k1_MSFT_s011
                                 + 0.850372 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.08809 w_k1_ADM_s011
                                 + 1.58762 w_k1_AXP_s011
                                 + 1.46458 w_k1_DIS_s011
                                 + 1.20513 w_k1_FDX_s011
                                 + 1.50497 w_k1_IBM_s011
                                 + 1.36348 w_k1_INTC_s011
                                 + 1.37791 w_k1_KO_s011 + 1.65701 w_k1_MAT_s011
                                 + 1.26342 w_k1_MSFT_s011
                                 + 1.14145 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.945895 w_k1_ADM_s011
                                 + 0.871997 w_k1_AXP_s011
                                 + 1.0466 w_k1_DIS_s011
                                 + 0.942082 w_k1_FDX_s011
                                 + 1.31899 w_k1_IBM_s011
                                 + 0.985636 w_k1_INTC_s011
                                 + 1.40921 w_k1_KO_s011 + 1.28047 w_k1_MAT_s011
                                 + 1.24933 w_k1_MSFT_s011
                                 + 1.08422 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.906849 w_k1_ADM_s011
                                 + 0.887662 w_k1_AXP_s011
                                 + 1.06017 w_k1_DIS_s011
                                 + 1.02198 w_k1_FDX_s011
                                 + 0.796655 w_k1_IBM_s011
                                 + 0.986268 w_k1_INTC_s011
                                 + 0.945797 w_k1_KO_s011
                                 + 0.774457 w_k1_MAT_s011
                                 + 0.949814 w_k1_MSFT_s011
                                 + 0.951913 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.45497 w_k1_ADM_s011
                                 + 1.29264 w_k1_AXP_s011
                                 + 1.90812 w_k1_DIS_s011
                                 + 1.54728 w_k1_FDX_s011
                                 + 1.16958 w_k1_IBM_s011
                                 + 0.970784 w_k1_INTC_s011
                                 + 1.36786 w_k1_KO_s011 + 1.01889 w_k1_MAT_s011
                                 + 1.29478 w_k1_MSFT_s011
                                 + 1.10554 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.851266 w_k1_ADM_s011
                                 + 1.07659 w_k1_AXP_s011
                                 + 0.876676 w_k1_DIS_s011
                                 + 0.962915 w_k1_FDX_s011
                                 + 1.02988 w_k1_IBM_s011
                                 + 1.10363 w_k1_INTC_s011
                                 + 0.924185 w_k1_KO_s011 + 1.0353 w_k1_MAT_s011
                                 + 1.00816 w_k1_MSFT_s011
                                 + 0.91301 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.981362 w_k1_ADM_s011
                                 + 1.2203 w_k1_AXP_s011 + 1.21398 w_k1_DIS_s011
                                 + 1.28957 w_k1_FDX_s011
                                 + 0.844868 w_k1_IBM_s011
                                 + 1.10721 w_k1_INTC_s011
                                 + 0.975598 w_k1_KO_s011 + 0.8332 w_k1_MAT_s011
                                 + 1.01373 w_k1_MSFT_s011
                                 + 0.930799 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.961554 w_k1_ADM_s012
                                 + 1.08597 w_k1_AXP_s012
                                 + 1.15614 w_k1_DIS_s012
                                 + 1.17663 w_k1_FDX_s012
                                 + 0.832677 w_k1_IBM_s012
                                 + 1.22368 w_k1_INTC_s012
                                 + 0.855101 w_k1_KO_s012
                                 + 0.825357 w_k1_MAT_s012
                                 + 1.09697 w_k1_MSFT_s012
                                 + 0.922939 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.969698 w_k1_ADM_s012
                                 + 1.15378 w_k1_AXP_s012
                                 + 1.21668 w_k1_DIS_s012 + 1.1046 w_k1_FDX_s012
                                 + 1.53042 w_k1_IBM_s012
                                 + 1.30235 w_k1_INTC_s012 + 1.2553 w_k1_KO_s012
                                 + 1.27282 w_k1_MAT_s012
                                 + 1.13847 w_k1_MSFT_s012
                                 + 0.955333 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.840346 w_k1_ADM_s012
                                 + 1.42159 w_k1_AXP_s012
                                 + 0.807967 w_k1_DIS_s012
                                 + 0.942488 w_k1_FDX_s012
                                 + 1.47093 w_k1_IBM_s012
                                 + 1.09635 w_k1_INTC_s012
                                 + 1.21434 w_k1_KO_s012 + 1.20981 w_k1_MAT_s012
                                 + 0.916218 w_k1_MSFT_s012
                                 + 0.999113 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.984874 w_k1_ADM_s012
                                 + 0.957902 w_k1_AXP_s012
                                 + 0.881223 w_k1_DIS_s012
                                 + 1.12027 w_k1_FDX_s012
                                 + 0.54579 w_k1_IBM_s012
                                 + 0.96211 w_k1_INTC_s012
                                 + 0.735412 w_k1_KO_s012
                                 + 0.469571 w_k1_MAT_s012
                                 + 0.935264 w_k1_MSFT_s012
                                 + 0.858272 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.28087 w_k1_ADM_s012
                                 + 1.64748 w_k1_AXP_s012
                                 + 1.61661 w_k1_DIS_s012
                                 + 1.40619 w_k1_FDX_s012
                                 + 1.27706 w_k1_IBM_s012
                                 + 1.76087 w_k1_INTC_s012
                                 + 1.39787 w_k1_KO_s012 + 1.14344 w_k1_MAT_s012
                                 + 1.2547 w_k1_MSFT_s012 + 1.151 w_k1_PG_s012
                                 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.762573 w_k1_ADM_s012
                                 + 0.689254 w_k1_AXP_s012
                                 + 0.648114 w_k1_DIS_s012
                                 + 0.775657 w_k1_FDX_s012
                                 + 0.688971 w_k1_IBM_s012
                                 + 0.81883 w_k1_INTC_s012
                                 + 0.699863 w_k1_KO_s012
                                 + 0.553415 w_k1_MAT_s012
                                 + 0.787877 w_k1_MSFT_s012
                                 + 0.794894 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.926572 w_k1_ADM_s012
                                 + 0.916177 w_k1_AXP_s012
                                 + 0.876832 w_k1_DIS_s012
                                 + 0.923553 w_k1_FDX_s012
                                 + 1.01508 w_k1_IBM_s012
                                 + 0.979562 w_k1_INTC_s012
                                 + 0.887502 w_k1_KO_s012
                                 + 0.767722 w_k1_MAT_s012
                                 + 0.989531 w_k1_MSFT_s012
                                 + 0.851663 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.678105 w_k1_ADM_s012
                                 + 0.777783 w_k1_AXP_s012
                                 + 0.611547 w_k1_DIS_s012
                                 + 0.673318 w_k1_FDX_s012
                                 + 1.06983 w_k1_IBM_s012
                                 + 0.995252 w_k1_INTC_s012
                                 + 0.795638 w_k1_KO_s012
                                 + 0.895998 w_k1_MAT_s012
                                 + 0.789198 w_k1_MSFT_s012
                                 + 0.791126 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.46141 w_k1_ADM_s013
                                 + 0.892376 w_k1_AXP_s013
                                 + 0.99738 w_k1_DIS_s013
                                 + 1.02092 w_k1_FDX_s013
                                 + 1.51327 w_k1_IBM_s013
                                 + 0.847494 w_k1_INTC_s013
                                 + 1.28956 w_k1_KO_s013 + 1.04007 w_k1_MAT_s013
                                 + 0.950658 w_k1_MSFT_s013
                                 + 1.08108 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.920944 w_k1_ADM_s013
                                 + 0.724138 w_k1_AXP_s013
                                 + 0.655906 w_k1_DIS_s013
                                 + 0.83229 w_k1_FDX_s013
                                 + 0.795905 w_k1_IBM_s013
                                 + 0.806785 w_k1_INTC_s013
                                 + 0.786409 w_k1_KO_s013
                                 + 0.619441 w_k1_MAT_s013
                                 + 0.833041 w_k1_MSFT_s013
                                 + 0.829273 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.52849 w_k1_ADM_s013
                                 + 1.47412 w_k1_AXP_s013
                                 + 1.39109 w_k1_DIS_s013
                                 + 1.39506 w_k1_FDX_s013
                                 + 1.21397 w_k1_IBM_s013
                                 + 0.868222 w_k1_INTC_s013
                                 + 1.29262 w_k1_KO_s013 + 1.28687 w_k1_MAT_s013
                                 + 1.11827 w_k1_MSFT_s013
                                 + 1.38433 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.54633 w_k1_ADM_s013
                                 + 1.11833 w_k1_AXP_s013
                                 + 1.15015 w_k1_DIS_s013
                                 + 1.33191 w_k1_FDX_s013 + 0.8037 w_k1_IBM_s013
                                 + 0.942864 w_k1_INTC_s013
                                 + 0.860507 w_k1_KO_s013
                                 + 0.628152 w_k1_MAT_s013
                                 + 1.05561 w_k1_MSFT_s013
                                 + 1.11391 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.04568 w_k1_ADM_s013
                                 + 0.778317 w_k1_AXP_s013
                                 + 0.801019 w_k1_DIS_s013
                                 + 0.843243 w_k1_FDX_s013
                                 + 1.14585 w_k1_IBM_s013
                                 + 0.751942 w_k1_INTC_s013
                                 + 0.936207 w_k1_KO_s013
                                 + 1.00345 w_k1_MAT_s013
                                 + 0.761112 w_k1_MSFT_s013
                                 + 0.98967 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.23155 w_k1_ADM_s013
                                 + 0.882039 w_k1_AXP_s013
                                 + 0.925887 w_k1_DIS_s013
                                 + 1.07751 w_k1_FDX_s013
                                 + 0.703101 w_k1_IBM_s013
                                 + 0.897668 w_k1_INTC_s013
                                 + 0.738377 w_k1_KO_s013
                                 + 0.641733 w_k1_MAT_s013
                                 + 0.976647 w_k1_MSFT_s013
                                 + 0.896858 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.26229 w_k1_ADM_s013
                                 + 0.745228 w_k1_AXP_s013
                                 + 1.16992 w_k1_DIS_s013
                                 + 1.03021 w_k1_FDX_s013
                                 + 0.916396 w_k1_IBM_s013
                                 + 0.701243 w_k1_INTC_s013
                                 + 0.970343 w_k1_KO_s013
                                 + 0.776066 w_k1_MAT_s013
                                 + 0.914883 w_k1_MSFT_s013
                                 + 1.03748 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.31427 w_k1_ADM_s013
                                 + 1.30405 w_k1_AXP_s013
                                 + 1.06167 w_k1_DIS_s013 + 1.1801 w_k1_FDX_s013
                                 + 0.956539 w_k1_IBM_s013
                                 + 1.05106 w_k1_INTC_s013
                                 + 1.07007 w_k1_KO_s013
                                 + 0.995986 w_k1_MAT_s013
                                 + 0.951067 w_k1_MSFT_s013
                                 + 1.01699 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.874873 w_k1_ADM_s014
                                 + 0.852789 w_k1_AXP_s014
                                 + 0.812407 w_k1_DIS_s014
                                 + 0.908445 w_k1_FDX_s014
                                 + 0.818658 w_k1_IBM_s014
                                 + 0.788291 w_k1_INTC_s014
                                 + 0.819294 w_k1_KO_s014
                                 + 0.588598 w_k1_MAT_s014
                                 + 0.799426 w_k1_MSFT_s014
                                 + 0.888623 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.705388 w_k1_ADM_s014
                                 + 0.81613 w_k1_AXP_s014
                                 + 0.683148 w_k1_DIS_s014
                                 + 0.73143 w_k1_FDX_s014
                                 + 1.15634 w_k1_IBM_s014
                                 + 0.716446 w_k1_INTC_s014
                                 + 0.915581 w_k1_KO_s014
                                 + 1.09878 w_k1_MAT_s014
                                 + 0.783204 w_k1_MSFT_s014
                                 + 0.874032 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.722141 w_k1_ADM_s014
                                 + 0.646275 w_k1_AXP_s014
                                 + 0.837646 w_k1_DIS_s014
                                 + 0.819706 w_k1_FDX_s014
                                 + 0.886233 w_k1_IBM_s014
                                 + 0.715145 w_k1_INTC_s014
                                 + 0.850727 w_k1_KO_s014
                                 + 0.83622 w_k1_MAT_s014
                                 + 0.777507 w_k1_MSFT_s014
                                 + 0.868424 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.986948 w_k1_ADM_s014
                                 + 1.33071 w_k1_AXP_s014
                                 + 1.23874 w_k1_DIS_s014
                                 + 1.22528 w_k1_FDX_s014
                                 + 1.09329 w_k1_IBM_s014
                                 + 1.09699 w_k1_INTC_s014
                                 + 1.08873 w_k1_KO_s014 + 1.36646 w_k1_MAT_s014
                                 + 1.16023 w_k1_MSFT_s014
                                 + 1.05077 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.983432 w_k1_ADM_s014
                                 + 0.932219 w_k1_AXP_s014
                                 + 1.29268 w_k1_DIS_s014 + 1.1124 w_k1_FDX_s014
                                 + 0.934892 w_k1_IBM_s014
                                 + 1.11369 w_k1_INTC_s014
                                 + 1.11784 w_k1_KO_s014 + 1.00707 w_k1_MAT_s014
                                 + 1.25843 w_k1_MSFT_s014 + 1.1645 w_k1_PG_s014
                                 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.659453 w_k1_ADM_s014
                                 + 0.682917 w_k1_AXP_s014
                                 + 0.555505 w_k1_DIS_s014
                                 + 0.712485 w_k1_FDX_s014
                                 + 0.790262 w_k1_IBM_s014
                                 + 0.639856 w_k1_INTC_s014
                                 + 0.608242 w_k1_KO_s014
                                 + 0.749361 w_k1_MAT_s014
                                 + 0.65082 w_k1_MSFT_s014
                                 + 0.818947 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.949026 w_k1_ADM_s014
                                 + 1.30128 w_k1_AXP_s014
                                 + 1.16985 w_k1_DIS_s014
                                 + 1.06897 w_k1_FDX_s014
                                 + 1.63019 w_k1_IBM_s014
                                 + 1.06289 w_k1_INTC_s014
                                 + 1.37228 w_k1_KO_s014 + 1.92991 w_k1_MAT_s014
                                 + 1.07793 w_k1_MSFT_s014
                                 + 1.24383 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.748273 w_k1_ADM_s014
                                 + 1.07533 w_k1_AXP_s014
                                 + 0.911273 w_k1_DIS_s014
                                 + 0.955299 w_k1_FDX_s014
                                 + 1.13071 w_k1_IBM_s014
                                 + 0.923272 w_k1_INTC_s014
                                 + 0.993246 w_k1_KO_s014
                                 + 1.25395 w_k1_MAT_s014
                                 + 0.873814 w_k1_MSFT_s014
                                 + 1.06729 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.940697 w_k1_ADM_s015
                                 + 0.90745 w_k1_AXP_s015 + 1.0444 w_k1_DIS_s015
                                 + 0.901167 w_k1_FDX_s015
                                 + 1.20181 w_k1_IBM_s015
                                 + 0.935956 w_k1_INTC_s015
                                 + 1.13082 w_k1_KO_s015 + 1.23635 w_k1_MAT_s015
                                 + 0.971931 w_k1_MSFT_s015
                                 + 0.980706 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.855073 w_k1_ADM_s015
                                 + 0.874391 w_k1_AXP_s015
                                 + 1.04084 w_k1_DIS_s015
                                 + 0.93258 w_k1_FDX_s015
                                 + 0.634148 w_k1_IBM_s015
                                 + 1.01359 w_k1_INTC_s015
                                 + 0.865563 w_k1_KO_s015
                                 + 0.659941 w_k1_MAT_s015
                                 + 1.11807 w_k1_MSFT_s015
                                 + 0.935332 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.926186 w_k1_ADM_s015
                                 + 1.31511 w_k1_AXP_s015
                                 + 1.19156 w_k1_DIS_s015
                                 + 1.17182 w_k1_FDX_s015
                                 + 1.15071 w_k1_IBM_s015
                                 + 1.02627 w_k1_INTC_s015
                                 + 1.08579 w_k1_KO_s015 + 1.37187 w_k1_MAT_s015
                                 + 1.01675 w_k1_MSFT_s015
                                 + 1.07044 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.23605 w_k1_ADM_s015
                                 + 1.41534 w_k1_AXP_s015
                                 + 1.22243 w_k1_DIS_s015
                                 + 1.10869 w_k1_FDX_s015
                                 + 1.62672 w_k1_IBM_s015
                                 + 1.36749 w_k1_INTC_s015
                                 + 1.11843 w_k1_KO_s015 + 1.45768 w_k1_MAT_s015
                                 + 1.31565 w_k1_MSFT_s015
                                 + 1.16142 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.17633 w_k1_ADM_s015
                                 + 1.18952 w_k1_AXP_s015
                                 + 1.65101 w_k1_DIS_s015
                                 + 1.28919 w_k1_FDX_s015
                                 + 1.29347 w_k1_IBM_s015
                                 + 1.4482 w_k1_INTC_s015 + 1.414 w_k1_KO_s015
                                 + 1.00051 w_k1_MAT_s015
                                 + 1.4625 w_k1_MSFT_s015 + 1.20967 w_k1_PG_s015
                                 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.935017 w_k1_ADM_s015
                                 + 0.85626 w_k1_AXP_s015
                                 + 0.911934 w_k1_DIS_s015
                                 + 0.905155 w_k1_FDX_s015
                                 + 0.928912 w_k1_IBM_s015
                                 + 1.01832 w_k1_INTC_s015
                                 + 0.896445 w_k1_KO_s015
                                 + 0.864322 w_k1_MAT_s015
                                 + 0.998592 w_k1_MSFT_s015
                                 + 0.948239 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 1.18729 w_k1_ADM_s015
                                 + 1.03355 w_k1_AXP_s015
                                 + 1.43204 w_k1_DIS_s015
                                 + 1.12825 w_k1_FDX_s015
                                 + 0.916326 w_k1_IBM_s015
                                 + 1.30513 w_k1_INTC_s015
                                 + 1.18571 w_k1_KO_s015 + 1.01041 w_k1_MAT_s015
                                 + 1.34475 w_k1_MSFT_s015 + 1.0072 w_k1_PG_s015
                                 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.830832 w_k1_ADM_s015
                                 + 0.699498 w_k1_AXP_s015
                                 + 0.75013 w_k1_DIS_s015
                                 + 0.844063 w_k1_FDX_s015
                                 + 0.724482 w_k1_IBM_s015
                                 + 0.799042 w_k1_INTC_s015
                                 + 0.856618 w_k1_KO_s015
                                 + 0.73567 w_k1_MAT_s015
                                 + 0.878995 w_k1_MSFT_s015
                                 + 0.890066 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.07483 w_k1_ADM_s016
                                 + 0.859836 w_k1_AXP_s016
                                 + 0.770977 w_k1_DIS_s016
                                 + 0.876241 w_k1_FDX_s016
                                 + 0.934475 w_k1_IBM_s016
                                 + 0.816222 w_k1_INTC_s016
                                 + 0.902941 w_k1_KO_s016
                                 + 0.990743 w_k1_MAT_s016
                                 + 0.896704 w_k1_MSFT_s016
                                 + 0.90409 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.47139 w_k1_ADM_s016
                                 + 1.34185 w_k1_AXP_s016
                                 + 1.25481 w_k1_DIS_s016
                                 + 1.17742 w_k1_FDX_s016
                                 + 1.53455 w_k1_IBM_s016
                                 + 0.986162 w_k1_INTC_s016
                                 + 1.43844 w_k1_KO_s016 + 1.76805 w_k1_MAT_s016
                                 + 1.07544 w_k1_MSFT_s016
                                 + 1.12984 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.12298 w_k1_ADM_s016
                                 + 0.817398 w_k1_AXP_s016
                                 + 0.737582 w_k1_DIS_s016
                                 + 0.914088 w_k1_FDX_s016
                                 + 0.683354 w_k1_IBM_s016
                                 + 0.741156 w_k1_INTC_s016
                                 + 0.842377 w_k1_KO_s016
                                 + 0.583638 w_k1_MAT_s016
                                 + 0.850642 w_k1_MSFT_s016
                                 + 0.854702 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.900335 w_k1_ADM_s016
                                 + 0.918076 w_k1_AXP_s016
                                 + 0.904505 w_k1_DIS_s016
                                 + 0.813626 w_k1_FDX_s016
                                 + 1.2033 w_k1_IBM_s016
                                 + 0.883574 w_k1_INTC_s016
                                 + 1.08394 w_k1_KO_s016 + 1.19873 w_k1_MAT_s016
                                 + 0.901002 w_k1_MSFT_s016
                                 + 0.988583 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.36107 w_k1_ADM_s016
                                 + 1.04687 w_k1_AXP_s016
                                 + 1.16683 w_k1_DIS_s016
                                 + 1.10442 w_k1_FDX_s016
                                 + 1.05351 w_k1_IBM_s016
                                 + 0.894392 w_k1_INTC_s016
                                 + 1.19507 w_k1_KO_s016 + 1.2639 w_k1_MAT_s016
                                 + 1.12394 w_k1_MSFT_s016
                                 + 1.12219 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.08016 w_k1_ADM_s016
                                 + 1.03236 w_k1_AXP_s016
                                 + 1.21994 w_k1_DIS_s016
                                 + 1.09107 w_k1_FDX_s016
                                 + 0.846926 w_k1_IBM_s016
                                 + 0.910235 w_k1_INTC_s016
                                 + 0.959333 w_k1_KO_s016
                                 + 0.727751 w_k1_MAT_s016
                                 + 1.01804 w_k1_MSFT_s016
                                 + 0.952675 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.22442 w_k1_ADM_s016
                                 + 1.33818 w_k1_AXP_s016
                                 + 1.74406 w_k1_DIS_s016
                                 + 1.40859 w_k1_FDX_s016
                                 + 1.00936 w_k1_IBM_s016
                                 + 1.26711 w_k1_INTC_s016
                                 + 1.20835 w_k1_KO_s016 + 1.08874 w_k1_MAT_s016
                                 + 1.26798 w_k1_MSFT_s016
                                 + 1.07548 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.06295 w_k1_ADM_s016
                                 + 1.19791 w_k1_AXP_s016
                                 + 0.984103 w_k1_DIS_s016
                                 + 1.08969 w_k1_FDX_s016
                                 + 0.918259 w_k1_IBM_s016
                                 + 1.15288 w_k1_INTC_s016
                                 + 0.962249 w_k1_KO_s016
                                 + 1.06665 w_k1_MAT_s016
                                 + 0.953756 w_k1_MSFT_s016
                                 + 1.04616 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.991269 w_k1_ADM_s017
                                 + 1.41676 w_k1_AXP_s017
                                 + 1.44003 w_k1_DIS_s017
                                 + 1.36979 w_k1_FDX_s017
                                 + 0.896133 w_k1_IBM_s017
                                 + 1.4889 w_k1_INTC_s017 + 1.29589 w_k1_KO_s017
                                 + 1.22501 w_k1_MAT_s017
                                 + 1.54706 w_k1_MSFT_s017
                                 + 1.11055 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.847765 w_k1_ADM_s017
                                 + 0.775851 w_k1_AXP_s017
                                 + 1.21091 w_k1_DIS_s017
                                 + 1.08679 w_k1_FDX_s017
                                 + 1.11006 w_k1_IBM_s017
                                 + 1.17752 w_k1_INTC_s017 + 1.1404 w_k1_KO_s017
                                 + 0.882665 w_k1_MAT_s017
                                 + 1.23824 w_k1_MSFT_s017
                                 + 0.994231 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.872529 w_k1_ADM_s017
                                 + 0.992235 w_k1_AXP_s017
                                 + 1.0409 w_k1_DIS_s017
                                 + 0.886525 w_k1_FDX_s017
                                 + 1.6334 w_k1_IBM_s017
                                 + 0.969942 w_k1_INTC_s017
                                 + 1.1027 w_k1_KO_s017 + 1.26022 w_k1_MAT_s017
                                 + 0.823325 w_k1_MSFT_s017
                                 + 0.89166 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.949163 w_k1_ADM_s017
                                 + 0.621607 w_k1_AXP_s017
                                 + 0.82447 w_k1_DIS_s017
                                 + 0.82683 w_k1_FDX_s017
                                 + 0.731505 w_k1_IBM_s017
                                 + 0.968347 w_k1_INTC_s017
                                 + 0.84436 w_k1_KO_s017
                                 + 0.446038 w_k1_MAT_s017
                                 + 0.83884 w_k1_MSFT_s017
                                 + 0.861536 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.951172 w_k1_ADM_s017
                                 + 0.856656 w_k1_AXP_s017
                                 + 0.756567 w_k1_DIS_s017
                                 + 0.794666 w_k1_FDX_s017
                                 + 0.939115 w_k1_IBM_s017
                                 + 1.06352 w_k1_INTC_s017 + 1.0185 w_k1_KO_s017
                                 + 0.869638 w_k1_MAT_s017
                                 + 0.903645 w_k1_MSFT_s017
                                 + 0.852109 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.52493 w_k1_ADM_s017
                                 + 1.28737 w_k1_AXP_s017
                                 + 1.82159 w_k1_DIS_s017
                                 + 1.37128 w_k1_FDX_s017
                                 + 1.30609 w_k1_IBM_s017
                                 + 1.6462 w_k1_INTC_s017 + 1.72406 w_k1_KO_s017
                                 + 1.29963 w_k1_MAT_s017
                                 + 1.54391 w_k1_MSFT_s017
                                 + 1.13949 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.131 w_k1_ADM_s017
                                 + 1.08565 w_k1_AXP_s017
                                 + 1.04195 w_k1_DIS_s017
                                 + 1.15744 w_k1_FDX_s017
                                 + 0.682016 w_k1_IBM_s017
                                 + 1.30898 w_k1_INTC_s017
                                 + 0.943032 w_k1_KO_s017
                                 + 0.657704 w_k1_MAT_s017
                                 + 1.12144 w_k1_MSFT_s017
                                 + 0.931733 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.937366 w_k1_ADM_s017
                                 + 0.778089 w_k1_AXP_s017
                                 + 0.77077 w_k1_DIS_s017
                                 + 0.908244 w_k1_FDX_s017
                                 + 0.648074 w_k1_IBM_s017
                                 + 0.941537 w_k1_INTC_s017
                                 + 0.758309 w_k1_KO_s017
                                 + 0.737301 w_k1_MAT_s017
                                 + 0.960763 w_k1_MSFT_s017
                                 + 0.813799 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.96654 w_k1_ADM_s100
                                 + 1.11885 w_k1_AXP_s100
                                 + 0.858212 w_k1_DIS_s100
                                 + 1.0231 w_k1_FDX_s100 + 0.90709 w_k1_IBM_s100
                                 + 0.8953 w_k1_INTC_s100
                                 + 0.794419 w_k1_KO_s100
                                 + 0.857839 w_k1_MAT_s100
                                 + 0.909965 w_k1_MSFT_s100
                                 + 1.02864 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.15199 w_k1_ADM_s100
                                 + 1.61342 w_k1_AXP_s100
                                 + 1.47094 w_k1_DIS_s100
                                 + 1.36426 w_k1_FDX_s100
                                 + 1.19856 w_k1_IBM_s100
                                 + 1.0841 w_k1_INTC_s100 + 1.19068 w_k1_KO_s100
                                 + 1.6038 w_k1_MAT_s100
                                 + 1.16275 w_k1_MSFT_s100
                                 + 1.17089 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.03372 w_k1_ADM_s100
                                 + 1.08213 w_k1_AXP_s100
                                 + 0.919528 w_k1_DIS_s100
                                 + 0.984108 w_k1_FDX_s100
                                 + 0.762288 w_k1_IBM_s100
                                 + 0.879509 w_k1_INTC_s100
                                 + 0.771702 w_k1_KO_s100
                                 + 0.696785 w_k1_MAT_s100
                                 + 0.829682 w_k1_MSFT_s100
                                 + 0.979347 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.956708 w_k1_ADM_s100
                                 + 0.91429 w_k1_AXP_s100
                                 + 1.04737 w_k1_DIS_s100
                                 + 0.976427 w_k1_FDX_s100
                                 + 0.906334 w_k1_IBM_s100
                                 + 0.992236 w_k1_INTC_s100
                                 + 0.945865 w_k1_KO_s100
                                 + 0.725518 w_k1_MAT_s100
                                 + 1.10501 w_k1_MSFT_s100
                                 + 0.955708 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.06983 w_k1_ADM_s100
                                 + 1.28931 w_k1_AXP_s100
                                 + 1.32927 w_k1_DIS_s100
                                 + 1.06168 w_k1_FDX_s100
                                 + 1.56651 w_k1_IBM_s100
                                 + 0.923509 w_k1_INTC_s100
                                 + 1.13795 w_k1_KO_s100 + 1.33806 w_k1_MAT_s100
                                 + 0.986025 w_k1_MSFT_s100
                                 + 1.09442 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.824829 w_k1_ADM_s100
                                 + 0.866699 w_k1_AXP_s100
                                 + 0.850657 w_k1_DIS_s100
                                 + 0.839822 w_k1_FDX_s100
                                 + 1.14691 w_k1_IBM_s100
                                 + 0.807724 w_k1_INTC_s100
                                 + 0.864102 w_k1_KO_s100
                                 + 1.00341 w_k1_MAT_s100
                                 + 0.900133 w_k1_MSFT_s100
                                 + 0.977992 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.27082 w_k1_ADM_s100
                                 + 1.2208 w_k1_AXP_s100 + 1.38525 w_k1_DIS_s100
                                 + 1.23049 w_k1_FDX_s100
                                 + 0.939495 w_k1_IBM_s100
                                 + 1.06967 w_k1_INTC_s100
                                 + 0.916575 w_k1_KO_s100
                                 + 0.885267 w_k1_MAT_s100
                                 + 1.06143 w_k1_MSFT_s100
                                 + 1.07042 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.798626 w_k1_ADM_s100
                                 + 0.873371 w_k1_AXP_s100
                                 + 0.609706 w_k1_DIS_s100
                                 + 0.734004 w_k1_FDX_s100
                                 + 0.850354 w_k1_IBM_s100
                                 + 0.747756 w_k1_INTC_s100
                                 + 0.7868 w_k1_KO_s100 + 0.789619 w_k1_MAT_s100
                                 + 0.729375 w_k1_MSFT_s100
                                 + 0.920606 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.718527 w_k1_ADM_s101
                                 + 0.959042 w_k1_AXP_s101
                                 + 0.927002 w_k1_DIS_s101
                                 + 0.917312 w_k1_FDX_s101
                                 + 1.21855 w_k1_IBM_s101
                                 + 1.01152 w_k1_INTC_s101
                                 + 0.990748 w_k1_KO_s101
                                 + 0.955379 w_k1_MAT_s101
                                 + 0.90708 w_k1_MSFT_s101
                                 + 0.999388 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.657943 w_k1_ADM_s101
                                 + 0.778026 w_k1_AXP_s101
                                 + 0.59889 w_k1_DIS_s101
                                 + 0.757358 w_k1_FDX_s101
                                 + 0.672272 w_k1_IBM_s101
                                 + 0.740324 w_k1_INTC_s101
                                 + 0.624012 w_k1_KO_s101
                                 + 0.940222 w_k1_MAT_s101
                                 + 0.820447 w_k1_MSFT_s101
                                 + 0.843986 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.638156 w_k1_ADM_s101
                                 + 0.639262 w_k1_AXP_s101
                                 + 0.473553 w_k1_DIS_s101
                                 + 0.635809 w_k1_FDX_s101
                                 + 0.928061 w_k1_IBM_s101
                                 + 0.671302 w_k1_INTC_s101
                                 + 0.643743 w_k1_KO_s101
                                 + 0.717901 w_k1_MAT_s101
                                 + 0.654942 w_k1_MSFT_s101
                                 + 0.753372 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.899436 w_k1_ADM_s101
                                 + 0.765231 w_k1_AXP_s101
                                 + 0.794829 w_k1_DIS_s101
                                 + 0.791571 w_k1_FDX_s101
                                 + 0.966316 w_k1_IBM_s101
                                 + 0.739598 w_k1_INTC_s101
                                 + 0.881856 w_k1_KO_s101
                                 + 0.97631 w_k1_MAT_s101
                                 + 0.890105 w_k1_MSFT_s101
                                 + 0.940678 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.00857 w_k1_ADM_s101
                                 + 1.34583 w_k1_AXP_s101 + 1.3414 w_k1_DIS_s101
                                 + 1.35825 w_k1_FDX_s101
                                 + 0.802586 w_k1_IBM_s101
                                 + 1.43659 w_k1_INTC_s101
                                 + 0.937837 w_k1_KO_s101
                                 + 0.548475 w_k1_MAT_s101
                                 + 1.16704 w_k1_MSFT_s101
                                 + 1.15556 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.19871 w_k1_ADM_s101
                                 + 1.33843 w_k1_AXP_s101
                                 + 1.56108 w_k1_DIS_s101
                                 + 1.32523 w_k1_FDX_s101
                                 + 1.36401 w_k1_IBM_s101
                                 + 1.14711 w_k1_INTC_s101
                                 + 1.17523 w_k1_KO_s101 + 1.22983 w_k1_MAT_s101
                                 + 1.24329 w_k1_MSFT_s101
                                 + 1.18864 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.12423 w_k1_ADM_s101
                                 + 1.15722 w_k1_AXP_s101
                                 + 0.911203 w_k1_DIS_s101
                                 + 1.08204 w_k1_FDX_s101
                                 + 1.00523 w_k1_IBM_s101
                                 + 1.03625 w_k1_INTC_s101
                                 + 0.943324 w_k1_KO_s101
                                 + 0.929645 w_k1_MAT_s101
                                 + 1.00656 w_k1_MSFT_s101
                                 + 1.00135 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.877352 w_k1_ADM_s101
                                 + 0.782066 w_k1_AXP_s101
                                 + 0.754921 w_k1_DIS_s101
                                 + 0.961491 w_k1_FDX_s101
                                 + 0.611477 w_k1_IBM_s101
                                 + 0.912237 w_k1_INTC_s101
                                 + 0.693602 w_k1_KO_s101
                                 + 0.450352 w_k1_MAT_s101
                                 + 0.851602 w_k1_MSFT_s101
                                 + 0.889452 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.31475 w_k1_ADM_s102
                                 + 1.16985 w_k1_AXP_s102 + 1.6083 w_k1_DIS_s102
                                 + 1.23681 w_k1_FDX_s102
                                 + 1.11635 w_k1_IBM_s102
                                 + 1.21867 w_k1_INTC_s102 + 1.2309 w_k1_KO_s102
                                 + 0.806892 w_k1_MAT_s102
                                 + 1.31933 w_k1_MSFT_s102
                                 + 1.05261 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.27157 w_k1_ADM_s102
                                 + 1.13011 w_k1_AXP_s102
                                 + 1.24497 w_k1_DIS_s102
                                 + 1.08415 w_k1_FDX_s102
                                 + 1.27481 w_k1_IBM_s102
                                 + 1.0921 w_k1_INTC_s102 + 1.22504 w_k1_KO_s102
                                 + 1.17801 w_k1_MAT_s102
                                 + 1.20667 w_k1_MSFT_s102
                                 + 1.07028 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.03458 w_k1_ADM_s102
                                 + 0.747405 w_k1_AXP_s102
                                 + 0.765686 w_k1_DIS_s102
                                 + 0.793431 w_k1_FDX_s102
                                 + 0.943563 w_k1_IBM_s102
                                 + 0.98876 w_k1_INTC_s102
                                 + 0.854964 w_k1_KO_s102
                                 + 0.942568 w_k1_MAT_s102
                                 + 0.90256 w_k1_MSFT_s102
                                 + 0.813108 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.18463 w_k1_ADM_s102
                                 + 0.892299 w_k1_AXP_s102
                                 + 1.08712 w_k1_DIS_s102
                                 + 0.998258 w_k1_FDX_s102
                                 + 0.953619 w_k1_IBM_s102
                                 + 1.05202 w_k1_INTC_s102
                                 + 1.20728 w_k1_KO_s102
                                 + 0.892381 w_k1_MAT_s102
                                 + 1.0602 w_k1_MSFT_s102 + 1.04918 w_k1_PG_s102
                                 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.912827 w_k1_ADM_s102
                                 + 0.690618 w_k1_AXP_s102
                                 + 0.755329 w_k1_DIS_s102
                                 + 0.836793 w_k1_FDX_s102
                                 + 0.660316 w_k1_IBM_s102
                                 + 0.856953 w_k1_INTC_s102
                                 + 0.666072 w_k1_KO_s102
                                 + 0.54154 w_k1_MAT_s102
                                 + 0.81518 w_k1_MSFT_s102
                                 + 0.771601 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.05312 w_k1_ADM_s102
                                 + 1.12492 w_k1_AXP_s102
                                 + 1.04463 w_k1_DIS_s102
                                 + 1.06042 w_k1_FDX_s102
                                 + 0.777247 w_k1_IBM_s102
                                 + 1.03027 w_k1_INTC_s102
                                 + 0.875096 w_k1_KO_s102
                                 + 0.648269 w_k1_MAT_s102
                                 + 0.907261 w_k1_MSFT_s102
                                 + 0.870963 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.0144 w_k1_ADM_s102
                                 + 1.11991 w_k1_AXP_s102
                                 + 0.864892 w_k1_DIS_s102
                                 + 0.885728 w_k1_FDX_s102
                                 + 1.02361 w_k1_IBM_s102
                                 + 0.901336 w_k1_INTC_s102
                                 + 0.887152 w_k1_KO_s102
                                 + 1.34589 w_k1_MAT_s102
                                 + 0.826138 w_k1_MSFT_s102
                                 + 0.921089 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.02769 w_k1_ADM_s102
                                 + 0.968796 w_k1_AXP_s102
                                 + 0.965461 w_k1_DIS_s102
                                 + 0.849258 w_k1_FDX_s102
                                 + 1.37049 w_k1_IBM_s102
                                 + 0.913076 w_k1_INTC_s102
                                 + 1.16772 w_k1_KO_s102 + 1.50439 w_k1_MAT_s102
                                 + 0.876268 w_k1_MSFT_s102
                                 + 0.967904 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.805705 w_k1_ADM_s103
                                 + 0.82057 w_k1_AXP_s103
                                 + 0.583046 w_k1_DIS_s103
                                 + 0.731185 w_k1_FDX_s103
                                 + 0.783106 w_k1_IBM_s103
                                 + 0.589864 w_k1_INTC_s103
                                 + 0.785752 w_k1_KO_s103
                                 + 1.01116 w_k1_MAT_s103
                                 + 0.726329 w_k1_MSFT_s103
                                 + 0.804735 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.970864 w_k1_ADM_s103
                                 + 1.19682 w_k1_AXP_s103
                                 + 1.22405 w_k1_DIS_s103
                                 + 1.09665 w_k1_FDX_s103
                                 + 1.29182 w_k1_IBM_s103
                                 + 1.00136 w_k1_INTC_s103
                                 + 1.16629 w_k1_KO_s103 + 1.64369 w_k1_MAT_s103
                                 + 1.04097 w_k1_MSFT_s103
                                 + 1.07544 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.938442 w_k1_ADM_s103
                                 + 1.00473 w_k1_AXP_s103
                                 + 1.11732 w_k1_DIS_s103
                                 + 1.21129 w_k1_FDX_s103
                                 + 0.86046 w_k1_IBM_s103
                                 + 1.03809 w_k1_INTC_s103 + 1.0212 w_k1_KO_s103
                                 + 0.965175 w_k1_MAT_s103
                                 + 1.15643 w_k1_MSFT_s103
                                 + 1.02517 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.08147 w_k1_ADM_s103
                                 + 1.259 w_k1_AXP_s103 + 1.47772 w_k1_DIS_s103
                                 + 1.12621 w_k1_FDX_s103
                                 + 1.68567 w_k1_IBM_s103
                                 + 0.986239 w_k1_INTC_s103
                                 + 1.48606 w_k1_KO_s103 + 1.94357 w_k1_MAT_s103
                                 + 1.12897 w_k1_MSFT_s103
                                 + 1.15763 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.917979 w_k1_ADM_s103
                                 + 1.07051 w_k1_AXP_s103
                                 + 0.775419 w_k1_DIS_s103
                                 + 1.00089 w_k1_FDX_s103
                                 + 0.726459 w_k1_IBM_s103
                                 + 0.874455 w_k1_INTC_s103
                                 + 0.8458 w_k1_KO_s103 + 0.986239 w_k1_MAT_s103
                                 + 0.950715 w_k1_MSFT_s103
                                 + 0.951519 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.736352 w_k1_ADM_s103
                                 + 0.95919 w_k1_AXP_s103
                                 + 0.789614 w_k1_DIS_s103
                                 + 0.951041 w_k1_FDX_s103
                                 + 0.912455 w_k1_IBM_s103
                                 + 0.778558 w_k1_INTC_s103
                                 + 1.04352 w_k1_KO_s103 + 1.4793 w_k1_MAT_s103
                                 + 0.953216 w_k1_MSFT_s103
                                 + 1.01198 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.898896 w_k1_ADM_s103
                                 + 1.01601 w_k1_AXP_s103
                                 + 0.864087 w_k1_DIS_s103
                                 + 0.89559 w_k1_FDX_s103
                                 + 1.12379 w_k1_IBM_s103
                                 + 0.766073 w_k1_INTC_s103
                                 + 1.00914 w_k1_KO_s103 + 1.28037 w_k1_MAT_s103
                                 + 0.840106 w_k1_MSFT_s103
                                 + 0.952142 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.3048 w_k1_ADM_s103
                                 + 1.73551 w_k1_AXP_s103
                                 + 1.63283 w_k1_DIS_s103
                                 + 1.44981 w_k1_FDX_s103
                                 + 1.27464 w_k1_IBM_s103
                                 + 1.2971 w_k1_INTC_s103 + 1.37599 w_k1_KO_s103
                                 + 1.93137 w_k1_MAT_s103
                                 + 1.42707 w_k1_MSFT_s103
                                 + 1.32708 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.987518 w_k1_ADM_s104
                                 + 0.791485 w_k1_AXP_s104
                                 + 0.998201 w_k1_DIS_s104
                                 + 0.953426 w_k1_FDX_s104
                                 + 1.12021 w_k1_IBM_s104
                                 + 1.21711 w_k1_INTC_s104
                                 + 1.10622 w_k1_KO_s104 + 1.05113 w_k1_MAT_s104
                                 + 1.1463 w_k1_MSFT_s104 + 1.04645 w_k1_PG_s104
                                 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.85971 w_k1_ADM_s104
                                 + 1.23116 w_k1_AXP_s104
                                 + 1.08928 w_k1_DIS_s104
                                 + 1.00377 w_k1_FDX_s104
                                 + 1.51625 w_k1_IBM_s104
                                 + 1.12913 w_k1_INTC_s104
                                 + 1.08816 w_k1_KO_s104 + 1.44835 w_k1_MAT_s104
                                 + 1.0983 w_k1_MSFT_s104
                                 + 0.977653 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.890345 w_k1_ADM_s104
                                 + 0.781516 w_k1_AXP_s104
                                 + 0.638527 w_k1_DIS_s104
                                 + 0.756644 w_k1_FDX_s104
                                 + 1.00133 w_k1_IBM_s104
                                 + 0.693024 w_k1_INTC_s104
                                 + 0.836787 w_k1_KO_s104
                                 + 0.949148 w_k1_MAT_s104
                                 + 0.767655 w_k1_MSFT_s104
                                 + 0.944144 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.42202 w_k1_ADM_s104
                                 + 1.30282 w_k1_AXP_s104
                                 + 1.48329 w_k1_DIS_s104
                                 + 1.30614 w_k1_FDX_s104
                                 + 1.00975 w_k1_IBM_s104
                                 + 1.12549 w_k1_INTC_s104
                                 + 1.28846 w_k1_KO_s104
                                 + 0.811531 w_k1_MAT_s104
                                 + 1.32546 w_k1_MSFT_s104
                                 + 1.12712 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.842052 w_k1_ADM_s104
                                 + 0.79765 w_k1_AXP_s104
                                 + 0.818597 w_k1_DIS_s104
                                 + 0.911324 w_k1_FDX_s104
                                 + 0.745125 w_k1_IBM_s104
                                 + 0.984472 w_k1_INTC_s104
                                 + 0.866229 w_k1_KO_s104
                                 + 0.705109 w_k1_MAT_s104
                                 + 1.06294 w_k1_MSFT_s104
                                 + 0.906872 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.51267 w_k1_ADM_s104
                                 + 1.64259 w_k1_AXP_s104
                                 + 1.76379 w_k1_DIS_s104
                                 + 1.35374 w_k1_FDX_s104
                                 + 1.47485 w_k1_IBM_s104
                                 + 1.27617 w_k1_INTC_s104
                                 + 1.38494 w_k1_KO_s104 + 1.74356 w_k1_MAT_s104
                                 + 1.2583 w_k1_MSFT_s104 + 1.2328 w_k1_PG_s104
                                 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.06441 w_k1_ADM_s104
                                 + 0.947548 w_k1_AXP_s104
                                 + 1.08925 w_k1_DIS_s104
                                 + 1.18577 w_k1_FDX_s104
                                 + 0.762614 w_k1_IBM_s104
                                 + 1.14871 w_k1_INTC_s104
                                 + 0.997299 w_k1_KO_s104
                                 + 0.665578 w_k1_MAT_s104
                                 + 1.20008 w_k1_MSFT_s104
                                 + 1.03606 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.11063 w_k1_ADM_s104
                                 + 1.21613 w_k1_AXP_s104
                                 + 0.95625 w_k1_DIS_s104
                                 + 1.08082 w_k1_FDX_s104
                                 + 1.04902 w_k1_IBM_s104
                                 + 0.99588 w_k1_INTC_s104
                                 + 1.04817 w_k1_KO_s104 + 1.34575 w_k1_MAT_s104
                                 + 1.04074 w_k1_MSFT_s104
                                 + 0.950462 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 0.727532 w_k1_ADM_s105
                                 + 0.780269 w_k1_AXP_s105
                                 + 0.823423 w_k1_DIS_s105
                                 + 0.860897 w_k1_FDX_s105
                                 + 1.28205 w_k1_IBM_s105
                                 + 0.747458 w_k1_INTC_s105
                                 + 1.00332 w_k1_KO_s105 + 1.03078 w_k1_MAT_s105
                                 + 0.839167 w_k1_MSFT_s105
                                 + 0.912802 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.04458 w_k1_ADM_s105
                                 + 0.97054 w_k1_AXP_s105
                                 + 1.51751 w_k1_DIS_s105
                                 + 1.22609 w_k1_FDX_s105
                                 + 0.843884 w_k1_IBM_s105
                                 + 0.91317 w_k1_INTC_s105
                                 + 0.963365 w_k1_KO_s105
                                 + 0.57406 w_k1_MAT_s105
                                 + 1.19005 w_k1_MSFT_s105
                                 + 0.921263 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.4149 w_k1_ADM_s105
                                 + 1.55136 w_k1_AXP_s105
                                 + 2.04507 w_k1_DIS_s105
                                 + 1.43415 w_k1_FDX_s105
                                 + 1.49659 w_k1_IBM_s105
                                 + 1.22941 w_k1_INTC_s105
                                 + 1.50815 w_k1_KO_s105 + 1.78004 w_k1_MAT_s105
                                 + 1.28061 w_k1_MSFT_s105
                                 + 1.23141 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.786575 w_k1_ADM_s105
                                 + 0.610417 w_k1_AXP_s105
                                 + 0.637695 w_k1_DIS_s105
                                 + 0.731822 w_k1_FDX_s105
                                 + 0.765227 w_k1_IBM_s105
                                 + 0.868181 w_k1_INTC_s105
                                 + 0.826944 w_k1_KO_s105
                                 + 0.520454 w_k1_MAT_s105
                                 + 0.835226 w_k1_MSFT_s105
                                 + 0.773893 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.965883 w_k1_ADM_s105
                                 + 0.902693 w_k1_AXP_s105
                                 + 0.868751 w_k1_DIS_s105
                                 + 1.05955 w_k1_FDX_s105
                                 + 0.728955 w_k1_IBM_s105
                                 + 0.802024 w_k1_INTC_s105
                                 + 0.849465 w_k1_KO_s105
                                 + 0.706181 w_k1_MAT_s105
                                 + 0.957836 w_k1_MSFT_s105
                                 + 0.918878 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.07032 w_k1_ADM_s105
                                 + 1.45952 w_k1_AXP_s105
                                 + 1.16521 w_k1_DIS_s105
                                 + 1.15873 w_k1_FDX_s105
                                 + 0.971564 w_k1_IBM_s105
                                 + 1.11001 w_k1_INTC_s105
                                 + 1.10954 w_k1_KO_s105 + 1.06493 w_k1_MAT_s105
                                 + 1.10899 w_k1_MSFT_s105
                                 + 1.06022 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.1729 w_k1_ADM_s105
                                 + 1.09339 w_k1_AXP_s105
                                 + 1.35672 w_k1_DIS_s105
                                 + 1.11315 w_k1_FDX_s105
                                 + 1.42141 w_k1_IBM_s105
                                 + 1.04442 w_k1_INTC_s105
                                 + 1.41504 w_k1_KO_s105 + 1.03868 w_k1_MAT_s105
                                 + 1.12761 w_k1_MSFT_s105
                                 + 1.12559 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.874695 w_k1_ADM_s105
                                 + 0.887221 w_k1_AXP_s105
                                 + 1.1647 w_k1_DIS_s105 + 1.02055 w_k1_FDX_s105
                                 + 1.01703 w_k1_IBM_s105
                                 + 0.959845 w_k1_INTC_s105
                                 + 1.19609 w_k1_KO_s105
                                 + 0.849982 w_k1_MAT_s105
                                 + 1.14317 w_k1_MSFT_s105
                                 + 0.96609 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.2914 w_k1_ADM_s106
                                 + 1.49763 w_k1_AXP_s106
                                 + 1.82121 w_k1_DIS_s106 + 1.4102 w_k1_FDX_s106
                                 + 1.23791 w_k1_IBM_s106
                                 + 1.44004 w_k1_INTC_s106
                                 + 1.18899 w_k1_KO_s106 + 1.04785 w_k1_MAT_s106
                                 + 1.38642 w_k1_MSFT_s106
                                 + 1.31773 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.902693 w_k1_ADM_s106
                                 + 0.833512 w_k1_AXP_s106
                                 + 0.912327 w_k1_DIS_s106
                                 + 0.870316 w_k1_FDX_s106
                                 + 1.35162 w_k1_IBM_s106
                                 + 0.778689 w_k1_INTC_s106
                                 + 0.992877 w_k1_KO_s106
                                 + 1.49026 w_k1_MAT_s106
                                 + 0.914335 w_k1_MSFT_s106
                                 + 0.95536 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.08439 w_k1_ADM_s106
                                 + 1.30286 w_k1_AXP_s106
                                 + 1.47856 w_k1_DIS_s106
                                 + 1.14007 w_k1_FDX_s106
                                 + 1.22647 w_k1_IBM_s106
                                 + 1.05766 w_k1_INTC_s106 + 1.2368 w_k1_KO_s106
                                 + 1.56427 w_k1_MAT_s106
                                 + 1.10924 w_k1_MSFT_s106
                                 + 1.08914 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.721818 w_k1_ADM_s106
                                 + 0.784141 w_k1_AXP_s106
                                 + 0.740785 w_k1_DIS_s106
                                 + 0.858995 w_k1_FDX_s106
                                 + 0.74029 w_k1_IBM_s106
                                 + 0.933208 w_k1_INTC_s106
                                 + 0.699789 w_k1_KO_s106
                                 + 0.697611 w_k1_MAT_s106
                                 + 0.794879 w_k1_MSFT_s106
                                 + 0.829093 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.852663 w_k1_ADM_s106
                                 + 1.18096 w_k1_AXP_s106
                                 + 1.26305 w_k1_DIS_s106
                                 + 1.16951 w_k1_FDX_s106
                                 + 0.912587 w_k1_IBM_s106
                                 + 1.11822 w_k1_INTC_s106
                                 + 0.906376 w_k1_KO_s106
                                 + 0.886909 w_k1_MAT_s106
                                 + 1.07597 w_k1_MSFT_s106
                                 + 1.03266 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.92514 w_k1_ADM_s106
                                 + 1.18385 w_k1_AXP_s106
                                 + 1.24771 w_k1_DIS_s106
                                 + 1.13745 w_k1_FDX_s106
                                 + 1.56129 w_k1_IBM_s106
                                 + 1.02343 w_k1_INTC_s106
                                 + 1.35176 w_k1_KO_s106 + 1.63652 w_k1_MAT_s106
                                 + 1.17588 w_k1_MSFT_s106
                                 + 1.12728 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.884949 w_k1_ADM_s106
                                 + 0.908322 w_k1_AXP_s106
                                 + 1.31207 w_k1_DIS_s106
                                 + 1.13329 w_k1_FDX_s106
                                 + 1.07967 w_k1_IBM_s106
                                 + 1.0628 w_k1_INTC_s106 + 1.02951 w_k1_KO_s106
                                 + 1.11858 w_k1_MAT_s106 + 1.019 w_k1_MSFT_s106
                                 + 1.07998 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.909964 w_k1_ADM_s106
                                 + 0.891573 w_k1_AXP_s106
                                 + 0.868218 w_k1_DIS_s106
                                 + 0.93879 w_k1_FDX_s106
                                 + 1.00972 w_k1_IBM_s106
                                 + 0.893119 w_k1_INTC_s106
                                 + 1.00032 w_k1_KO_s106 + 1.04545 w_k1_MAT_s106
                                 + 0.944711 w_k1_MSFT_s106
                                 + 0.934101 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.49961 w_k1_ADM_s107
                                 + 2.04048 w_k1_AXP_s107
                                 + 2.02514 w_k1_DIS_s107
                                 + 1.52637 w_k1_FDX_s107
                                 + 1.36124 w_k1_IBM_s107
                                 + 2.0111 w_k1_INTC_s107 + 1.59224 w_k1_KO_s107
                                 + 1.86132 w_k1_MAT_s107
                                 + 1.5736 w_k1_MSFT_s107 + 1.3858 w_k1_PG_s107
                                 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.903592 w_k1_ADM_s107
                                 + 0.706185 w_k1_AXP_s107
                                 + 0.720455 w_k1_DIS_s107
                                 + 0.817819 w_k1_FDX_s107
                                 + 0.661485 w_k1_IBM_s107
                                 + 0.916798 w_k1_INTC_s107
                                 + 0.829154 w_k1_KO_s107
                                 + 0.452736 w_k1_MAT_s107
                                 + 0.878013 w_k1_MSFT_s107
                                 + 0.815075 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.10036 w_k1_ADM_s107
                                 + 1.01904 w_k1_AXP_s107
                                 + 1.09702 w_k1_DIS_s107
                                 + 1.02695 w_k1_FDX_s107
                                 + 0.875569 w_k1_IBM_s107
                                 + 1.202 w_k1_INTC_s107 + 1.08461 w_k1_KO_s107
                                 + 0.827792 w_k1_MAT_s107
                                 + 0.997002 w_k1_MSFT_s107
                                 + 1.08918 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.27006 w_k1_ADM_s107
                                 + 1.52732 w_k1_AXP_s107
                                 + 1.10564 w_k1_DIS_s107
                                 + 1.21662 w_k1_FDX_s107
                                 + 0.881449 w_k1_IBM_s107
                                 + 1.25496 w_k1_INTC_s107
                                 + 1.19101 w_k1_KO_s107 + 1.21007 w_k1_MAT_s107
                                 + 1.14103 w_k1_MSFT_s107
                                 + 1.17476 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.16174 w_k1_ADM_s107
                                 + 1.44982 w_k1_AXP_s107
                                 + 1.52651 w_k1_DIS_s107
                                 + 1.17432 w_k1_FDX_s107
                                 + 1.56599 w_k1_IBM_s107
                                 + 1.35182 w_k1_INTC_s107 + 1.5515 w_k1_KO_s107
                                 + 1.83905 w_k1_MAT_s107
                                 + 1.16218 w_k1_MSFT_s107
                                 + 1.39118 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.39065 w_k1_ADM_s107
                                 + 1.29085 w_k1_AXP_s107
                                 + 1.50883 w_k1_DIS_s107
                                 + 1.32578 w_k1_FDX_s107
                                 + 1.00469 w_k1_IBM_s107
                                 + 1.33852 w_k1_INTC_s107
                                 + 1.21184 w_k1_KO_s107 + 1.16506 w_k1_MAT_s107
                                 + 1.3318 w_k1_MSFT_s107 + 1.35861 w_k1_PG_s107
                                 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.99258 w_k1_ADM_s107
                                 + 0.916944 w_k1_AXP_s107
                                 + 0.864683 w_k1_DIS_s107
                                 + 0.86792 w_k1_FDX_s107
                                 + 1.11142 w_k1_IBM_s107
                                 + 0.951631 w_k1_INTC_s107
                                 + 1.06345 w_k1_KO_s107
                                 + 0.991938 w_k1_MAT_s107
                                 + 0.916148 w_k1_MSFT_s107
                                 + 1.06134 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.08136 w_k1_ADM_s107
                                 + 1.01284 w_k1_AXP_s107
                                 + 1.37534 w_k1_DIS_s107
                                 + 0.969539 w_k1_FDX_s107
                                 + 1.44084 w_k1_IBM_s107
                                 + 1.24409 w_k1_INTC_s107
                                 + 1.42468 w_k1_KO_s107 + 1.15817 w_k1_MAT_s107
                                 + 1.09336 w_k1_MSFT_s107
                                 + 1.09891 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.11264 w_k1_ADM_s110
                                 + 1.32249 w_k1_AXP_s110
                                 + 1.20462 w_k1_DIS_s110
                                 + 1.19301 w_k1_FDX_s110
                                 + 0.923907 w_k1_IBM_s110
                                 + 1.30576 w_k1_INTC_s110
                                 + 0.961769 w_k1_KO_s110
                                 + 1.12725 w_k1_MAT_s110
                                 + 1.1489 w_k1_MSFT_s110 + 1.00256 w_k1_PG_s110
                                 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.09886 w_k1_ADM_s110
                                 + 1.13665 w_k1_AXP_s110
                                 + 1.18383 w_k1_DIS_s110
                                 + 1.12469 w_k1_FDX_s110
                                 + 1.42421 w_k1_IBM_s110
                                 + 1.3484 w_k1_INTC_s110 + 1.20686 w_k1_KO_s110
                                 + 1.8052 w_k1_MAT_s110
                                 + 1.19467 w_k1_MSFT_s110
                                 + 1.02694 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.735058 w_k1_ADM_s110
                                 + 1.00665 w_k1_AXP_s110
                                 + 0.91666 w_k1_DIS_s110
                                 + 1.09014 w_k1_FDX_s110
                                 + 0.652741 w_k1_IBM_s110
                                 + 1.04339 w_k1_INTC_s110
                                 + 0.678103 w_k1_KO_s110
                                 + 1.03312 w_k1_MAT_s110
                                 + 1.0578 w_k1_MSFT_s110
                                 + 0.885876 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.929505 w_k1_ADM_s110
                                 + 0.751295 w_k1_AXP_s110
                                 + 0.792908 w_k1_DIS_s110
                                 + 0.766025 w_k1_FDX_s110
                                 + 0.97012 w_k1_IBM_s110
                                 + 0.718024 w_k1_INTC_s110
                                 + 0.808868 w_k1_KO_s110
                                 + 0.926828 w_k1_MAT_s110
                                 + 0.843709 w_k1_MSFT_s110
                                 + 0.932658 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.978745 w_k1_ADM_s110
                                 + 0.635881 w_k1_AXP_s110
                                 + 0.633296 w_k1_DIS_s110
                                 + 0.835296 w_k1_FDX_s110
                                 + 0.691095 w_k1_IBM_s110
                                 + 0.829885 w_k1_INTC_s110
                                 + 0.733701 w_k1_KO_s110
                                 + 0.73276 w_k1_MAT_s110
                                 + 0.856599 w_k1_MSFT_s110
                                 + 0.838683 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.660774 w_k1_ADM_s110
                                 + 0.653211 w_k1_AXP_s110
                                 + 0.553358 w_k1_DIS_s110
                                 + 0.74961 w_k1_FDX_s110
                                 + 0.789349 w_k1_IBM_s110
                                 + 0.793872 w_k1_INTC_s110
                                 + 0.673684 w_k1_KO_s110
                                 + 0.865518 w_k1_MAT_s110
                                 + 0.658038 w_k1_MSFT_s110
                                 + 0.723344 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.6518 w_k1_ADM_s110
                                 + 0.653792 w_k1_AXP_s110
                                 + 0.639694 w_k1_DIS_s110
                                 + 0.726602 w_k1_FDX_s110
                                 + 0.550865 w_k1_IBM_s110
                                 + 0.987186 w_k1_INTC_s110
                                 + 0.589612 w_k1_KO_s110
                                 + 0.559909 w_k1_MAT_s110
                                 + 0.855743 w_k1_MSFT_s110
                                 + 0.785497 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.20459 w_k1_ADM_s110
                                 + 0.949123 w_k1_AXP_s110
                                 + 1.00619 w_k1_DIS_s110
                                 + 0.952567 w_k1_FDX_s110
                                 + 0.9393 w_k1_IBM_s110
                                 + 0.92117 w_k1_INTC_s110
                                 + 0.874364 w_k1_KO_s110
                                 + 0.991736 w_k1_MAT_s110
                                 + 0.985093 w_k1_MSFT_s110
                                 + 1.05151 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.19381 w_k1_ADM_s111
                                 + 1.06821 w_k1_AXP_s111
                                 + 1.21916 w_k1_DIS_s111
                                 + 1.15429 w_k1_FDX_s111
                                 + 1.08896 w_k1_IBM_s111
                                 + 1.15116 w_k1_INTC_s111
                                 + 1.30816 w_k1_KO_s111 + 1.39702 w_k1_MAT_s111
                                 + 1.28898 w_k1_MSFT_s111
                                 + 1.05329 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.13033 w_k1_ADM_s111
                                 + 1.19301 w_k1_AXP_s111
                                 + 1.11949 w_k1_DIS_s111
                                 + 1.12483 w_k1_FDX_s111
                                 + 1.69449 w_k1_IBM_s111
                                 + 1.06231 w_k1_INTC_s111
                                 + 1.60568 w_k1_KO_s111 + 2.74528 w_k1_MAT_s111
                                 + 1.25596 w_k1_MSFT_s111
                                 + 1.27013 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.06887 w_k1_ADM_s111
                                 + 1.5679 w_k1_AXP_s111 + 1.46854 w_k1_DIS_s111
                                 + 1.43874 w_k1_FDX_s111
                                 + 0.971685 w_k1_IBM_s111
                                 + 1.30261 w_k1_INTC_s111
                                 + 1.03532 w_k1_KO_s111 + 1.49849 w_k1_MAT_s111
                                 + 1.35356 w_k1_MSFT_s111 + 1.1181 w_k1_PG_s111
                                 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.08002 w_k1_ADM_s111
                                 + 0.843019 w_k1_AXP_s111
                                 + 0.84411 w_k1_DIS_s111
                                 + 1.08631 w_k1_FDX_s111
                                 + 0.585366 w_k1_IBM_s111
                                 + 0.936708 w_k1_INTC_s111
                                 + 0.772529 w_k1_KO_s111
                                 + 0.595655 w_k1_MAT_s111
                                 + 1.09842 w_k1_MSFT_s111
                                 + 0.872977 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.693761 w_k1_ADM_s111
                                 + 0.685433 w_k1_AXP_s111
                                 + 0.642541 w_k1_DIS_s111
                                 + 0.847192 w_k1_FDX_s111
                                 + 0.745685 w_k1_IBM_s111
                                 + 0.7989 w_k1_INTC_s111
                                 + 0.825174 w_k1_KO_s111
                                 + 0.831201 w_k1_MAT_s111
                                 + 0.870158 w_k1_MSFT_s111
                                 + 0.791163 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.92779 w_k1_ADM_s111
                                 + 0.868784 w_k1_AXP_s111
                                 + 0.856561 w_k1_DIS_s111
                                 + 0.991758 w_k1_FDX_s111
                                 + 0.875209 w_k1_IBM_s111
                                 + 1.19228 w_k1_INTC_s111
                                 + 0.977935 w_k1_KO_s111
                                 + 1.43978 w_k1_MAT_s111
                                 + 1.10011 w_k1_MSFT_s111
                                 + 0.947645 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.37524 w_k1_ADM_s111
                                 + 1.24396 w_k1_AXP_s111
                                 + 2.04015 w_k1_DIS_s111
                                 + 1.50975 w_k1_FDX_s111
                                 + 1.32245 w_k1_IBM_s111
                                 + 1.42537 w_k1_INTC_s111
                                 + 1.61336 w_k1_KO_s111 + 1.58073 w_k1_MAT_s111
                                 + 1.59207 w_k1_MSFT_s111
                                 + 1.21908 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.08657 w_k1_ADM_s111
                                 + 0.81499 w_k1_AXP_s111
                                 + 0.913485 w_k1_DIS_s111
                                 + 0.929557 w_k1_FDX_s111
                                 + 1.20779 w_k1_IBM_s111
                                 + 0.892549 w_k1_INTC_s111
                                 + 1.07506 w_k1_KO_s111 + 1.07471 w_k1_MAT_s111
                                 + 0.972549 w_k1_MSFT_s111
                                 + 0.929283 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.12213 w_k1_ADM_s112
                                 + 0.816731 w_k1_AXP_s112
                                 + 0.816699 w_k1_DIS_s112
                                 + 0.87424 w_k1_FDX_s112
                                 + 0.845843 w_k1_IBM_s112
                                 + 0.939851 w_k1_INTC_s112
                                 + 0.876202 w_k1_KO_s112
                                 + 0.809366 w_k1_MAT_s112
                                 + 1.0052 w_k1_MSFT_s112 + 1.05437 w_k1_PG_s112
                                 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.985295 w_k1_ADM_s112
                                 + 1.02999 w_k1_AXP_s112
                                 + 0.851387 w_k1_DIS_s112
                                 + 0.937533 w_k1_FDX_s112
                                 + 1.13304 w_k1_IBM_s112
                                 + 0.761554 w_k1_INTC_s112
                                 + 0.944221 w_k1_KO_s112
                                 + 1.09003 w_k1_MAT_s112
                                 + 0.831215 w_k1_MSFT_s112
                                 + 1.02772 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.73013 w_k1_ADM_s112
                                 + 1.36816 w_k1_AXP_s112 + 1.558 w_k1_DIS_s112
                                 + 1.3146 w_k1_FDX_s112 + 1.37019 w_k1_IBM_s112
                                 + 1.15215 w_k1_INTC_s112
                                 + 1.42099 w_k1_KO_s112 + 1.12801 w_k1_MAT_s112
                                 + 1.34163 w_k1_MSFT_s112
                                 + 1.26932 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.803686 w_k1_ADM_s112
                                 + 0.865362 w_k1_AXP_s112
                                 + 0.640088 w_k1_DIS_s112
                                 + 0.77343 w_k1_FDX_s112
                                 + 1.07677 w_k1_IBM_s112
                                 + 0.781723 w_k1_INTC_s112
                                 + 0.835103 w_k1_KO_s112
                                 + 1.08869 w_k1_MAT_s112
                                 + 0.788357 w_k1_MSFT_s112
                                 + 0.951414 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.22291 w_k1_ADM_s112
                                 + 1.38118 w_k1_AXP_s112
                                 + 1.07321 w_k1_DIS_s112
                                 + 0.966199 w_k1_FDX_s112
                                 + 1.40614 w_k1_IBM_s112
                                 + 0.945256 w_k1_INTC_s112
                                 + 1.28753 w_k1_KO_s112 + 1.84525 w_k1_MAT_s112
                                 + 0.921891 w_k1_MSFT_s112
                                 + 1.23637 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.217 w_k1_ADM_s112
                                 + 1.09944 w_k1_AXP_s112
                                 + 1.28145 w_k1_DIS_s112
                                 + 1.22229 w_k1_FDX_s112
                                 + 0.726762 w_k1_IBM_s112
                                 + 1.04514 w_k1_INTC_s112
                                 + 1.03333 w_k1_KO_s112
                                 + 0.886309 w_k1_MAT_s112
                                 + 1.17785 w_k1_MSFT_s112
                                 + 1.13856 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.905165 w_k1_ADM_s112
                                 + 0.913312 w_k1_AXP_s112
                                 + 0.635787 w_k1_DIS_s112
                                 + 0.825695 w_k1_FDX_s112
                                 + 0.705866 w_k1_IBM_s112
                                 + 0.860157 w_k1_INTC_s112
                                 + 0.665552 w_k1_KO_s112
                                 + 0.72804 w_k1_MAT_s112
                                 + 0.800464 w_k1_MSFT_s112
                                 + 0.916748 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.24836 w_k1_ADM_s112
                                 + 1.1586 w_k1_AXP_s112 + 1.11766 w_k1_DIS_s112
                                 + 1.1757 w_k1_FDX_s112
                                 + 0.938693 w_k1_IBM_s112
                                 + 0.885233 w_k1_INTC_s112
                                 + 1.01875 w_k1_KO_s112 + 1.25067 w_k1_MAT_s112
                                 + 1.12949 w_k1_MSFT_s112
                                 + 1.13664 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.3513 w_k1_ADM_s113
                                 + 1.16459 w_k1_AXP_s113
                                 + 1.36456 w_k1_DIS_s113
                                 + 1.19845 w_k1_FDX_s113
                                 + 0.89279 w_k1_IBM_s113
                                 + 1.0067 w_k1_INTC_s113 + 1.2432 w_k1_KO_s113
                                 + 0.715355 w_k1_MAT_s113
                                 + 1.06483 w_k1_MSFT_s113
                                 + 1.06988 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.828625 w_k1_ADM_s113
                                 + 0.904184 w_k1_AXP_s113
                                 + 0.74432 w_k1_DIS_s113
                                 + 0.907255 w_k1_FDX_s113
                                 + 0.808805 w_k1_IBM_s113
                                 + 1.042 w_k1_INTC_s113 + 0.841345 w_k1_KO_s113
                                 + 0.669886 w_k1_MAT_s113
                                 + 0.814418 w_k1_MSFT_s113
                                 + 0.774791 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.06999 w_k1_ADM_s113
                                 + 1.50159 w_k1_AXP_s113
                                 + 1.56838 w_k1_DIS_s113
                                 + 1.30608 w_k1_FDX_s113
                                 + 1.55354 w_k1_IBM_s113
                                 + 1.32856 w_k1_INTC_s113
                                 + 1.51776 w_k1_KO_s113 + 1.34582 w_k1_MAT_s113
                                 + 1.21797 w_k1_MSFT_s113
                                 + 1.10973 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.35323 w_k1_ADM_s113
                                 + 0.992915 w_k1_AXP_s113
                                 + 1.0743 w_k1_DIS_s113 + 1.23626 w_k1_FDX_s113
                                 + 0.6402 w_k1_IBM_s113
                                 + 0.963348 w_k1_INTC_s113
                                 + 0.87859 w_k1_KO_s113
                                 + 0.535478 w_k1_MAT_s113
                                 + 0.98294 w_k1_MSFT_s113
                                 + 0.94339 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 0.901691 w_k1_ADM_s113
                                 + 1.20902 w_k1_AXP_s113
                                 + 0.846639 w_k1_DIS_s113
                                 + 0.973298 w_k1_FDX_s113
                                 + 1.45883 w_k1_IBM_s113
                                 + 1.01312 w_k1_INTC_s113
                                 + 1.27008 w_k1_KO_s113 + 1.36597 w_k1_MAT_s113
                                 + 0.777327 w_k1_MSFT_s113
                                 + 0.92631 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 1.14472 w_k1_ADM_s113
                                 + 1.06004 w_k1_AXP_s113
                                 + 1.05486 w_k1_DIS_s113
                                 + 1.06735 w_k1_FDX_s113 + 1.09 w_k1_IBM_s113
                                 + 0.800592 w_k1_INTC_s113
                                 + 1.17734 w_k1_KO_s113 + 1.09436 w_k1_MAT_s113
                                 + 0.970009 w_k1_MSFT_s113
                                 + 1.00187 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.21721 w_k1_ADM_s113
                                 + 1.19931 w_k1_AXP_s113
                                 + 0.894976 w_k1_DIS_s113
                                 + 1.0759 w_k1_FDX_s113
                                 + 0.913914 w_k1_IBM_s113
                                 + 0.895004 w_k1_INTC_s113
                                 + 0.92244 w_k1_KO_s113 + 1.19485 w_k1_MAT_s113
                                 + 0.891788 w_k1_MSFT_s113
                                 + 1.06064 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.613742 w_k1_ADM_s113
                                 + 0.603788 w_k1_AXP_s113
                                 + 0.535412 w_k1_DIS_s113
                                 + 0.770136 w_k1_FDX_s113
                                 + 0.679605 w_k1_IBM_s113
                                 + 0.749592 w_k1_INTC_s113
                                 + 0.73262 w_k1_KO_s113
                                 + 0.490281 w_k1_MAT_s113
                                 + 0.699197 w_k1_MSFT_s113
                                 + 0.809189 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.834761 w_k1_ADM_s114
                                 + 0.911111 w_k1_AXP_s114
                                 + 0.769977 w_k1_DIS_s114
                                 + 0.859257 w_k1_FDX_s114
                                 + 1.2753 w_k1_IBM_s114
                                 + 0.91027 w_k1_INTC_s114
                                 + 0.950553 w_k1_KO_s114
                                 + 1.42381 w_k1_MAT_s114
                                 + 0.769254 w_k1_MSFT_s114
                                 + 0.914698 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.878508 w_k1_ADM_s114
                                 + 1.43781 w_k1_AXP_s114
                                 + 1.26972 w_k1_DIS_s114
                                 + 1.13817 w_k1_FDX_s114
                                 + 1.41314 w_k1_IBM_s114
                                 + 1.28074 w_k1_INTC_s114
                                 + 1.19628 w_k1_KO_s114 + 1.56191 w_k1_MAT_s114
                                 + 1.09214 w_k1_MSFT_s114
                                 + 1.08943 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.987281 w_k1_ADM_s114
                                 + 1.17469 w_k1_AXP_s114
                                 + 1.19052 w_k1_DIS_s114
                                 + 1.05174 w_k1_FDX_s114
                                 + 1.03712 w_k1_IBM_s114
                                 + 1.16686 w_k1_INTC_s114
                                 + 1.20356 w_k1_KO_s114 + 1.27534 w_k1_MAT_s114
                                 + 1.10073 w_k1_MSFT_s114
                                 + 1.15342 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.907812 w_k1_ADM_s114
                                 + 1.16146 w_k1_AXP_s114
                                 + 0.66412 w_k1_DIS_s114
                                 + 0.937866 w_k1_FDX_s114
                                 + 0.800034 w_k1_IBM_s114
                                 + 0.915659 w_k1_INTC_s114
                                 + 0.798957 w_k1_KO_s114
                                 + 1.07851 w_k1_MAT_s114
                                 + 0.881961 w_k1_MSFT_s114
                                 + 0.968156 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.800114 w_k1_ADM_s114
                                 + 0.683873 w_k1_AXP_s114
                                 + 0.979752 w_k1_DIS_s114
                                 + 0.862363 w_k1_FDX_s114
                                 + 1.04273 w_k1_IBM_s114
                                 + 0.907197 w_k1_INTC_s114
                                 + 1.0142 w_k1_KO_s114 + 0.915134 w_k1_MAT_s114
                                 + 0.98334 w_k1_MSFT_s114
                                 + 0.98568 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.992899 w_k1_ADM_s114
                                 + 1.0834 w_k1_AXP_s114 + 1.07714 w_k1_DIS_s114
                                 + 1.11646 w_k1_FDX_s114
                                 + 0.980044 w_k1_IBM_s114
                                 + 1.09659 w_k1_INTC_s114
                                 + 1.04912 w_k1_KO_s114
                                 + 0.685611 w_k1_MAT_s114
                                 + 1.06427 w_k1_MSFT_s114
                                 + 1.11745 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.934973 w_k1_ADM_s114
                                 + 0.678378 w_k1_AXP_s114
                                 + 0.606616 w_k1_DIS_s114
                                 + 0.731451 w_k1_FDX_s114
                                 + 0.766116 w_k1_IBM_s114
                                 + 0.783521 w_k1_INTC_s114
                                 + 0.798965 w_k1_KO_s114
                                 + 0.741543 w_k1_MAT_s114
                                 + 0.834677 w_k1_MSFT_s114
                                 + 0.973233 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.854251 w_k1_ADM_s114
                                 + 0.706994 w_k1_AXP_s114
                                 + 0.795518 w_k1_DIS_s114
                                 + 1.00435 w_k1_FDX_s114
                                 + 0.550142 w_k1_IBM_s114
                                 + 0.804865 w_k1_INTC_s114
                                 + 0.710257 w_k1_KO_s114
                                 + 0.473995 w_k1_MAT_s114
                                 + 0.887089 w_k1_MSFT_s114
                                 + 0.91188 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.16026 w_k1_ADM_s115
                                 + 0.933374 w_k1_AXP_s115
                                 + 1.30362 w_k1_DIS_s115
                                 + 1.11767 w_k1_FDX_s115
                                 + 0.936399 w_k1_IBM_s115
                                 + 0.973469 w_k1_INTC_s115
                                 + 1.01211 w_k1_KO_s115
                                 + 0.821435 w_k1_MAT_s115
                                 + 1.12919 w_k1_MSFT_s115
                                 + 1.16064 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.94086 w_k1_ADM_s115
                                 + 0.944293 w_k1_AXP_s115
                                 + 1.16787 w_k1_DIS_s115
                                 + 0.906929 w_k1_FDX_s115
                                 + 1.63112 w_k1_IBM_s115
                                 + 0.999251 w_k1_INTC_s115
                                 + 1.22168 w_k1_KO_s115 + 1.82112 w_k1_MAT_s115
                                 + 1.12124 w_k1_MSFT_s115 + 1.1533 w_k1_PG_s115
                                 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.788533 w_k1_ADM_s115
                                 + 0.789891 w_k1_AXP_s115
                                 + 0.769664 w_k1_DIS_s115
                                 + 0.86553 w_k1_FDX_s115
                                 + 0.758031 w_k1_IBM_s115
                                 + 0.811788 w_k1_INTC_s115
                                 + 0.66425 w_k1_KO_s115 + 0.83294 w_k1_MAT_s115
                                 + 0.928282 w_k1_MSFT_s115
                                 + 0.886815 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.966538 w_k1_ADM_s115
                                 + 0.92722 w_k1_AXP_s115
                                 + 0.754676 w_k1_DIS_s115
                                 + 0.831564 w_k1_FDX_s115
                                 + 1.02883 w_k1_IBM_s115
                                 + 0.815301 w_k1_INTC_s115
                                 + 0.800339 w_k1_KO_s115
                                 + 1.42057 w_k1_MAT_s115
                                 + 0.843376 w_k1_MSFT_s115
                                 + 0.899943 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.15929 w_k1_ADM_s115
                                 + 1.22376 w_k1_AXP_s115
                                 + 1.01231 w_k1_DIS_s115 + 1.13 w_k1_FDX_s115
                                 + 0.985114 w_k1_IBM_s115
                                 + 1.06354 w_k1_INTC_s115
                                 + 0.935842 w_k1_KO_s115
                                 + 1.18631 w_k1_MAT_s115
                                 + 1.05498 w_k1_MSFT_s115
                                 + 0.979315 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.34006 w_k1_ADM_s115
                                 + 1.50918 w_k1_AXP_s115
                                 + 2.06741 w_k1_DIS_s115
                                 + 1.43171 w_k1_FDX_s115 + 1.4372 w_k1_IBM_s115
                                 + 1.24901 w_k1_INTC_s115
                                 + 1.38144 w_k1_KO_s115 + 1.77303 w_k1_MAT_s115
                                 + 1.46486 w_k1_MSFT_s115
                                 + 1.32143 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.124 w_k1_ADM_s115
                                 + 1.01634 w_k1_AXP_s115
                                 + 1.18291 w_k1_DIS_s115
                                 + 1.00134 w_k1_FDX_s115 + 1.2308 w_k1_IBM_s115
                                 + 0.91844 w_k1_INTC_s115
                                 + 1.04419 w_k1_KO_s115 + 1.07002 w_k1_MAT_s115
                                 + 1.0188 w_k1_MSFT_s115 + 1.04468 w_k1_PG_s115
                                 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.05688 w_k1_ADM_s115
                                 + 1.15535 w_k1_AXP_s115
                                 + 1.36213 w_k1_DIS_s115
                                 + 1.14608 w_k1_FDX_s115
                                 + 1.32509 w_k1_IBM_s115
                                 + 0.988868 w_k1_INTC_s115
                                 + 1.15656 w_k1_KO_s115 + 1.54156 w_k1_MAT_s115
                                 + 1.11811 w_k1_MSFT_s115
                                 + 1.13944 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.28673 w_k1_ADM_s116
                                 + 1.38565 w_k1_AXP_s116
                                 + 1.46818 w_k1_DIS_s116
                                 + 1.25789 w_k1_FDX_s116
                                 + 1.60436 w_k1_IBM_s116
                                 + 1.40027 w_k1_INTC_s116
                                 + 1.28825 w_k1_KO_s116 + 1.55894 w_k1_MAT_s116
                                 + 1.43817 w_k1_MSFT_s116
                                 + 1.11287 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.17217 w_k1_ADM_s116
                                 + 1.5109 w_k1_AXP_s116 + 1.61366 w_k1_DIS_s116
                                 + 1.43239 w_k1_FDX_s116
                                 + 1.04678 w_k1_IBM_s116
                                 + 1.47265 w_k1_INTC_s116
                                 + 1.11726 w_k1_KO_s116 + 1.2822 w_k1_MAT_s116
                                 + 1.4758 w_k1_MSFT_s116 + 1.19455 w_k1_PG_s116
                                 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.28061 w_k1_ADM_s116
                                 + 1.20472 w_k1_AXP_s116
                                 + 1.20731 w_k1_DIS_s116
                                 + 1.25645 w_k1_FDX_s116
                                 + 0.846213 w_k1_IBM_s116
                                 + 1.25965 w_k1_INTC_s116
                                 + 0.899376 w_k1_KO_s116
                                 + 0.753744 w_k1_MAT_s116
                                 + 1.35144 w_k1_MSFT_s116
                                 + 1.04944 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.0387 w_k1_ADM_s116
                                 + 1.07425 w_k1_AXP_s116
                                 + 0.952333 w_k1_DIS_s116
                                 + 1.0857 w_k1_FDX_s116 + 1.01695 w_k1_IBM_s116
                                 + 0.961347 w_k1_INTC_s116
                                 + 0.937676 w_k1_KO_s116
                                 + 0.946032 w_k1_MAT_s116
                                 + 1.02487 w_k1_MSFT_s116
                                 + 0.993465 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.990864 w_k1_ADM_s116
                                 + 0.964203 w_k1_AXP_s116
                                 + 0.77708 w_k1_DIS_s116
                                 + 0.844152 w_k1_FDX_s116
                                 + 1.21446 w_k1_IBM_s116
                                 + 0.824702 w_k1_INTC_s116
                                 + 0.918022 w_k1_KO_s116
                                 + 1.16371 w_k1_MAT_s116
                                 + 0.845218 w_k1_MSFT_s116
                                 + 0.957027 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.849749 w_k1_ADM_s116
                                 + 0.824159 w_k1_AXP_s116
                                 + 0.734021 w_k1_DIS_s116
                                 + 0.864893 w_k1_FDX_s116
                                 + 0.73675 w_k1_IBM_s116
                                 + 0.763736 w_k1_INTC_s116
                                 + 0.717029 w_k1_KO_s116
                                 + 0.750802 w_k1_MAT_s116
                                 + 0.83872 w_k1_MSFT_s116
                                 + 0.892987 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.15816 w_k1_ADM_s116
                                 + 0.970495 w_k1_AXP_s116
                                 + 0.994717 w_k1_DIS_s116
                                 + 1.14962 w_k1_FDX_s116
                                 + 0.621663 w_k1_IBM_s116
                                 + 1.08356 w_k1_INTC_s116
                                 + 0.792019 w_k1_KO_s116
                                 + 0.584384 w_k1_MAT_s116
                                 + 1.09695 w_k1_MSFT_s116
                                 + 0.990424 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.946203 w_k1_ADM_s116
                                 + 0.993725 w_k1_AXP_s116
                                 + 1.25667 w_k1_DIS_s116
                                 + 1.12456 w_k1_FDX_s116
                                 + 1.21952 w_k1_IBM_s116
                                 + 1.13986 w_k1_INTC_s116
                                 + 1.19451 w_k1_KO_s116 + 1.01674 w_k1_MAT_s116
                                 + 1.1598 w_k1_MSFT_s116 + 1.02913 w_k1_PG_s116
                                 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.03124 w_k1_ADM_s117
                                 + 1.09188 w_k1_AXP_s117
                                 + 1.28863 w_k1_DIS_s117
                                 + 1.24885 w_k1_FDX_s117
                                 + 0.950973 w_k1_IBM_s117
                                 + 1.10691 w_k1_INTC_s117
                                 + 1.12152 w_k1_KO_s117
                                 + 0.956615 w_k1_MAT_s117
                                 + 1.2694 w_k1_MSFT_s117 + 1.06613 w_k1_PG_s117
                                 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.27369 w_k1_ADM_s117
                                 + 1.08584 w_k1_AXP_s117
                                 + 0.919924 w_k1_DIS_s117
                                 + 1.0461 w_k1_FDX_s117
                                 + 0.786032 w_k1_IBM_s117
                                 + 0.946206 w_k1_INTC_s117
                                 + 0.842898 w_k1_KO_s117
                                 + 0.817659 w_k1_MAT_s117
                                 + 0.944584 w_k1_MSFT_s117
                                 + 0.991891 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.748191 w_k1_ADM_s117
                                 + 0.766764 w_k1_AXP_s117
                                 + 0.856305 w_k1_DIS_s117
                                 + 0.837505 w_k1_FDX_s117
                                 + 0.752685 w_k1_IBM_s117
                                 + 0.824491 w_k1_INTC_s117
                                 + 0.80572 w_k1_KO_s117
                                 + 0.573948 w_k1_MAT_s117
                                 + 0.863357 w_k1_MSFT_s117
                                 + 0.779291 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.0156 w_k1_ADM_s117
                                 + 0.990076 w_k1_AXP_s117
                                 + 1.01049 w_k1_DIS_s117
                                 + 1.01715 w_k1_FDX_s117
                                 + 1.41385 w_k1_IBM_s117
                                 + 1.1607 w_k1_INTC_s117 + 1.14327 w_k1_KO_s117
                                 + 1.22542 w_k1_MAT_s117
                                 + 1.18057 w_k1_MSFT_s117
                                 + 0.928617 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.41792 w_k1_ADM_s117
                                 + 1.2923 w_k1_AXP_s117 + 1.39089 w_k1_DIS_s117
                                 + 1.16989 w_k1_FDX_s117
                                 + 1.49635 w_k1_IBM_s117
                                 + 0.989296 w_k1_INTC_s117
                                 + 1.27381 w_k1_KO_s117 + 1.26311 w_k1_MAT_s117
                                 + 1.066 w_k1_MSFT_s117 + 1.0774 w_k1_PG_s117
                                 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.14192 w_k1_ADM_s117
                                 + 0.956192 w_k1_AXP_s117
                                 + 0.97474 w_k1_DIS_s117
                                 + 0.961216 w_k1_FDX_s117
                                 + 1.07999 w_k1_IBM_s117
                                 + 0.862528 w_k1_INTC_s117
                                 + 0.952615 w_k1_KO_s117
                                 + 0.878847 w_k1_MAT_s117
                                 + 0.960773 w_k1_MSFT_s117
                                 + 0.966381 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 0.835406 w_k1_ADM_s117
                                 + 0.964802 w_k1_AXP_s117
                                 + 0.789447 w_k1_DIS_s117
                                 + 0.892907 w_k1_FDX_s117
                                 + 0.937015 w_k1_IBM_s117
                                 + 0.983133 w_k1_INTC_s117
                                 + 0.945043 w_k1_KO_s117
                                 + 0.836917 w_k1_MAT_s117
                                 + 0.868687 w_k1_MSFT_s117
                                 + 0.853161 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.40963 w_k1_ADM_s117
                                 + 1.60205 w_k1_AXP_s117
                                 + 1.29029 w_k1_DIS_s117 + 1.2412 w_k1_FDX_s117
                                 + 1.06096 w_k1_IBM_s117
                                 + 0.864615 w_k1_INTC_s117
                                 + 1.13197 w_k1_KO_s117 + 1.24635 w_k1_MAT_s117
                                 + 1.07768 w_k1_MSFT_s117
                                 + 1.01218 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0259398179259353 w_k1_ADM_s000
                                 + 0.0199345863670849 w_k1_ADM_s001
                                 + 0.00848896736575848 w_k1_ADM_s002
                                 + 0.00903228028373099 w_k1_ADM_s003
                                 + 0.0222086472926204 w_k1_ADM_s004
                                 + 0.0358653972666048 w_k1_ADM_s005
                                 + 0.0304645585395669 w_k1_ADM_s006
                                 + 0.0192161588371622 w_k1_ADM_s007
                                 + 0.0147310355838255 w_k1_ADM_s010
                                 + 0.0189930279062772 w_k1_ADM_s011
                                 + 0.0694246847411126 w_k1_ADM_s012
                                 + 0.0202376715354926 w_k1_ADM_s013
                                 + 0.0368874153450069 w_k1_ADM_s014
                                 + 0.0181064467111632 w_k1_ADM_s015
                                 + 0.0219251021355685 w_k1_ADM_s016
                                 + 0.0219275132354092 w_k1_ADM_s017
                                 + 0.0371639894943645 w_k1_ADM_s100
                                 + 0.032489188896168 w_k1_ADM_s101
                                 + 0.0337323490940336 w_k1_ADM_s102
                                 + 0.112049060667701 w_k1_ADM_s103
                                 + 0.0397543861729499 w_k1_ADM_s104
                                 + 0.0424233428842471 w_k1_ADM_s105
                                 + 0.0384166554696039 w_k1_ADM_s106
                                 + 0.0898902974946454 w_k1_ADM_s107
                                 + 0.00990157979200715 w_k1_ADM_s110
                                 + 0.0151071881017151 w_k1_ADM_s111
                                 + 0.0363195089804004 w_k1_ADM_s112
                                 + 0.0111869050429252 w_k1_ADM_s113
                                 + 0.0194965567502442 w_k1_ADM_s114
                                 + 0.0265004711034147 w_k1_ADM_s115
                                 + 0.0477760370121102 w_k1_ADM_s116
                                 + 0.0114980518443091 w_k1_ADM_s117
                                 + 0.0244865839511251 w_k1_AXP_s000
                                 + 0.022465809807864 w_k1_AXP_s001
                                 + 0.00691514725885547 w_k1_AXP_s002
                                 + 0.0104544230458943 w_k1_AXP_s003
                                 + 0.0212048049084379 w_k1_AXP_s004
                                 + 0.0360871262574213 w_k1_AXP_s005
                                 + 0.0333860197447627 w_k1_AXP_s006
                                 + 0.0200181221921107 w_k1_AXP_s007
                                 + 0.0156877367194229 w_k1_AXP_s010
                                 + 0.0204996201370629 w_k1_AXP_s011
                                 + 0.0777774447795227 w_k1_AXP_s012
                                 + 0.0154075999827541 w_k1_AXP_s013
                                 + 0.0440214177648453 w_k1_AXP_s014
                                 + 0.0179326946814936 w_k1_AXP_s015
                                 + 0.0191329727102924 w_k1_AXP_s016
                                 + 0.0209359563413251 w_k1_AXP_s017
                                 + 0.0419789119397029 w_k1_AXP_s100
                                 + 0.0364398271549253 w_k1_AXP_s101
                                 + 0.0290617232811744 w_k1_AXP_s102
                                 + 0.13029776126025 w_k1_AXP_s103
                                 + 0.0394305124077809 w_k1_AXP_s104
                                 + 0.0428007366942896 w_k1_AXP_s105
                                 + 0.0438384238004891 w_k1_AXP_s106
                                 + 0.0873137010463919 w_k1_AXP_s107
                                 + 0.00967456808553327 w_k1_AXP_s110
                                 + 0.0140380962896254 w_k1_AXP_s111
                                 + 0.0338134468135066 w_k1_AXP_s112
                                 + 0.0116801299609645 w_k1_AXP_s113
                                 + 0.0227994155846844 w_k1_AXP_s114
                                 + 0.0257152741576726 w_k1_AXP_s115
                                 + 0.0482952857845915 w_k1_AXP_s116
                                 + 0.0116396395013715 w_k1_AXP_s117
                                 + 0.0273078471829932 w_k1_DIS_s000
                                 + 0.0219029840671468 w_k1_DIS_s001
                                 + 0.00744437122549275 w_k1_DIS_s002
                                 + 0.00983241054409805 w_k1_DIS_s003
                                 + 0.0213019094620659 w_k1_DIS_s004
                                 + 0.0393693480871517 w_k1_DIS_s005
                                 + 0.0320676077765449 w_k1_DIS_s006
                                 + 0.022090000690567 w_k1_DIS_s007
                                 + 0.0156580341634193 w_k1_DIS_s010
                                 + 0.0208159440052039 w_k1_DIS_s011
                                 + 0.0745838772413548 w_k1_DIS_s012
                                 + 0.0158063748173848 w_k1_DIS_s013
                                 + 0.0429901715204447 w_k1_DIS_s014
                                 + 0.0203648465671963 w_k1_DIS_s015
                                 + 0.0194713913804473 w_k1_DIS_s016
                                 + 0.023626216273759 w_k1_DIS_s017
                                 + 0.03934887069835 w_k1_DIS_s100
                                 + 0.0348717394660117 w_k1_DIS_s101
                                 + 0.0326003391891885 w_k1_DIS_s102
                                 + 0.120761970563115 w_k1_DIS_s103
                                 + 0.0389177128302125 w_k1_DIS_s104
                                 + 0.0478974173175827 w_k1_DIS_s105
                                 + 0.0470451310733616 w_k1_DIS_s106
                                 + 0.0909438076643362 w_k1_DIS_s107
                                 + 0.00959951735262387 w_k1_DIS_s110
                                 + 0.0148271141033701 w_k1_DIS_s111
                                 + 0.0311304364972777 w_k1_DIS_s112
                                 + 0.0106972906790031 w_k1_DIS_s113
                                 + 0.02095165766506 w_k1_DIS_s114
                                 + 0.0287725410986388 w_k1_DIS_s115
                                 + 0.0471081464863211 w_k1_DIS_s116
                                 + 0.011304268275214 w_k1_DIS_s117
                                 + 0.0260019577037551 w_k1_FDX_s000
                                 + 0.0215346388237069 w_k1_FDX_s001
                                 + 0.0073724215919267 w_k1_FDX_s002
                                 + 0.00984866844706055 w_k1_FDX_s003
                                 + 0.0203609742150519 w_k1_FDX_s004
                                 + 0.0357349700904169 w_k1_FDX_s005
                                 + 0.033449136779796 w_k1_FDX_s006
                                 + 0.0204006426731165 w_k1_FDX_s007
                                 + 0.0146003773925253 w_k1_FDX_s010
                                 + 0.0206177629905233 w_k1_FDX_s011
                                 + 0.0759299759355895 w_k1_FDX_s012
                                 + 0.016938647607765 w_k1_FDX_s013
                                 + 0.0426516123316842 w_k1_FDX_s014
                                 + 0.0181319645631827 w_k1_FDX_s015
                                 + 0.0191090726166432 w_k1_FDX_s016
                                 + 0.0222935424673275 w_k1_FDX_s017
                                 + 0.0383205065286617 w_k1_FDX_s100
                                 + 0.0363634880595075 w_k1_FDX_s101
                                 + 0.0298029225548514 w_k1_FDX_s102
                                 + 0.124256040368624 w_k1_FDX_s103
                                 + 0.0392099445488366 w_k1_FDX_s104
                                 + 0.0447696340407455 w_k1_FDX_s105
                                 + 0.0446491156837761 w_k1_FDX_s106
                                 + 0.0838292888114403 w_k1_FDX_s107
                                 + 0.00969611137140491 w_k1_FDX_s110
                                 + 0.0156918613902268 w_k1_FDX_s111
                                 + 0.0317076602571492 w_k1_FDX_s112
                                 + 0.0113074625044429 w_k1_FDX_s113
                                 + 0.0216227339833718 w_k1_FDX_s114
                                 + 0.0263403020259546 w_k1_FDX_s115
                                 + 0.0491551184725938 w_k1_FDX_s116
                                 + 0.0112159767711856 w_k1_FDX_s117
                                 + 0.0263466604997823 w_k1_IBM_s000
                                 + 0.0215289321919773 w_k1_IBM_s001
                                 + 0.00755981680870901 w_k1_IBM_s002
                                 + 0.0100744962446224 w_k1_IBM_s003
                                 + 0.0216016779953498 w_k1_IBM_s004
                                 + 0.0379175818447884 w_k1_IBM_s005
                                 + 0.0334003221247152 w_k1_IBM_s006
                                 + 0.0204603095286567 w_k1_IBM_s007
                                 + 0.0152535245080883 w_k1_IBM_s010
                                 + 0.0196623590488453 w_k1_IBM_s011
                                 + 0.0735100222826454 w_k1_IBM_s012
                                 + 0.0171179845026221 w_k1_IBM_s013
                                 + 0.0449253326438451 w_k1_IBM_s014
                                 + 0.0191298756780965 w_k1_IBM_s015
                                 + 0.0193027195434651 w_k1_IBM_s016
                                 + 0.0225489795962771 w_k1_IBM_s017
                                 + 0.0390418298790164 w_k1_IBM_s100
                                 + 0.0343103837844765 w_k1_IBM_s101
                                 + 0.0297601937021557 w_k1_IBM_s102
                                 + 0.125685432610711 w_k1_IBM_s103
                                 + 0.0396774779025592 w_k1_IBM_s104
                                 + 0.046967273313745 w_k1_IBM_s105
                                 + 0.0444476003830341 w_k1_IBM_s106
                                 + 0.0809507539390486 w_k1_IBM_s107
                                 + 0.00986300843393056 w_k1_IBM_s110
                                 + 0.0147212030646995 w_k1_IBM_s111
                                 + 0.0315580649043701 w_k1_IBM_s112
                                 + 0.0113495537857323 w_k1_IBM_s113
                                 + 0.0214216001589837 w_k1_IBM_s114
                                 + 0.0277985130628554 w_k1_IBM_s115
                                 + 0.0463336576797497 w_k1_IBM_s116
                                 + 0.0108513583999502 w_k1_IBM_s117
                                 + 0.0239172813255155 w_k1_INTC_s000
                                 + 0.0244337185327636 w_k1_INTC_s001
                                 + 0.00678574638341413 w_k1_INTC_s002
                                 + 0.00942228680403544 w_k1_INTC_s003
                                 + 0.0213760003441384 w_k1_INTC_s004
                                 + 0.0375645039893284 w_k1_INTC_s005
                                 + 0.0335655955956584 w_k1_INTC_s006
                                 + 0.0185794651953459 w_k1_INTC_s007
                                 + 0.0142260446524102 w_k1_INTC_s010
                                 + 0.019785200040444 w_k1_INTC_s011
                                 + 0.0845535588678943 w_k1_INTC_s012
                                 + 0.0137756647089445 w_k1_INTC_s013
                                 + 0.0397046809727953 w_k1_INTC_s014
                                 + 0.0197640828145043 w_k1_INTC_s015
                                 + 0.0170735186200174 w_k1_INTC_s016
                                 + 0.0257510088264349 w_k1_INTC_s017
                                 + 0.0346541603734861 w_k1_INTC_s100
                                 + 0.0360171431264754 w_k1_INTC_s101
                                 + 0.0308110656882925 w_k1_INTC_s102
                                 + 0.107512930172755 w_k1_INTC_s103
                                 + 0.0391311582266644 w_k1_INTC_s104
                                 + 0.0414373367652054 w_k1_INTC_s105
                                 + 0.0435715304955686 w_k1_INTC_s106
                                 + 0.0957211364482938 w_k1_INTC_s107
                                 + 0.0105154525122515 w_k1_INTC_s110
                                 + 0.0154845174222863 w_k1_INTC_s111
                                 + 0.0289550954377421 w_k1_INTC_s112
                                 + 0.0105289321851009 w_k1_INTC_s113
                                 + 0.0221197381781868 w_k1_INTC_s114
                                 + 0.0242918906337853 w_k1_INTC_s115
                                 + 0.0470267168826516 w_k1_INTC_s116
                                 + 0.0102086884812175 w_k1_INTC_s117
                                 + 0.0265575683279016 w_k1_KO_s000
                                 + 0.0250149431648161 w_k1_KO_s001
                                 + 0.00693647080938979 w_k1_KO_s002
                                 + 0.00948629100426116 w_k1_KO_s003
                                 + 0.0217665692260208 w_k1_KO_s004
                                 + 0.0390726871192393 w_k1_KO_s005
                                 + 0.0292577570319847 w_k1_KO_s006
                                 + 0.0192747776964248 w_k1_KO_s007
                                 + 0.0146018480036071 w_k1_KO_s010
                                 + 0.0203447557261148 w_k1_KO_s011
                                 + 0.0702951300438098 w_k1_KO_s012
                                 + 0.0163507102177234 w_k1_KO_s013
                                 + 0.0426416106495652 w_k1_KO_s014
                                 + 0.0191647049955555 w_k1_KO_s015
                                 + 0.0201821468044528 w_k1_KO_s016
                                 + 0.0241703068811983 w_k1_KO_s017
                                 + 0.0351786967535067 w_k1_KO_s100
                                 + 0.0317555626454743 w_k1_KO_s101
                                 + 0.0308712556958183 w_k1_KO_s102
                                 + 0.12652970880835 w_k1_KO_s103
                                 + 0.0394016791155373 w_k1_KO_s104
                                 + 0.0478839275042245 w_k1_KO_s105
                                 + 0.0415317213363847 w_k1_KO_s106
                                 + 0.0920768318221611 w_k1_KO_s107
                                 + 0.00901590512418625 w_k1_KO_s110
                                 + 0.015919084530498 w_k1_KO_s111
                                 + 0.0314141401184039 w_k1_KO_s112
                                 + 0.011857590516295 w_k1_KO_s113
                                 + 0.0213314897589422 w_k1_KO_s114
                                 + 0.0247014185321131 w_k1_KO_s115
                                 + 0.0434440033254271 w_k1_KO_s116
                                 + 0.010850606455648 w_k1_KO_s117
                                 + 0.0286038601940708 w_k1_MAT_s000
                                 + 0.0225153556547728 w_k1_MAT_s001
                                 + 0.00646611024661936 w_k1_MAT_s002
                                 + 0.0105680658827382 w_k1_MAT_s003
                                 + 0.0217437088418525 w_k1_MAT_s004
                                 + 0.0340870888772314 w_k1_MAT_s005
                                 + 0.0325524405193791 w_k1_MAT_s006
                                 + 0.0200208588195617 w_k1_MAT_s007
                                 + 0.0138210491538335 w_k1_MAT_s010
                                 + 0.0190617345373686 w_k1_MAT_s011
                                 + 0.0625827391877875 w_k1_MAT_s012
                                 + 0.0145729611653966 w_k1_MAT_s013
                                 + 0.0486599193881915 w_k1_MAT_s014
                                 + 0.0186241122982524 w_k1_MAT_s015
                                 + 0.0205677786474897 w_k1_MAT_s016
                                 + 0.0204576984315664 w_k1_MAT_s017
                                 + 0.0380613958880401 w_k1_MAT_s100
                                 + 0.0298513489998944 w_k1_MAT_s101
                                 + 0.0269937396251126 w_k1_MAT_s102
                                 + 0.158946974639936 w_k1_MAT_s103
                                 + 0.0396180536764259 w_k1_MAT_s104
                                 + 0.0403074438636332 w_k1_MAT_s105
                                 + 0.0452391597101911 w_k1_MAT_s106
                                 + 0.0786572187648814 w_k1_MAT_s107
                                 + 0.0115055378964319 w_k1_MAT_s110
                                 + 0.019099098104438 w_k1_MAT_s111
                                 + 0.0343551230318784 w_k1_MAT_s112
                                 + 0.0104405542221473 w_k1_MAT_s113
                                 + 0.0221275558061045 w_k1_MAT_s114
                                 + 0.0304624623444918 w_k1_MAT_s115
                                 + 0.0445692575719136 w_k1_MAT_s116
                                 + 0.0101302509665958 w_k1_MAT_s117
                                 + 0.0269314112462293 w_k1_MSFT_s000
                                 + 0.0240423706186825 w_k1_MSFT_s001
                                 + 0.00682136481998997 w_k1_MSFT_s002
                                 + 0.00915810145366237 w_k1_MSFT_s003
                                 + 0.0225531349896456 w_k1_MSFT_s004
                                 + 0.0373709821244043 w_k1_MSFT_s005
                                 + 0.0294180799024072 w_k1_MSFT_s006
                                 + 0.019575126012718 w_k1_MSFT_s007
                                 + 0.0140249352916754 w_k1_MSFT_s010
                                 + 0.0201895033208869 w_k1_MSFT_s011
                                 + 0.0740353039538256 w_k1_MSFT_s012
                                 + 0.0150174926327917 w_k1_MSFT_s013
                                 + 0.041545405338156 w_k1_MSFT_s014
                                 + 0.0200867333288878 w_k1_MSFT_s015
                                 + 0.0187310670208127 w_k1_MSFT_s016
                                 + 0.023980640004556 w_k1_MSFT_s017
                                 + 0.0363745604290118 w_k1_MSFT_s100
                                 + 0.0344875001303474 w_k1_MSFT_s101
                                 + 0.0308811417888384 w_k1_MSFT_s102
                                 + 0.119871413927507 w_k1_MSFT_s103
                                 + 0.0410287253788946 w_k1_MSFT_s104
                                 + 0.0449039464709018 w_k1_MSFT_s105
                                 + 0.0430387477553117 w_k1_MSFT_s106
                                 + 0.0859676882866459 w_k1_MSFT_s107
                                 + 0.00997103854712806 w_k1_MSFT_s110
                                 + 0.0166203036399032 w_k1_MSFT_s111
                                 + 0.0313820389184828 w_k1_MSFT_s112
                                 + 0.00988965221292398 w_k1_MSFT_s113
                                 + 0.0210416461082205 w_k1_MSFT_s114
                                 + 0.0271636287906745 w_k1_MSFT_s115
                                 + 0.049285648737419 w_k1_MSFT_s116
                                 + 0.0109349239172987 w_k1_MSFT_s117
                                 + 0.0261684788826013 w_k1_PG_s000
                                 + 0.0226090137635733 w_k1_PG_s001
                                 + 0.00726204977716074 w_k1_PG_s002
                                 + 0.00933586555253188 w_k1_PG_s003
                                 + 0.0216850717257175 w_k1_PG_s004
                                 + 0.0362524188075423 w_k1_PG_s005
                                 + 0.0310385697704062 w_k1_PG_s006
                                 + 0.0202382226399016 w_k1_PG_s007
                                 + 0.0140159829259934 w_k1_PG_s010
                                 + 0.0188474211584953 w_k1_PG_s011
                                 + 0.0669374780956098 w_k1_PG_s012
                                 + 0.0168635934099595 w_k1_PG_s013
                                 + 0.0430629684430373 w_k1_PG_s014
                                 + 0.0181563650897759 w_k1_PG_s015
                                 + 0.0189223685835953 w_k1_PG_s016
                                 + 0.0206212898764849 w_k1_PG_s017
                                 + 0.0384317897556998 w_k1_PG_s100
                                 + 0.0352894813579897 w_k1_PG_s101
                                 + 0.0284003751320431 w_k1_PG_s102
                                 + 0.121721620225477 w_k1_PG_s103
                                 + 0.0382737418612675 w_k1_PG_s104
                                 + 0.0426974790874555 w_k1_PG_s105
                                 + 0.0427803215891999 w_k1_PG_s106
                                 + 0.0884735918994531 w_k1_PG_s107
                                 + 0.00940892519259845 w_k1_PG_s110
                                 + 0.0144405895483421 w_k1_PG_s111
                                 + 0.0342275427609322 w_k1_PG_s112
                                 + 0.0102433673459633 w_k1_PG_s113
                                 + 0.0221680734872296 w_k1_PG_s114
                                 + 0.0270483949992997 w_k1_PG_s115
                                 + 0.0463915526118889 w_k1_PG_s116
                                 + 0.010159500799094 w_k1_PG_s117 >= 1.07
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
