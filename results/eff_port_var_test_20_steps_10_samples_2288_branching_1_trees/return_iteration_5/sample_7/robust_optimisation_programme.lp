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
 w_k1_ADM_s0_return_constraint:  1.14286 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.00555 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.18087 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.985412 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.992952 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.896937 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.714709 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.991421 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.849019 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.690262 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.37432 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.33 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.09426 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.17696 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.871641 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.39495 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.02675 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.809018 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.934079 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.954479 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.10016 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 1.36689 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.19433 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.33241 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 0.859005 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.21496 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.01985 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.4617 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.25559 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.00924 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.935569 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.955629 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.05445 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.11594 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.27331 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.10947 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.799289 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.913159 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.29082 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.957194 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.31599 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.926452 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.41277 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.896881 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.706031 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.910405 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 1.2004 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.94513 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.65966 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.4183 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.11433 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.20312 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.867886 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.58922 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.846561 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 0.716579 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.746677 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.14152 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.20217 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 1.35187 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.1747 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.04433 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 0.808782 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.01877 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.647673 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.48237 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.735443 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.777702 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.830874 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.948144 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.37373 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.886396 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.25544 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.05765 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.642445 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.966857 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.38021 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.9205 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.4592 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.962919 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.55586 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.899612 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.660075 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 0.989601 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.21152 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.742814 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.72392 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.60755 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.06873 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.06074 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.778436 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.58349 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.958092 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.542982 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.62905 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.01943 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.04528 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 1.6544 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.839012 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.30024 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 0.723744 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.1305 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.573503 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.50861 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.891646 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.864759 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.776335 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.884541 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.49051 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.2842 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.4381 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.954722 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.52773 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.839486 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.17802 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.977295 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.26125 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.97464 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.31091 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.907126 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.71607 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.01793 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.901494 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.755818 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.50929 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.26787 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.951568 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.01396 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.764738 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.53072 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.05094 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.780014 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.745417 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.884544 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.09178 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 1.31462 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.03262 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.25927 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.785173 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.02135 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.74215 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.19264 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.00479 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.86399 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.962462 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.865908 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.19318 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.2084 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.26391 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.14422 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.691675 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.87306 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.37141 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.938208 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.40137 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.907947 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.21593 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.933299 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.938703 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.645932 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.83579 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.28061 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.791152 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.09631 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.2437 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.884777 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 1.26597 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.928895 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.687172 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 0.62231 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.914684 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.71049 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.09575 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.3285 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.823598 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.857423 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.23013 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.28115 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.606071 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.62538 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.620699 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.934456 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.643439 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.44607 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.68018 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.817831 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.10335 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.63075 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.682263 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.970029 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.18807 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.943656 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.09833 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.90444 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.27421 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.01724 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.792876 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.910397 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 1.0512 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.991183 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.30573 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.27637 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.03632 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.14177 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.859448 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.43163 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.965829 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.869298 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.832498 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.14114 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.08877 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.37849 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.09275 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.1526 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.904137 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.09271 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.707875 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.19121 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.839404 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.936164 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.950214 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.845724 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.29843 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.16381 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.977112 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.06813 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.632052 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.717771 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.28712 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.914354 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.37716 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.909467 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.34633 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.944082 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.780943 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.896811 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.1152 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.01978 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.14748 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.22314 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.01671 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.96321 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.11804 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.25071 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.932109 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.734231 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.816487 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.41214 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.1066 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 1.46432 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.823489 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.05296 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 0.959247 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.25391 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.625112 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.54994 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.817408 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.916763 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.873401 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.17661 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.52205 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.998372 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.19888 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.850465 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.629431 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.929897 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.41767 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.897581 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.37423 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.989646 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.44151 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.03573 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.05173 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.827841 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.64095 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.47573 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.890911 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.48865 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.27979 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.1506 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.13293 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.689954 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.644467 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 0.562999 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.839725 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.5016 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.0656 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 1.02019 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.815277 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.868879 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.03231 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.36154 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.623884 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.93874 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.655227 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.978569 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.67126 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.1729 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.52095 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.746954 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.06585 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.68993 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.483004 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.921711 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.22822 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.954663 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.13533 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.909101 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.30262 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.897899 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.771767 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.0368 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 1.0114 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.962826 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.41325 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.41347 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.06397 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.04076 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.850889 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.33827 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.02473 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.822261 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.812544 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.06911 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.990078 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.4747 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.946273 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.21548 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.790066 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.981393 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.738832 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.13174 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.02782 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.920114 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.993891 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.962839 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.35342 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.26418 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.23642 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.1365 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.741351 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.926527 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.1492 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.962512 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.23474 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.982826 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.18248 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.988159 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.792313 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.951056 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.04324 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.956454 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.1474 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.27947 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.946197 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.01658 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.951375 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.20993 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.963925 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 0.807172 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.913887 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.0909 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.02166 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 1.22993 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.975177 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.09577 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 0.902922 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.12997 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.852405 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.21871 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.986585 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.94292 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.886469 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.89647 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.1324 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.06239 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.1377 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.90379 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.724319 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.886063 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.267001042249513 z_k1_s0000
                                 - 0.0660002434773683 z_k1_s0001
                                 - 0.174000183711788 z_k1_s0002
                                 - 0.178000694833009 z_k1_s0003
                                 - 0.149000454213793 z_k1_s0004
                                 - 0.0920002897561936 z_k1_s0005
                                 - 0.141999874752534 z_k1_s0006
                                 - 0.192000593752026 z_k1_s0007
                                 - 0.0559999260521904 z_k1_s0010
                                 - 0.0970001803329987 z_k1_s0011
                                 - 0.0359999746906751 z_k1_s0012
                                 - 0.134000273181648 z_k1_s0013
                                 - 0.118999667811052 z_k1_s0014
                                 - 0.0809998458041009 z_k1_s0015
                                 - 0.137000238655898 z_k1_s0016
                                 - 0.11799993865275 z_k1_s0017
                                 - 0.0529999574571447 z_k1_s0020
                                 - 0.0190000091355951 z_k1_s0021
                                 - 0.0160000047983941 z_k1_s0022
                                 - 0.0170000154108001 z_k1_s0023
                                 - 0.0419999507207467 z_k1_s0024
                                 - 0.0529999574571447 z_k1_s0025
                                 - 0.0410000501084081 z_k1_s0026
                                 - 0.0339999208215328 z_k1_s0027
                                 - 0.0310000383016261 z_k1_s0030
                                 - 0.0529999029541754 z_k1_s0031
                                 - 0.0570001723095779 z_k1_s0032
                                 - 0.0430001695669275 z_k1_s0033
                                 - 0.0970002338600794 z_k1_s0034
                                 - 0.0340000335179011 z_k1_s0035
                                 - 0.0270000321465759 z_k1_s0036
                                 - 0.0340000335179011 z_k1_s0037
                                 - 0.063999990056788 z_k1_s0040
                                 - 0.0400001617856344 z_k1_s0041
                                 - 0.0220000217820421 z_k1_s0042
                                 - 0.0570000961444155 z_k1_s0043
                                 - 0.0570000961444155 z_k1_s0044
                                 - 0.0170000015588378 z_k1_s0045
                                 - 0.0489999293871751 z_k1_s0046
                                 - 0.0300000205391406 z_k1_s0047
                                 - 0.045000043538962 z_k1_s0050
                                 - 0.0890002012038946 z_k1_s0051
                                 - 0.053000226132813 z_k1_s0052
                                 - 0.0310000477001999 z_k1_s0053
                                 - 0.0590001389778709 z_k1_s0054
                                 - 0.0430000725906094 z_k1_s0055
                                 - 0.114000336059037 z_k1_s0056
                                 - 0.0640000663487525 z_k1_s0057
                                 - 0.0939999803869793 z_k1_s0060
                                 - 0.127000719588635 z_k1_s0061
                                 - 0.074000172339182 z_k1_s0062
                                 - 0.0440001450666532 z_k1_s0063
                                 - 0.153000060289689 z_k1_s0064
                                 - 0.0840002339634971 z_k1_s0065
                                 - 0.170000417211691 z_k1_s0066
                                 - 0.0420001169817486 z_k1_s0067
                                 - 0.0620002952551876 z_k1_s0070
                                 - 0.0360000290253878 z_k1_s0071
                                 - 0.0430000441025601 z_k1_s0072
                                 - 0.106000066596948 z_k1_s0073
                                 - 0.104999943157177 z_k1_s0074
                                 - 0.0480000953005985 z_k1_s0075
                                 - 0.0719998316504496 z_k1_s0076
                                 - 0.0940002833221447 z_k1_s0077
                                 - 0.324000530913015 z_k1_s0100
                                 - 0.114000136624933 z_k1_s0101
                                 - 0.263000640031315 z_k1_s0102
                                 - 0.307000043093348 z_k1_s0103
                                 - 0.500000401161917 z_k1_s0104
                                 - 0.15700014854485 z_k1_s0105
                                 - 0.394000812875761 z_k1_s0106
                                 - 0.19900009192242 z_k1_s0107
                                 - 0.0929999646731017 z_k1_s0110
                                 - 0.202999843869259 z_k1_s0111
                                 - 0.192999353796977 z_k1_s0112
                                 - 0.214000076748827 z_k1_s0113
                                 - 0.104999940359957 z_k1_s0114
                                 - 0.344000323689665 z_k1_s0115
                                 - 0.289000690291527 z_k1_s0116
                                 - 0.0899999707513878 z_k1_s0117
                                 - 0.242998830189228 z_k1_s0120
                                 - 0.185999835588203 z_k1_s0121
                                 - 0.151999690432554 z_k1_s0122
                                 - 0.369999352138298 z_k1_s0123
                                 - 0.386000068445879 z_k1_s0124
                                 - 0.321999498212157 z_k1_s0125
                                 - 0.401000166235086 z_k1_s0126
                                 - 0.235000767032039 z_k1_s0127
                                 - 0.112999866795548 z_k1_s0130
                                 - 0.218999936063814 z_k1_s0131
                                 - 0.143999338987561 z_k1_s0132
                                 - 0.185000181788916 z_k1_s0133
                                 - 0.105999940168349 z_k1_s0134
                                 - 0.156000467319102 z_k1_s0135
                                 - 0.27199932519836 z_k1_s0136
                                 - 0.095999989658099 z_k1_s0137
                                 - 0.313999614116971 z_k1_s0140
                                 - 0.281999271033922 z_k1_s0141
                                 - 0.434999148344276 z_k1_s0142
                                 - 0.184999616994738 z_k1_s0143
                                 - 0.445999106001304 z_k1_s0144
                                 - 0.414999499603619 z_k1_s0145
                                 - 0.634998652744173 z_k1_s0146
                                 - 0.304999923033341 z_k1_s0147
                                 - 0.0819998879740671 z_k1_s0150
                                 - 0.115999859553545 z_k1_s0151
                                 - 0.0849999289486946 z_k1_s0152
                                 - 0.162000487831167 z_k1_s0153
                                 - 0.217999774291979 z_k1_s0154
                                 - 0.0879999699233222 z_k1_s0155
                                 - 0.168000323380605 z_k1_s0156
                                 - 0.312999223823222 z_k1_s0157
                                 - 0.332999308480445 z_k1_s0160
                                 - 0.402002036847557 z_k1_s0161
                                 - 0.23500042000621 z_k1_s0162
                                 - 0.652000265253553 z_k1_s0163
                                 - 0.245000647043103 z_k1_s0164
                                 - 0.523000345274229 z_k1_s0165
                                 - 0.205000334696837 z_k1_s0166
                                 - 0.383999900357359 z_k1_s0167
                                 - 0.248999665633519 z_k1_s0170
                                 - 0.178999811106376 z_k1_s0171
                                 - 0.271999642430985 z_k1_s0172
                                 - 0.219000028593025 z_k1_s0173
                                 - 0.24499958649491 z_k1_s0174
                                 - 0.0959999863284744 z_k1_s0175
                                 - 0.266000480222148 z_k1_s0176
                                 - 0.386999526418316 z_k1_s0177
                                 - 0.218999471402541 z_k1_s1000
                                 - 0.855997992844472 z_k1_s1001
                                 - 0.239999392786424 z_k1_s1002
                                 - 0.414999983581984 z_k1_s1003
                                 - 0.702996672452877 z_k1_s1004
                                 - 0.342999961866624 z_k1_s1005
                                 - 0.44499744413714 z_k1_s1006
                                 - 0.176999628634775 z_k1_s1007
                                 - 0.241999730844782 z_k1_s1010
                                 - 0.365000957161491 z_k1_s1011
                                 - 0.238999683251706 z_k1_s1012
                                 - 0.773998504024729 z_k1_s1013
                                 - 0.313999785579485 z_k1_s1014
                                 - 0.345999809906019 z_k1_s1015
                                 - 0.576998037077255 z_k1_s1016
                                 - 0.767999133837995 z_k1_s1017
                                 - 0.581998100834375 z_k1_s1020
                                 - 0.570001907930229 z_k1_s1021
                                 - 0.764998927117219 z_k1_s1022
                                 - 0.52200089231513 z_k1_s1023
                                 - 0.44599928425789 z_k1_s1024
                                 - 0.588999264194818 z_k1_s1025
                                 - 0.352000138044728 z_k1_s1026
                                 - 0.23499989603272 z_k1_s1027
                                 - 0.17299974838428 z_k1_s1030
                                 - 0.527999636732989 z_k1_s1031
                                 - 0.55399882330044 z_k1_s1032
                                 - 0.184999687091918 z_k1_s1033
                                 - 0.354998429850641 z_k1_s1034
                                 - 0.254999718485959 z_k1_s1035
                                 - 0.29300059395873 z_k1_s1036
                                 - 0.574000665810595 z_k1_s1037
                                 - 0.391999142484066 z_k1_s1040
                                 - 0.352000237413283 z_k1_s1041
                                 - 0.124999928042937 z_k1_s1042
                                 - 0.127999845923246 z_k1_s1043
                                 - 0.0879998830534923 z_k1_s1044
                                 - 0.222999889963782 z_k1_s1045
                                 - 0.196999191269146 z_k1_s1046
                                 - 0.257998931900717 z_k1_s1047
                                 - 0.35499792573924 z_k1_s1050
                                 - 0.502997363883153 z_k1_s1051
                                 - 0.436999794136851 z_k1_s1052
                                 - 0.632997193278879 z_k1_s1053
                                 - 0.368000298271743 z_k1_s1054
                                 - 0.298997815415471 z_k1_s1055
                                 - 0.150999572068023 z_k1_s1056
                                 - 0.240999408003661 z_k1_s1057
                                 - 0.113999950703749 z_k1_s1060
                                 - 0.311000766887991 z_k1_s1061
                                 - 0.163000029285151 z_k1_s1062
                                 - 0.456999128074032 z_k1_s1063
                                 - 0.186000002137668 z_k1_s1064
                                 - 0.149999976420709 z_k1_s1065
                                 - 0.163999942843986 z_k1_s1066
                                 - 0.416000515062567 z_k1_s1067
                                 - 0.437998940302467 z_k1_s1070
                                 - 0.255998983281989 z_k1_s1071
                                 - 0.167999819403583 z_k1_s1072
                                 - 0.109999870847589 z_k1_s1073
                                 - 0.226999810503626 z_k1_s1074
                                 - 0.349998860424479 z_k1_s1075
                                 - 0.290000887779448 z_k1_s1076
                                 - 0.450999264375209 z_k1_s1077
                                 - 0.334000371260991 z_k1_s1100
                                 - 0.237000118376084 z_k1_s1101
                                 - 0.339000366457104 z_k1_s1102
                                 - 1.09700102250913 z_k1_s1103
                                 - 0.248000280847686 z_k1_s1104
                                 - 0.585001168346802 z_k1_s1105
                                 - 0.326000292427133 z_k1_s1106
                                 - 1.16000061589985 z_k1_s1107
                                 - 1.2920021558346 z_k1_s1110
                                 - 0.887998400704701 z_k1_s1111
                                 - 1.56000100939195 z_k1_s1112
                                 - 1.48099317503625 z_k1_s1113
                                 - 1.93599473106453 z_k1_s1114
                                 - 0.408998770104688 z_k1_s1115
                                 - 1.42599653612411 z_k1_s1116
                                 - 1.10100263394558 z_k1_s1117
                                 - 1.35099543823055 z_k1_s1120
                                 - 0.445998240144118 z_k1_s1121
                                 - 0.647997452199044 z_k1_s1122
                                 - 0.496997823072164 z_k1_s1123
                                 - 0.47299821872293 z_k1_s1124
                                 - 1.07099666466877 z_k1_s1125
                                 - 1.69599102180857 z_k1_s1126
                                 - 0.596997869270998 z_k1_s1127
                                 - 0.781998696744734 z_k1_s1130
                                 - 0.841999087236744 z_k1_s1131
                                 - 0.374999987574628 z_k1_s1132
                                 - 0.369999880700281 z_k1_s1133
                                 - 0.604999105793581 z_k1_s1134
                                 - 0.72000022590333 z_k1_s1135
                                 - 0.286000226011624 z_k1_s1136
                                 - 0.479998663935293 z_k1_s1137
                                 - 0.860000992745204 z_k1_s1140
                                 - 0.936001463531655 z_k1_s1141
                                 - 0.500999655516324 z_k1_s1142
                                 - 1.32700221088183 z_k1_s1143
                                 - 1.04299808731347 z_k1_s1144
                                 - 0.399999364810353 z_k1_s1145
                                 - 1.95499624975741 z_k1_s1146
                                 - 0.46499994478019 z_k1_s1147
                                 - 0.597000533051707 z_k1_s1150
                                 - 0.361000127385248 z_k1_s1151
                                 - 1.18900039597024 z_k1_s1152
                                 - 0.289000412573704 z_k1_s1153
                                 - 0.608998356568964 z_k1_s1154
                                 - 0.648998986631427 z_k1_s1155
                                 - 0.396000087314533 z_k1_s1156
                                 - 0.640998860618935 z_k1_s1157
                                 - 0.680000418433724 z_k1_s1160
                                 - 1.05300261429238 z_k1_s1161
                                 - 0.614000192650436 z_k1_s1162
                                 - 2.34800380795664 z_k1_s1163
                                 - 0.503999816344956 z_k1_s1164
                                 - 0.595000366129509 z_k1_s1165
                                 - 1.01599979633032 z_k1_s1166
                                 - 0.685000333333965 z_k1_s1167
                                 - 0.633999938783932 z_k1_s1170
                                 - 0.565999151166036 z_k1_s1171
                                 - 1.3749983176098 z_k1_s1172
                                 - 0.847999889065595 z_k1_s1173
                                 - 2.53200235760336 z_k1_s1174
                                 - 1.00699929260331 z_k1_s1175
                                 - 1.67499757839766 z_k1_s1176
                                 - 1.59699408635546 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.13148 w_k1_ADM_s000
                                 + 0.951499 w_k1_AXP_s000
                                 + 0.807685 w_k1_DIS_s000
                                 + 0.999052 w_k1_FDX_s000
                                 + 0.737256 w_k1_IBM_s000
                                 + 0.953295 w_k1_INTC_s000
                                 + 0.806354 w_k1_KO_s000
                                 + 0.63728 w_k1_MAT_s000
                                 + 0.975949 w_k1_MSFT_s000
                                 + 1.00041 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.860791 w_k1_ADM_s000
                                 + 0.623433 w_k1_AXP_s000
                                 + 0.545592 w_k1_DIS_s000
                                 + 0.573352 w_k1_FDX_s000
                                 + 0.883131 w_k1_IBM_s000
                                 + 0.739812 w_k1_INTC_s000
                                 + 0.819027 w_k1_KO_s000
                                 + 0.529354 w_k1_MAT_s000
                                 + 0.731249 w_k1_MSFT_s000
                                 + 0.717665 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.04075 w_k1_ADM_s000
                                 + 1.2665 w_k1_AXP_s000 + 1.32575 w_k1_DIS_s000
                                 + 1.20564 w_k1_FDX_s000
                                 + 0.91834 w_k1_IBM_s000
                                 + 1.4161 w_k1_INTC_s000 + 1.09339 w_k1_KO_s000
                                 + 0.858154 w_k1_MAT_s000
                                 + 1.20366 w_k1_MSFT_s000
                                 + 0.988066 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 0.947793 w_k1_ADM_s000
                                 + 1.01805 w_k1_AXP_s000
                                 + 1.04969 w_k1_DIS_s000
                                 + 0.929195 w_k1_FDX_s000
                                 + 1.67416 w_k1_IBM_s000
                                 + 1.11464 w_k1_INTC_s000
                                 + 1.30509 w_k1_KO_s000 + 1.40792 w_k1_MAT_s000
                                 + 1.14909 w_k1_MSFT_s000
                                 + 1.07883 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 1.43501 w_k1_ADM_s000
                                 + 1.58055 w_k1_AXP_s000
                                 + 1.52778 w_k1_DIS_s000
                                 + 1.35284 w_k1_FDX_s000
                                 + 1.22042 w_k1_IBM_s000
                                 + 1.28419 w_k1_INTC_s000 + 1.3405 w_k1_KO_s000
                                 + 0.912184 w_k1_MAT_s000
                                 + 1.09374 w_k1_MSFT_s000
                                 + 1.28027 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.30237 w_k1_ADM_s000
                                 + 1.49552 w_k1_AXP_s000
                                 + 0.863359 w_k1_DIS_s000
                                 + 0.991965 w_k1_FDX_s000
                                 + 1.11702 w_k1_IBM_s000
                                 + 0.998191 w_k1_INTC_s000
                                 + 0.883344 w_k1_KO_s000
                                 + 1.13145 w_k1_MAT_s000
                                 + 0.85776 w_k1_MSFT_s000
                                 + 1.03456 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.988365 w_k1_ADM_s000
                                 + 0.702143 w_k1_AXP_s000
                                 + 0.72418 w_k1_DIS_s000
                                 + 0.796729 w_k1_FDX_s000
                                 + 0.625173 w_k1_IBM_s000
                                 + 0.741609 w_k1_INTC_s000
                                 + 0.679814 w_k1_KO_s000
                                 + 0.553649 w_k1_MAT_s000
                                 + 0.852082 w_k1_MSFT_s000
                                 + 0.84777 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.787455 w_k1_ADM_s000
                                 + 0.73243 w_k1_AXP_s000
                                 + 0.645273 w_k1_DIS_s000
                                 + 0.782789 w_k1_FDX_s000
                                 + 1.08847 w_k1_IBM_s000
                                 + 0.875769 w_k1_INTC_s000
                                 + 0.887485 w_k1_KO_s000
                                 + 0.975719 w_k1_MAT_s000
                                 + 0.789787 w_k1_MSFT_s000
                                 + 0.890195 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.42803 w_k1_ADM_s001
                                 + 1.42914 w_k1_AXP_s001
                                 + 0.939403 w_k1_DIS_s001
                                 + 1.08708 w_k1_FDX_s001
                                 + 1.22946 w_k1_IBM_s001
                                 + 1.11675 w_k1_INTC_s001
                                 + 0.930319 w_k1_KO_s001
                                 + 1.20101 w_k1_MAT_s001
                                 + 0.955085 w_k1_MSFT_s001
                                 + 1.07785 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.13987 w_k1_ADM_s001
                                 + 1.43724 w_k1_AXP_s001 + 1.6316 w_k1_DIS_s001
                                 + 1.37704 w_k1_FDX_s001
                                 + 1.20094 w_k1_IBM_s001
                                 + 1.79457 w_k1_INTC_s001
                                 + 1.27246 w_k1_KO_s001
                                 + 0.824234 w_k1_MAT_s001
                                 + 1.5036 w_k1_MSFT_s001 + 1.19111 w_k1_PG_s001
                                 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.820088 w_k1_ADM_s001
                                 + 1.10709 w_k1_AXP_s001
                                 + 0.988031 w_k1_DIS_s001
                                 + 1.2367 w_k1_FDX_s001
                                 + 0.614246 w_k1_IBM_s001
                                 + 1.49131 w_k1_INTC_s001
                                 + 0.660414 w_k1_KO_s001
                                 + 0.490816 w_k1_MAT_s001
                                 + 1.20966 w_k1_MSFT_s001
                                 + 0.875689 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.02198 w_k1_ADM_s001
                                 + 0.825751 w_k1_AXP_s001
                                 + 1.04216 w_k1_DIS_s001
                                 + 0.88902 w_k1_FDX_s001 + 1.1151 w_k1_IBM_s001
                                 + 1.13955 w_k1_INTC_s001
                                 + 1.02547 w_k1_KO_s001
                                 + 0.760281 w_k1_MAT_s001
                                 + 1.08144 w_k1_MSFT_s001
                                 + 0.927835 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.993826 w_k1_ADM_s001
                                 + 0.992575 w_k1_AXP_s001
                                 + 0.706575 w_k1_DIS_s001
                                 + 0.824066 w_k1_FDX_s001
                                 + 1.06539 w_k1_IBM_s001
                                 + 0.998775 w_k1_INTC_s001
                                 + 0.869433 w_k1_KO_s001
                                 + 0.85858 w_k1_MAT_s001
                                 + 0.979839 w_k1_MSFT_s001
                                 + 0.964768 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.22073 w_k1_ADM_s001
                                 + 1.11186 w_k1_AXP_s001
                                 + 1.26059 w_k1_DIS_s001
                                 + 1.21073 w_k1_FDX_s001
                                 + 0.931241 w_k1_IBM_s001
                                 + 0.92704 w_k1_INTC_s001
                                 + 0.924305 w_k1_KO_s001
                                 + 0.927179 w_k1_MAT_s001
                                 + 0.932165 w_k1_MSFT_s001
                                 + 1.0169 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.11285 w_k1_ADM_s001
                                 + 0.87755 w_k1_AXP_s001
                                 + 1.11211 w_k1_DIS_s001
                                 + 1.03936 w_k1_FDX_s001
                                 + 1.11149 w_k1_IBM_s001
                                 + 1.15578 w_k1_INTC_s001
                                 + 1.00332 w_k1_KO_s001
                                 + 0.911336 w_k1_MAT_s001
                                 + 0.993727 w_k1_MSFT_s001
                                 + 1.00788 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.95086 w_k1_ADM_s001
                                 + 0.733577 w_k1_AXP_s001
                                 + 0.532493 w_k1_DIS_s001
                                 + 0.790393 w_k1_FDX_s001
                                 + 0.764087 w_k1_IBM_s001
                                 + 0.940843 w_k1_INTC_s001
                                 + 0.696521 w_k1_KO_s001
                                 + 0.605787 w_k1_MAT_s001
                                 + 0.78231 w_k1_MSFT_s001
                                 + 0.76308 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.10504 w_k1_ADM_s002
                                 + 0.731357 w_k1_AXP_s002
                                 + 0.604979 w_k1_DIS_s002
                                 + 0.769449 w_k1_FDX_s002
                                 + 1.25994 w_k1_IBM_s002
                                 + 0.913138 w_k1_INTC_s002
                                 + 1.14209 w_k1_KO_s002 + 1.08682 w_k1_MAT_s002
                                 + 0.720631 w_k1_MSFT_s002
                                 + 0.966827 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.97436 w_k1_ADM_s002
                                 + 1.49834 w_k1_AXP_s002
                                 + 1.60386 w_k1_DIS_s002 + 1.5653 w_k1_FDX_s002
                                 + 1.06167 w_k1_IBM_s002
                                 + 1.24431 w_k1_INTC_s002
                                 + 1.49168 w_k1_KO_s002 + 1.5845 w_k1_MAT_s002
                                 + 1.34515 w_k1_MSFT_s002
                                 + 1.21941 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.65804 w_k1_ADM_s002
                                 + 1.76977 w_k1_AXP_s002
                                 + 1.60484 w_k1_DIS_s002
                                 + 1.40986 w_k1_FDX_s002
                                 + 1.50367 w_k1_IBM_s002
                                 + 1.40918 w_k1_INTC_s002
                                 + 1.77783 w_k1_KO_s002 + 2.31404 w_k1_MAT_s002
                                 + 1.34769 w_k1_MSFT_s002
                                 + 1.33326 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.74188 w_k1_ADM_s002
                                 + 1.52743 w_k1_AXP_s002
                                 + 1.34382 w_k1_DIS_s002
                                 + 1.28427 w_k1_FDX_s002 + 1.5956 w_k1_IBM_s002
                                 + 1.07299 w_k1_INTC_s002
                                 + 1.76566 w_k1_KO_s002 + 2.39836 w_k1_MAT_s002
                                 + 1.19236 w_k1_MSFT_s002
                                 + 1.29248 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.18457 w_k1_ADM_s002
                                 + 1.21323 w_k1_AXP_s002
                                 + 1.19812 w_k1_DIS_s002
                                 + 1.16125 w_k1_FDX_s002
                                 + 1.01682 w_k1_IBM_s002
                                 + 1.09012 w_k1_INTC_s002
                                 + 1.27696 w_k1_KO_s002 + 1.09666 w_k1_MAT_s002
                                 + 0.992002 w_k1_MSFT_s002
                                 + 1.03331 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.958687 w_k1_ADM_s002
                                 + 0.515718 w_k1_AXP_s002
                                 + 0.568364 w_k1_DIS_s002
                                 + 0.704958 w_k1_FDX_s002
                                 + 0.809047 w_k1_IBM_s002
                                 + 0.676181 w_k1_INTC_s002
                                 + 0.876215 w_k1_KO_s002
                                 + 0.855274 w_k1_MAT_s002
                                 + 0.744169 w_k1_MSFT_s002
                                 + 0.800887 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.21374 w_k1_ADM_s002
                                 + 1.10069 w_k1_AXP_s002
                                 + 0.912947 w_k1_DIS_s002
                                 + 1.09419 w_k1_FDX_s002
                                 + 0.850923 w_k1_IBM_s002
                                 + 1.20223 w_k1_INTC_s002 + 1.1023 w_k1_KO_s002
                                 + 0.782459 w_k1_MAT_s002
                                 + 1.03328 w_k1_MSFT_s002
                                 + 1.10824 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.24464 w_k1_ADM_s002
                                 + 1.21348 w_k1_AXP_s002
                                 + 0.921689 w_k1_DIS_s002
                                 + 1.16824 w_k1_FDX_s002
                                 + 1.17335 w_k1_IBM_s002
                                 + 0.882294 w_k1_INTC_s002
                                 + 1.09999 w_k1_KO_s002 + 1.71355 w_k1_MAT_s002
                                 + 0.930618 w_k1_MSFT_s002
                                 + 1.16769 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.943852 w_k1_ADM_s003
                                 + 0.785264 w_k1_AXP_s003
                                 + 0.669505 w_k1_DIS_s003
                                 + 0.814958 w_k1_FDX_s003
                                 + 1.09639 w_k1_IBM_s003
                                 + 0.841955 w_k1_INTC_s003
                                 + 0.930973 w_k1_KO_s003
                                 + 0.93066 w_k1_MAT_s003
                                 + 0.723948 w_k1_MSFT_s003
                                 + 0.982853 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.43683 w_k1_ADM_s003
                                 + 1.08696 w_k1_AXP_s003
                                 + 0.964703 w_k1_DIS_s003
                                 + 1.0978 w_k1_FDX_s003
                                 + 0.752714 w_k1_IBM_s003
                                 + 0.928871 w_k1_INTC_s003
                                 + 0.786958 w_k1_KO_s003
                                 + 0.657398 w_k1_MAT_s003
                                 + 0.901706 w_k1_MSFT_s003
                                 + 1.01227 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.904812 w_k1_ADM_s003
                                 + 0.875688 w_k1_AXP_s003
                                 + 0.915444 w_k1_DIS_s003
                                 + 0.849544 w_k1_FDX_s003
                                 + 1.39503 w_k1_IBM_s003
                                 + 0.912455 w_k1_INTC_s003
                                 + 1.03478 w_k1_KO_s003 + 1.31769 w_k1_MAT_s003
                                 + 0.848416 w_k1_MSFT_s003
                                 + 0.891429 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.93276 w_k1_ADM_s003
                                 + 0.993268 w_k1_AXP_s003
                                 + 1.10533 w_k1_DIS_s003
                                 + 1.15152 w_k1_FDX_s003
                                 + 0.820395 w_k1_IBM_s003
                                 + 0.808809 w_k1_INTC_s003
                                 + 0.8629 w_k1_KO_s003 + 0.69326 w_k1_MAT_s003
                                 + 0.805006 w_k1_MSFT_s003
                                 + 0.951336 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.61427 w_k1_ADM_s003
                                 + 1.42736 w_k1_AXP_s003
                                 + 1.40385 w_k1_DIS_s003
                                 + 1.23802 w_k1_FDX_s003
                                 + 1.42298 w_k1_IBM_s003
                                 + 1.35088 w_k1_INTC_s003 + 1.277 w_k1_KO_s003
                                 + 1.54094 w_k1_MAT_s003
                                 + 1.01392 w_k1_MSFT_s003
                                 + 1.12444 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.01099 w_k1_ADM_s003
                                 + 0.849982 w_k1_AXP_s003
                                 + 0.825052 w_k1_DIS_s003
                                 + 0.937047 w_k1_FDX_s003
                                 + 1.00669 w_k1_IBM_s003
                                 + 0.817982 w_k1_INTC_s003
                                 + 0.957753 w_k1_KO_s003
                                 + 0.921313 w_k1_MAT_s003
                                 + 0.760694 w_k1_MSFT_s003
                                 + 0.97885 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.18687 w_k1_ADM_s003
                                 + 0.79582 w_k1_AXP_s003
                                 + 0.573394 w_k1_DIS_s003
                                 + 0.759927 w_k1_FDX_s003
                                 + 0.806354 w_k1_IBM_s003
                                 + 0.544924 w_k1_INTC_s003
                                 + 0.71065 w_k1_KO_s003
                                 + 0.872118 w_k1_MAT_s003
                                 + 0.60163 w_k1_MSFT_s003
                                 + 0.754503 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.18106 w_k1_ADM_s003
                                 + 1.08895 w_k1_AXP_s003
                                 + 0.99644 w_k1_DIS_s003
                                 + 0.966403 w_k1_FDX_s003
                                 + 0.951758 w_k1_IBM_s003
                                 + 0.83088 w_k1_INTC_s003
                                 + 0.878427 w_k1_KO_s003
                                 + 0.857964 w_k1_MAT_s003
                                 + 0.828451 w_k1_MSFT_s003
                                 + 1.01416 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.897546 w_k1_ADM_s004
                                 + 0.974743 w_k1_AXP_s004
                                 + 0.953279 w_k1_DIS_s004
                                 + 1.10279 w_k1_FDX_s004
                                 + 0.982999 w_k1_IBM_s004
                                 + 1.02999 w_k1_INTC_s004
                                 + 0.922124 w_k1_KO_s004
                                 + 1.87239 w_k1_MAT_s004
                                 + 1.02597 w_k1_MSFT_s004
                                 + 1.04713 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.813391 w_k1_ADM_s004
                                 + 0.695304 w_k1_AXP_s004
                                 + 0.687775 w_k1_DIS_s004
                                 + 0.871667 w_k1_FDX_s004
                                 + 0.809761 w_k1_IBM_s004
                                 + 1.03474 w_k1_INTC_s004
                                 + 0.827626 w_k1_KO_s004
                                 + 0.867366 w_k1_MAT_s004
                                 + 0.929086 w_k1_MSFT_s004
                                 + 0.923373 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.867324 w_k1_ADM_s004
                                 + 0.83118 w_k1_AXP_s004
                                 + 0.883093 w_k1_DIS_s004
                                 + 0.869808 w_k1_FDX_s004
                                 + 1.30842 w_k1_IBM_s004
                                 + 1.04003 w_k1_INTC_s004
                                 + 1.01813 w_k1_KO_s004 + 1.39137 w_k1_MAT_s004
                                 + 0.955169 w_k1_MSFT_s004
                                 + 0.902367 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.28692 w_k1_ADM_s004
                                 + 0.945655 w_k1_AXP_s004
                                 + 1.30418 w_k1_DIS_s004 + 1.0707 w_k1_FDX_s004
                                 + 1.10185 w_k1_IBM_s004
                                 + 1.06598 w_k1_INTC_s004
                                 + 1.15033 w_k1_KO_s004
                                 + 0.966063 w_k1_MAT_s004
                                 + 1.10122 w_k1_MSFT_s004
                                 + 1.03383 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.963995 w_k1_ADM_s004
                                 + 1.27966 w_k1_AXP_s004
                                 + 1.62437 w_k1_DIS_s004
                                 + 1.22341 w_k1_FDX_s004
                                 + 1.45071 w_k1_IBM_s004
                                 + 1.11511 w_k1_INTC_s004 + 1.3541 w_k1_KO_s004
                                 + 1.47962 w_k1_MAT_s004
                                 + 1.17307 w_k1_MSFT_s004 + 1.3506 w_k1_PG_s004
                                 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.943194 w_k1_ADM_s004
                                 + 0.781066 w_k1_AXP_s004
                                 + 0.567474 w_k1_DIS_s004
                                 + 0.805745 w_k1_FDX_s004
                                 + 0.612728 w_k1_IBM_s004
                                 + 0.913484 w_k1_INTC_s004
                                 + 0.597351 w_k1_KO_s004
                                 + 0.632671 w_k1_MAT_s004
                                 + 0.810254 w_k1_MSFT_s004
                                 + 0.835161 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.29311 w_k1_ADM_s004
                                 + 1.27381 w_k1_AXP_s004
                                 + 1.00898 w_k1_DIS_s004
                                 + 1.07595 w_k1_FDX_s004 + 1.3921 w_k1_IBM_s004
                                 + 1.37644 w_k1_INTC_s004
                                 + 1.04459 w_k1_KO_s004 + 1.03008 w_k1_MAT_s004
                                 + 0.955448 w_k1_MSFT_s004
                                 + 0.988996 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.798436 w_k1_ADM_s004
                                 + 0.659181 w_k1_AXP_s004
                                 + 0.663741 w_k1_DIS_s004
                                 + 0.688022 w_k1_FDX_s004
                                 + 1.01972 w_k1_IBM_s004
                                 + 0.774127 w_k1_INTC_s004
                                 + 0.859244 w_k1_KO_s004
                                 + 0.822138 w_k1_MAT_s004
                                 + 0.66936 w_k1_MSFT_s004
                                 + 0.719057 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.608209 w_k1_ADM_s005
                                 + 0.648333 w_k1_AXP_s005
                                 + 0.742873 w_k1_DIS_s005
                                 + 0.768065 w_k1_FDX_s005
                                 + 1.10133 w_k1_IBM_s005
                                 + 0.903305 w_k1_INTC_s005
                                 + 0.89876 w_k1_KO_s005
                                 + 0.789443 w_k1_MAT_s005
                                 + 0.746187 w_k1_MSFT_s005
                                 + 0.743862 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.04743 w_k1_ADM_s005
                                 + 0.888303 w_k1_AXP_s005
                                 + 0.680443 w_k1_DIS_s005
                                 + 0.868154 w_k1_FDX_s005
                                 + 1.00587 w_k1_IBM_s005
                                 + 1.20249 w_k1_INTC_s005
                                 + 1.07642 w_k1_KO_s005
                                 + 0.866332 w_k1_MAT_s005
                                 + 0.900529 w_k1_MSFT_s005
                                 + 0.80361 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.51684 w_k1_ADM_s005
                                 + 1.44642 w_k1_AXP_s005
                                 + 1.05335 w_k1_DIS_s005
                                 + 1.30412 w_k1_FDX_s005
                                 + 1.30269 w_k1_IBM_s005
                                 + 1.08246 w_k1_INTC_s005
                                 + 1.21717 w_k1_KO_s005 + 1.1701 w_k1_MAT_s005
                                 + 0.968564 w_k1_MSFT_s005
                                 + 1.03876 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.970888 w_k1_ADM_s005
                                 + 1.06539 w_k1_AXP_s005
                                 + 0.759955 w_k1_DIS_s005
                                 + 0.970525 w_k1_FDX_s005
                                 + 1.57889 w_k1_IBM_s005
                                 + 0.911475 w_k1_INTC_s005
                                 + 1.04238 w_k1_KO_s005 + 1.75103 w_k1_MAT_s005
                                 + 0.778502 w_k1_MSFT_s005
                                 + 0.908178 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.855946 w_k1_ADM_s005
                                 + 1.11299 w_k1_AXP_s005
                                 + 1.22172 w_k1_DIS_s005
                                 + 1.03498 w_k1_FDX_s005
                                 + 1.39932 w_k1_IBM_s005
                                 + 1.08846 w_k1_INTC_s005
                                 + 1.23577 w_k1_KO_s005 + 1.21565 w_k1_MAT_s005
                                 + 0.935545 w_k1_MSFT_s005
                                 + 0.857364 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.916423 w_k1_ADM_s005
                                 + 1.06482 w_k1_AXP_s005
                                 + 1.03212 w_k1_DIS_s005
                                 + 1.20926 w_k1_FDX_s005
                                 + 1.05219 w_k1_IBM_s005
                                 + 1.12869 w_k1_INTC_s005
                                 + 0.88188 w_k1_KO_s005 + 1.04582 w_k1_MAT_s005
                                 + 0.932247 w_k1_MSFT_s005
                                 + 0.824026 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.819519 w_k1_ADM_s005
                                 + 0.692406 w_k1_AXP_s005
                                 + 0.755149 w_k1_DIS_s005
                                 + 0.835735 w_k1_FDX_s005
                                 + 0.711327 w_k1_IBM_s005
                                 + 0.865181 w_k1_INTC_s005
                                 + 0.818777 w_k1_KO_s005
                                 + 0.644665 w_k1_MAT_s005
                                 + 0.763689 w_k1_MSFT_s005
                                 + 0.756289 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.912436 w_k1_ADM_s005
                                 + 1.16059 w_k1_AXP_s005
                                 + 1.53836 w_k1_DIS_s005
                                 + 1.19615 w_k1_FDX_s005
                                 + 1.21149 w_k1_IBM_s005
                                 + 1.11878 w_k1_INTC_s005
                                 + 1.63881 w_k1_KO_s005 + 1.33824 w_k1_MAT_s005
                                 + 1.13688 w_k1_MSFT_s005
                                 + 1.01429 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.67825 w_k1_ADM_s006
                                 + 0.730688 w_k1_AXP_s006
                                 + 0.68763 w_k1_DIS_s006
                                 + 0.737425 w_k1_FDX_s006
                                 + 1.03957 w_k1_IBM_s006
                                 + 0.859166 w_k1_INTC_s006
                                 + 0.984429 w_k1_KO_s006
                                 + 1.18511 w_k1_MAT_s006
                                 + 0.858371 w_k1_MSFT_s006
                                 + 0.984474 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.758704 w_k1_ADM_s006
                                 + 0.735147 w_k1_AXP_s006
                                 + 0.853895 w_k1_DIS_s006
                                 + 0.872524 w_k1_FDX_s006
                                 + 0.805714 w_k1_IBM_s006
                                 + 0.981838 w_k1_INTC_s006
                                 + 1.0188 w_k1_KO_s006 + 0.832841 w_k1_MAT_s006
                                 + 1.01322 w_k1_MSFT_s006
                                 + 1.01667 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.905168 w_k1_ADM_s006
                                 + 0.648276 w_k1_AXP_s006
                                 + 0.731813 w_k1_DIS_s006
                                 + 0.855415 w_k1_FDX_s006
                                 + 0.748043 w_k1_IBM_s006
                                 + 0.819151 w_k1_INTC_s006
                                 + 0.927906 w_k1_KO_s006
                                 + 0.700101 w_k1_MAT_s006
                                 + 0.764302 w_k1_MSFT_s006
                                 + 0.946485 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.22948 w_k1_ADM_s006
                                 + 0.744951 w_k1_AXP_s006
                                 + 0.843081 w_k1_DIS_s006
                                 + 1.01084 w_k1_FDX_s006
                                 + 0.598911 w_k1_IBM_s006
                                 + 1.03369 w_k1_INTC_s006
                                 + 0.986566 w_k1_KO_s006
                                 + 0.614461 w_k1_MAT_s006
                                 + 0.919561 w_k1_MSFT_s006
                                 + 1.03261 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.33158 w_k1_ADM_s006
                                 + 1.33109 w_k1_AXP_s006
                                 + 1.03592 w_k1_DIS_s006
                                 + 1.11432 w_k1_FDX_s006
                                 + 0.924667 w_k1_IBM_s006
                                 + 0.876679 w_k1_INTC_s006
                                 + 1.16469 w_k1_KO_s006 + 1.3589 w_k1_MAT_s006
                                 + 0.954289 w_k1_MSFT_s006
                                 + 1.1187 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.08194 w_k1_ADM_s006
                                 + 1.61778 w_k1_AXP_s006
                                 + 1.04321 w_k1_DIS_s006 + 1.0476 w_k1_FDX_s006
                                 + 1.63527 w_k1_IBM_s006
                                 + 1.03665 w_k1_INTC_s006
                                 + 1.28325 w_k1_KO_s006 + 1.69349 w_k1_MAT_s006
                                 + 0.997381 w_k1_MSFT_s006
                                 + 1.15031 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.898535 w_k1_ADM_s006
                                 + 1.15447 w_k1_AXP_s006
                                 + 1.29097 w_k1_DIS_s006
                                 + 1.03067 w_k1_FDX_s006
                                 + 1.12825 w_k1_IBM_s006 + 1.134 w_k1_INTC_s006
                                 + 1.19165 w_k1_KO_s006
                                 + 0.950554 w_k1_MAT_s006
                                 + 1.11596 w_k1_MSFT_s006
                                 + 1.13272 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.44841 w_k1_ADM_s006
                                 + 1.14606 w_k1_AXP_s006
                                 + 1.66835 w_k1_DIS_s006
                                 + 1.36739 w_k1_FDX_s006
                                 + 1.20329 w_k1_IBM_s006
                                 + 0.998247 w_k1_INTC_s006
                                 + 1.65656 w_k1_KO_s006 + 1.52963 w_k1_MAT_s006
                                 + 1.27598 w_k1_MSFT_s006
                                 + 1.29196 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.19348 w_k1_ADM_s007
                                 + 1.02512 w_k1_AXP_s007
                                 + 1.49054 w_k1_DIS_s007
                                 + 1.46531 w_k1_FDX_s007
                                 + 1.06066 w_k1_IBM_s007
                                 + 1.07861 w_k1_INTC_s007
                                 + 1.22199 w_k1_KO_s007
                                 + 0.852862 w_k1_MAT_s007
                                 + 1.30256 w_k1_MSFT_s007
                                 + 1.10113 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.95718 w_k1_ADM_s007
                                 + 1.08062 w_k1_AXP_s007
                                 + 1.15781 w_k1_DIS_s007
                                 + 1.20126 w_k1_FDX_s007
                                 + 0.881642 w_k1_IBM_s007
                                 + 1.11219 w_k1_INTC_s007
                                 + 0.960584 w_k1_KO_s007
                                 + 0.721273 w_k1_MAT_s007
                                 + 1.05172 w_k1_MSFT_s007
                                 + 0.931638 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.777108 w_k1_ADM_s007
                                 + 0.960095 w_k1_AXP_s007
                                 + 0.673577 w_k1_DIS_s007
                                 + 0.916826 w_k1_FDX_s007
                                 + 0.980629 w_k1_IBM_s007
                                 + 0.841805 w_k1_INTC_s007
                                 + 0.848538 w_k1_KO_s007
                                 + 0.966175 w_k1_MAT_s007
                                 + 0.850772 w_k1_MSFT_s007
                                 + 0.825767 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.925288 w_k1_ADM_s007
                                 + 0.876297 w_k1_AXP_s007
                                 + 0.940147 w_k1_DIS_s007
                                 + 1.00107 w_k1_FDX_s007
                                 + 1.01177 w_k1_IBM_s007
                                 + 0.816652 w_k1_INTC_s007
                                 + 0.931118 w_k1_KO_s007
                                 + 0.790632 w_k1_MAT_s007
                                 + 1.0305 w_k1_MSFT_s007
                                 + 0.900645 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.10643 w_k1_ADM_s007
                                 + 1.05216 w_k1_AXP_s007
                                 + 0.87782 w_k1_DIS_s007
                                 + 1.01878 w_k1_FDX_s007
                                 + 0.765791 w_k1_IBM_s007
                                 + 1.09164 w_k1_INTC_s007
                                 + 0.918937 w_k1_KO_s007
                                 + 0.793636 w_k1_MAT_s007
                                 + 1.11942 w_k1_MSFT_s007
                                 + 0.95037 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.758874 w_k1_ADM_s007
                                 + 0.886252 w_k1_AXP_s007
                                 + 0.942183 w_k1_DIS_s007
                                 + 0.99288 w_k1_FDX_s007
                                 + 1.26147 w_k1_IBM_s007
                                 + 0.84299 w_k1_INTC_s007
                                 + 1.14156 w_k1_KO_s007 + 1.25498 w_k1_MAT_s007
                                 + 0.970318 w_k1_MSFT_s007
                                 + 1.12645 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.83415 w_k1_ADM_s007
                                 + 0.674521 w_k1_AXP_s007
                                 + 0.711434 w_k1_DIS_s007
                                 + 0.835027 w_k1_FDX_s007
                                 + 0.755847 w_k1_IBM_s007
                                 + 0.753745 w_k1_INTC_s007
                                 + 0.744357 w_k1_KO_s007
                                 + 0.625705 w_k1_MAT_s007
                                 + 0.862779 w_k1_MSFT_s007
                                 + 0.819811 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.03486 w_k1_ADM_s007
                                 + 1.4161 w_k1_AXP_s007 + 1.29076 w_k1_DIS_s007
                                 + 1.20831 w_k1_FDX_s007
                                 + 1.37315 w_k1_IBM_s007
                                 + 1.13305 w_k1_INTC_s007
                                 + 1.17843 w_k1_KO_s007 + 1.38643 w_k1_MAT_s007
                                 + 1.24131 w_k1_MSFT_s007
                                 + 0.921957 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.04588 w_k1_ADM_s010
                                 + 0.799932 w_k1_AXP_s010
                                 + 0.739079 w_k1_DIS_s010
                                 + 1.00298 w_k1_FDX_s010
                                 + 0.698165 w_k1_IBM_s010
                                 + 0.888517 w_k1_INTC_s010
                                 + 0.7837 w_k1_KO_s010 + 0.659476 w_k1_MAT_s010
                                 + 0.908635 w_k1_MSFT_s010
                                 + 0.880047 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 0.963723 w_k1_ADM_s010
                                 + 1.47942 w_k1_AXP_s010
                                 + 1.76753 w_k1_DIS_s010 + 1.4092 w_k1_FDX_s010
                                 + 1.19356 w_k1_IBM_s010
                                 + 1.21488 w_k1_INTC_s010
                                 + 1.16663 w_k1_KO_s010 + 1.42429 w_k1_MAT_s010
                                 + 1.38284 w_k1_MSFT_s010 + 1.156 w_k1_PG_s010
                                 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.868711 w_k1_ADM_s010
                                 + 0.780463 w_k1_AXP_s010
                                 + 0.993664 w_k1_DIS_s010
                                 + 0.957423 w_k1_FDX_s010
                                 + 0.930634 w_k1_IBM_s010
                                 + 1.15422 w_k1_INTC_s010
                                 + 1.07157 w_k1_KO_s010
                                 + 0.601687 w_k1_MAT_s010
                                 + 1.0529 w_k1_MSFT_s010 + 1.03765 w_k1_PG_s010
                                 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 1.11825 w_k1_ADM_s010
                                 + 1.20328 w_k1_AXP_s010
                                 + 1.31752 w_k1_DIS_s010
                                 + 1.14571 w_k1_FDX_s010
                                 + 1.42228 w_k1_IBM_s010
                                 + 1.14133 w_k1_INTC_s010
                                 + 1.25808 w_k1_KO_s010 + 1.38844 w_k1_MAT_s010
                                 + 1.14253 w_k1_MSFT_s010
                                 + 1.09466 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.831932 w_k1_ADM_s010
                                 + 0.891987 w_k1_AXP_s010
                                 + 0.735358 w_k1_DIS_s010
                                 + 0.785982 w_k1_FDX_s010
                                 + 1.05769 w_k1_IBM_s010
                                 + 0.936378 w_k1_INTC_s010
                                 + 0.83869 w_k1_KO_s010
                                 + 0.712363 w_k1_MAT_s010
                                 + 0.762791 w_k1_MSFT_s010
                                 + 0.863752 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.36716 w_k1_ADM_s010
                                 + 1.32404 w_k1_AXP_s010
                                 + 1.44573 w_k1_DIS_s010
                                 + 1.43669 w_k1_FDX_s010
                                 + 0.95729 w_k1_IBM_s010
                                 + 1.20784 w_k1_INTC_s010
                                 + 0.954633 w_k1_KO_s010
                                 + 1.01249 w_k1_MAT_s010
                                 + 1.36978 w_k1_MSFT_s010
                                 + 1.09506 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.05362 w_k1_ADM_s010
                                 + 0.931757 w_k1_AXP_s010
                                 + 1.07783 w_k1_DIS_s010 + 1.133 w_k1_FDX_s010
                                 + 0.888826 w_k1_IBM_s010
                                 + 0.984969 w_k1_INTC_s010
                                 + 0.923713 w_k1_KO_s010
                                 + 0.714214 w_k1_MAT_s010
                                 + 1.03311 w_k1_MSFT_s010
                                 + 1.01534 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.12713 w_k1_ADM_s010
                                 + 1.46134 w_k1_AXP_s010
                                 + 0.980977 w_k1_DIS_s010
                                 + 1.14829 w_k1_FDX_s010 + 1.1605 w_k1_IBM_s010
                                 + 1.19252 w_k1_INTC_s010
                                 + 0.884972 w_k1_KO_s010
                                 + 1.33279 w_k1_MAT_s010
                                 + 0.934396 w_k1_MSFT_s010
                                 + 0.918983 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.37516 w_k1_ADM_s011
                                 + 1.7479 w_k1_AXP_s011 + 1.53179 w_k1_DIS_s011
                                 + 1.24186 w_k1_FDX_s011
                                 + 1.74995 w_k1_IBM_s011
                                 + 1.05701 w_k1_INTC_s011 + 1.3598 w_k1_KO_s011
                                 + 2.98429 w_k1_MAT_s011
                                 + 1.12234 w_k1_MSFT_s011
                                 + 1.25492 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.929424 w_k1_ADM_s011
                                 + 0.869903 w_k1_AXP_s011
                                 + 1.07606 w_k1_DIS_s011
                                 + 0.924389 w_k1_FDX_s011
                                 + 0.96559 w_k1_IBM_s011
                                 + 1.10331 w_k1_INTC_s011
                                 + 0.982276 w_k1_KO_s011
                                 + 1.13845 w_k1_MAT_s011
                                 + 1.17885 w_k1_MSFT_s011
                                 + 0.862915 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.14418 w_k1_ADM_s011
                                 + 0.810155 w_k1_AXP_s011
                                 + 0.877208 w_k1_DIS_s011
                                 + 0.927203 w_k1_FDX_s011
                                 + 0.642953 w_k1_IBM_s011
                                 + 0.747532 w_k1_INTC_s011
                                 + 0.822078 w_k1_KO_s011
                                 + 0.97092 w_k1_MAT_s011
                                 + 1.04479 w_k1_MSFT_s011
                                 + 0.889572 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.18393 w_k1_ADM_s011
                                 + 1.35836 w_k1_AXP_s011
                                 + 1.60376 w_k1_DIS_s011
                                 + 1.32228 w_k1_FDX_s011
                                 + 1.03969 w_k1_IBM_s011
                                 + 1.30307 w_k1_INTC_s011
                                 + 1.13211 w_k1_KO_s011 + 1.71194 w_k1_MAT_s011
                                 + 1.27466 w_k1_MSFT_s011
                                 + 1.21233 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.03853 w_k1_ADM_s011
                                 + 1.25203 w_k1_AXP_s011
                                 + 1.17733 w_k1_DIS_s011
                                 + 1.14836 w_k1_FDX_s011
                                 + 1.44901 w_k1_IBM_s011
                                 + 0.979782 w_k1_INTC_s011
                                 + 1.23638 w_k1_KO_s011 + 2.14354 w_k1_MAT_s011
                                 + 1.04327 w_k1_MSFT_s011
                                 + 1.19039 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.866126 w_k1_ADM_s011
                                 + 0.716334 w_k1_AXP_s011
                                 + 0.639737 w_k1_DIS_s011
                                 + 0.747167 w_k1_FDX_s011
                                 + 0.738338 w_k1_IBM_s011
                                 + 0.922942 w_k1_INTC_s011
                                 + 0.69283 w_k1_KO_s011 + 1.1004 w_k1_MAT_s011
                                 + 0.820193 w_k1_MSFT_s011
                                 + 0.797479 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.864082 w_k1_ADM_s011
                                 + 1.08218 w_k1_AXP_s011
                                 + 0.839776 w_k1_DIS_s011
                                 + 0.946364 w_k1_FDX_s011
                                 + 1.18904 w_k1_IBM_s011
                                 + 1.03068 w_k1_INTC_s011
                                 + 0.988066 w_k1_KO_s011
                                 + 1.50708 w_k1_MAT_s011
                                 + 0.964851 w_k1_MSFT_s011
                                 + 0.934799 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.12404 w_k1_ADM_s011
                                 + 1.19688 w_k1_AXP_s011
                                 + 1.08621 w_k1_DIS_s011
                                 + 1.15594 w_k1_FDX_s011
                                 + 0.731841 w_k1_IBM_s011
                                 + 1.03316 w_k1_INTC_s011
                                 + 0.772584 w_k1_KO_s011
                                 + 1.24459 w_k1_MAT_s011
                                 + 1.11284 w_k1_MSFT_s011
                                 + 0.867373 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.783236 w_k1_ADM_s012
                                 + 0.74029 w_k1_AXP_s012
                                 + 0.694698 w_k1_DIS_s012
                                 + 0.844441 w_k1_FDX_s012
                                 + 0.687361 w_k1_IBM_s012
                                 + 0.963407 w_k1_INTC_s012
                                 + 0.74374 w_k1_KO_s012
                                 + 0.697549 w_k1_MAT_s012
                                 + 0.944295 w_k1_MSFT_s012
                                 + 0.84939 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.2782 w_k1_ADM_s012
                                 + 1.42597 w_k1_AXP_s012
                                 + 1.82359 w_k1_DIS_s012
                                 + 1.44576 w_k1_FDX_s012
                                 + 1.31817 w_k1_IBM_s012
                                 + 1.24494 w_k1_INTC_s012
                                 + 1.13845 w_k1_KO_s012 + 1.10231 w_k1_MAT_s012
                                 + 1.36025 w_k1_MSFT_s012
                                 + 1.24277 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.956108 w_k1_ADM_s012
                                 + 1.24761 w_k1_AXP_s012
                                 + 1.16995 w_k1_DIS_s012
                                 + 0.921815 w_k1_FDX_s012
                                 + 1.78849 w_k1_IBM_s012
                                 + 0.97053 w_k1_INTC_s012
                                 + 1.52626 w_k1_KO_s012 + 2.05762 w_k1_MAT_s012
                                 + 1.02083 w_k1_MSFT_s012
                                 + 1.17637 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.924266 w_k1_ADM_s012
                                 + 1.13006 w_k1_AXP_s012
                                 + 0.911596 w_k1_DIS_s012
                                 + 0.912325 w_k1_FDX_s012
                                 + 1.23114 w_k1_IBM_s012
                                 + 0.932672 w_k1_INTC_s012
                                 + 1.03236 w_k1_KO_s012 + 1.43518 w_k1_MAT_s012
                                 + 0.940429 w_k1_MSFT_s012
                                 + 1.06923 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.02245 w_k1_ADM_s012
                                 + 1.13472 w_k1_AXP_s012
                                 + 1.27189 w_k1_DIS_s012
                                 + 1.14998 w_k1_FDX_s012
                                 + 1.03595 w_k1_IBM_s012
                                 + 1.11502 w_k1_INTC_s012
                                 + 0.982105 w_k1_KO_s012
                                 + 0.727639 w_k1_MAT_s012
                                 + 1.17325 w_k1_MSFT_s012
                                 + 1.08043 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.949522 w_k1_ADM_s012
                                 + 0.915417 w_k1_AXP_s012
                                 + 0.693558 w_k1_DIS_s012
                                 + 0.761743 w_k1_FDX_s012
                                 + 0.880779 w_k1_IBM_s012
                                 + 1.00222 w_k1_INTC_s012
                                 + 0.815236 w_k1_KO_s012
                                 + 0.506575 w_k1_MAT_s012
                                 + 0.831632 w_k1_MSFT_s012
                                 + 0.84576 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.09332 w_k1_ADM_s012
                                 + 1.0217 w_k1_AXP_s012
                                 + 0.908231 w_k1_DIS_s012
                                 + 1.01058 w_k1_FDX_s012
                                 + 0.836377 w_k1_IBM_s012
                                 + 1.03738 w_k1_INTC_s012
                                 + 0.840295 w_k1_KO_s012
                                 + 1.02297 w_k1_MAT_s012
                                 + 1.15951 w_k1_MSFT_s012
                                 + 0.931455 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.18368 w_k1_ADM_s012
                                 + 1.57024 w_k1_AXP_s012
                                 + 1.22944 w_k1_DIS_s012
                                 + 1.29782 w_k1_FDX_s012
                                 + 1.14539 w_k1_IBM_s012
                                 + 1.40549 w_k1_INTC_s012
                                 + 1.05407 w_k1_KO_s012 + 1.29646 w_k1_MAT_s012
                                 + 1.25516 w_k1_MSFT_s012
                                 + 1.18917 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.98537 w_k1_ADM_s013
                                 + 1.18323 w_k1_AXP_s013
                                 + 1.12197 w_k1_DIS_s013
                                 + 1.07221 w_k1_FDX_s013
                                 + 0.794733 w_k1_IBM_s013
                                 + 1.05444 w_k1_INTC_s013
                                 + 0.811406 w_k1_KO_s013
                                 + 0.914384 w_k1_MAT_s013
                                 + 0.974957 w_k1_MSFT_s013
                                 + 1.08211 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.00938 w_k1_ADM_s013
                                 + 1.32275 w_k1_AXP_s013
                                 + 1.10143 w_k1_DIS_s013
                                 + 1.09901 w_k1_FDX_s013
                                 + 1.10713 w_k1_IBM_s013
                                 + 1.03037 w_k1_INTC_s013
                                 + 0.961796 w_k1_KO_s013
                                 + 1.20016 w_k1_MAT_s013
                                 + 0.914289 w_k1_MSFT_s013
                                 + 1.04395 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.06082 w_k1_ADM_s013
                                 + 0.865085 w_k1_AXP_s013
                                 + 0.778833 w_k1_DIS_s013
                                 + 0.843253 w_k1_FDX_s013
                                 + 0.787917 w_k1_IBM_s013
                                 + 0.782795 w_k1_INTC_s013
                                 + 0.749942 w_k1_KO_s013
                                 + 0.622092 w_k1_MAT_s013
                                 + 0.753556 w_k1_MSFT_s013
                                 + 0.825015 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.713837 w_k1_ADM_s013
                                 + 0.642938 w_k1_AXP_s013
                                 + 0.63203 w_k1_DIS_s013
                                 + 0.682168 w_k1_FDX_s013
                                 + 0.916316 w_k1_IBM_s013
                                 + 0.828086 w_k1_INTC_s013
                                 + 0.803403 w_k1_KO_s013
                                 + 0.842775 w_k1_MAT_s013
                                 + 0.768614 w_k1_MSFT_s013
                                 + 0.817307 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.962173 w_k1_ADM_s013
                                 + 1.09912 w_k1_AXP_s013
                                 + 1.11761 w_k1_DIS_s013
                                 + 0.887963 w_k1_FDX_s013
                                 + 1.17111 w_k1_IBM_s013
                                 + 0.915047 w_k1_INTC_s013
                                 + 0.958439 w_k1_KO_s013 + 1.46 w_k1_MAT_s013
                                 + 0.875646 w_k1_MSFT_s013
                                 + 0.922278 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.26157 w_k1_ADM_s013
                                 + 1.78125 w_k1_AXP_s013
                                 + 1.72559 w_k1_DIS_s013
                                 + 1.36102 w_k1_FDX_s013
                                 + 1.20201 w_k1_IBM_s013
                                 + 1.14441 w_k1_INTC_s013
                                 + 1.28315 w_k1_KO_s013 + 1.87522 w_k1_MAT_s013
                                 + 1.20239 w_k1_MSFT_s013
                                 + 1.21723 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.853257 w_k1_ADM_s013
                                 + 0.87218 w_k1_AXP_s013
                                 + 1.29051 w_k1_DIS_s013
                                 + 1.04926 w_k1_FDX_s013
                                 + 0.91494 w_k1_IBM_s013
                                 + 1.10289 w_k1_INTC_s013
                                 + 1.02348 w_k1_KO_s013
                                 + 0.734997 w_k1_MAT_s013
                                 + 1.14894 w_k1_MSFT_s013
                                 + 1.07607 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.8165 w_k1_ADM_s013
                                 + 1.02964 w_k1_AXP_s013
                                 + 1.26031 w_k1_DIS_s013
                                 + 1.06677 w_k1_FDX_s013
                                 + 1.47912 w_k1_IBM_s013
                                 + 1.12559 w_k1_INTC_s013 + 1.3168 w_k1_KO_s013
                                 + 1.60916 w_k1_MAT_s013
                                 + 1.03444 w_k1_MSFT_s013 + 1.0455 w_k1_PG_s013
                                 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.756046 w_k1_ADM_s014
                                 + 0.710366 w_k1_AXP_s014
                                 + 0.644672 w_k1_DIS_s014
                                 + 0.865258 w_k1_FDX_s014
                                 + 0.660584 w_k1_IBM_s014
                                 + 0.841308 w_k1_INTC_s014
                                 + 0.752791 w_k1_KO_s014
                                 + 0.563835 w_k1_MAT_s014
                                 + 0.875747 w_k1_MSFT_s014
                                 + 0.779598 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.761995 w_k1_ADM_s014
                                 + 0.808739 w_k1_AXP_s014
                                 + 0.497252 w_k1_DIS_s014
                                 + 0.698078 w_k1_FDX_s014
                                 + 0.789076 w_k1_IBM_s014
                                 + 0.669644 w_k1_INTC_s014
                                 + 0.737029 w_k1_KO_s014
                                 + 0.695492 w_k1_MAT_s014
                                 + 0.655877 w_k1_MSFT_s014
                                 + 0.733372 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.00252 w_k1_ADM_s014
                                 + 1.14181 w_k1_AXP_s014
                                 + 1.14999 w_k1_DIS_s014
                                 + 1.11624 w_k1_FDX_s014
                                 + 0.761317 w_k1_IBM_s014
                                 + 1.13241 w_k1_INTC_s014
                                 + 0.96781 w_k1_KO_s014
                                 + 0.679877 w_k1_MAT_s014
                                 + 1.07202 w_k1_MSFT_s014
                                 + 0.930308 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.665269 w_k1_ADM_s014
                                 + 0.555636 w_k1_AXP_s014
                                 + 0.645119 w_k1_DIS_s014
                                 + 0.666366 w_k1_FDX_s014
                                 + 0.962708 w_k1_IBM_s014
                                 + 0.680128 w_k1_INTC_s014
                                 + 0.887614 w_k1_KO_s014
                                 + 0.645617 w_k1_MAT_s014
                                 + 0.669789 w_k1_MSFT_s014
                                 + 0.737843 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.762554 w_k1_ADM_s014
                                 + 0.897964 w_k1_AXP_s014
                                 + 0.973384 w_k1_DIS_s014
                                 + 0.948399 w_k1_FDX_s014
                                 + 1.00339 w_k1_IBM_s014
                                 + 1.22016 w_k1_INTC_s014
                                 + 0.970941 w_k1_KO_s014
                                 + 0.59083 w_k1_MAT_s014
                                 + 1.02416 w_k1_MSFT_s014
                                 + 0.959619 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.17495 w_k1_ADM_s014
                                 + 1.32633 w_k1_AXP_s014
                                 + 1.13661 w_k1_DIS_s014
                                 + 1.26292 w_k1_FDX_s014
                                 + 1.00432 w_k1_IBM_s014
                                 + 1.04507 w_k1_INTC_s014
                                 + 1.10209 w_k1_KO_s014 + 1.10659 w_k1_MAT_s014
                                 + 1.06446 w_k1_MSFT_s014
                                 + 1.14004 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.958065 w_k1_ADM_s014
                                 + 1.31717 w_k1_AXP_s014
                                 + 1.63438 w_k1_DIS_s014
                                 + 1.27923 w_k1_FDX_s014
                                 + 1.33383 w_k1_IBM_s014
                                 + 1.59415 w_k1_INTC_s014
                                 + 1.40012 w_k1_KO_s014 + 1.20911 w_k1_MAT_s014
                                 + 1.33015 w_k1_MSFT_s014
                                 + 1.11592 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.980335 w_k1_ADM_s014
                                 + 0.889149 w_k1_AXP_s014
                                 + 0.833719 w_k1_DIS_s014
                                 + 0.840848 w_k1_FDX_s014
                                 + 1.26221 w_k1_IBM_s014
                                 + 0.779896 w_k1_INTC_s014
                                 + 1.27969 w_k1_KO_s014 + 1.2491 w_k1_MAT_s014
                                 + 0.814195 w_k1_MSFT_s014
                                 + 0.974664 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.17887 w_k1_ADM_s015
                                 + 1.58138 w_k1_AXP_s015
                                 + 1.48252 w_k1_DIS_s015
                                 + 1.30085 w_k1_FDX_s015
                                 + 1.80129 w_k1_IBM_s015
                                 + 1.06205 w_k1_INTC_s015
                                 + 1.20128 w_k1_KO_s015 + 2.00349 w_k1_MAT_s015
                                 + 1.02943 w_k1_MSFT_s015
                                 + 1.26932 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.10131 w_k1_ADM_s015
                                 + 1.59209 w_k1_AXP_s015
                                 + 1.34785 w_k1_DIS_s015
                                 + 1.29906 w_k1_FDX_s015
                                 + 1.23287 w_k1_IBM_s015
                                 + 1.08895 w_k1_INTC_s015
                                 + 1.15271 w_k1_KO_s015 + 1.54549 w_k1_MAT_s015
                                 + 1.03102 w_k1_MSFT_s015
                                 + 1.10599 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.847277 w_k1_ADM_s015
                                 + 1.01349 w_k1_AXP_s015
                                 + 1.03577 w_k1_DIS_s015
                                 + 0.969344 w_k1_FDX_s015
                                 + 1.45805 w_k1_IBM_s015
                                 + 0.771393 w_k1_INTC_s015
                                 + 1.03163 w_k1_KO_s015 + 1.68967 w_k1_MAT_s015
                                 + 0.899034 w_k1_MSFT_s015
                                 + 0.950077 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.43304 w_k1_ADM_s015
                                 + 1.14324 w_k1_AXP_s015
                                 + 1.36353 w_k1_DIS_s015
                                 + 1.21466 w_k1_FDX_s015
                                 + 1.05081 w_k1_IBM_s015
                                 + 1.18861 w_k1_INTC_s015
                                 + 1.08778 w_k1_KO_s015
                                 + 0.916522 w_k1_MAT_s015
                                 + 1.28691 w_k1_MSFT_s015
                                 + 1.14216 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.947209 w_k1_ADM_s015
                                 + 1.0556 w_k1_AXP_s015 + 1.05048 w_k1_DIS_s015
                                 + 0.97775 w_k1_FDX_s015
                                 + 1.09602 w_k1_IBM_s015
                                 + 0.947069 w_k1_INTC_s015
                                 + 1.08359 w_k1_KO_s015 + 1.28563 w_k1_MAT_s015
                                 + 0.93232 w_k1_MSFT_s015 + 1.0301 w_k1_PG_s015
                                 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.765904 w_k1_ADM_s015
                                 + 0.691609 w_k1_AXP_s015
                                 + 0.63979 w_k1_DIS_s015
                                 + 0.748106 w_k1_FDX_s015
                                 + 1.1261 w_k1_IBM_s015
                                 + 0.839698 w_k1_INTC_s015
                                 + 0.767083 w_k1_KO_s015
                                 + 1.12568 w_k1_MAT_s015
                                 + 0.740989 w_k1_MSFT_s015
                                 + 0.869984 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.646776 w_k1_ADM_s015
                                 + 0.632575 w_k1_AXP_s015
                                 + 0.642201 w_k1_DIS_s015
                                 + 0.748443 w_k1_FDX_s015
                                 + 0.771367 w_k1_IBM_s015
                                 + 0.818582 w_k1_INTC_s015
                                 + 0.56618 w_k1_KO_s015
                                 + 0.706624 w_k1_MAT_s015
                                 + 0.798606 w_k1_MSFT_s015
                                 + 0.718327 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.08118 w_k1_ADM_s015
                                 + 1.03071 w_k1_AXP_s015
                                 + 0.962405 w_k1_DIS_s015
                                 + 1.08177 w_k1_FDX_s015
                                 + 0.800902 w_k1_IBM_s015
                                 + 1.00413 w_k1_INTC_s015
                                 + 0.875671 w_k1_KO_s015
                                 + 0.850081 w_k1_MAT_s015
                                 + 0.907939 w_k1_MSFT_s015
                                 + 0.994614 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.950421 w_k1_ADM_s016
                                 + 0.859831 w_k1_AXP_s016
                                 + 0.849968 w_k1_DIS_s016
                                 + 0.829134 w_k1_FDX_s016
                                 + 1.29615 w_k1_IBM_s016
                                 + 0.841311 w_k1_INTC_s016
                                 + 0.983645 w_k1_KO_s016 + 1.0339 w_k1_MAT_s016
                                 + 0.935038 w_k1_MSFT_s016
                                 + 1.03695 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.04074 w_k1_ADM_s016
                                 + 1.08479 w_k1_AXP_s016 + 1.3578 w_k1_DIS_s016
                                 + 1.16752 w_k1_FDX_s016
                                 + 0.923487 w_k1_IBM_s016
                                 + 0.992939 w_k1_INTC_s016
                                 + 1.20363 w_k1_KO_s016 + 1.13381 w_k1_MAT_s016
                                 + 1.28033 w_k1_MSFT_s016
                                 + 1.08469 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.17521 w_k1_ADM_s016
                                 + 1.55687 w_k1_AXP_s016
                                 + 1.67401 w_k1_DIS_s016
                                 + 1.37426 w_k1_FDX_s016
                                 + 1.10711 w_k1_IBM_s016
                                 + 1.28347 w_k1_INTC_s016
                                 + 1.28073 w_k1_KO_s016 + 1.40455 w_k1_MAT_s016
                                 + 1.39634 w_k1_MSFT_s016
                                 + 1.26575 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.942218 w_k1_ADM_s016
                                 + 1.0032 w_k1_AXP_s016
                                 + 0.978196 w_k1_DIS_s016
                                 + 0.997865 w_k1_FDX_s016
                                 + 1.0932 w_k1_IBM_s016
                                 + 0.772785 w_k1_INTC_s016
                                 + 1.02535 w_k1_KO_s016 + 1.13283 w_k1_MAT_s016
                                 + 0.954885 w_k1_MSFT_s016
                                 + 0.982022 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.05071 w_k1_ADM_s016
                                 + 1.46841 w_k1_AXP_s016
                                 + 1.50599 w_k1_DIS_s016
                                 + 1.10545 w_k1_FDX_s016
                                 + 1.60901 w_k1_IBM_s016
                                 + 1.24202 w_k1_INTC_s016
                                 + 1.50873 w_k1_KO_s016 + 2.23102 w_k1_MAT_s016
                                 + 1.33251 w_k1_MSFT_s016
                                 + 1.24932 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.01836 w_k1_ADM_s016
                                 + 1.00253 w_k1_AXP_s016
                                 + 0.905896 w_k1_DIS_s016
                                 + 1.00622 w_k1_FDX_s016
                                 + 0.765502 w_k1_IBM_s016
                                 + 1.02694 w_k1_INTC_s016
                                 + 0.807337 w_k1_KO_s016
                                 + 0.956516 w_k1_MAT_s016
                                 + 1.07552 w_k1_MSFT_s016
                                 + 0.958911 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.28327 w_k1_ADM_s016
                                 + 1.51524 w_k1_AXP_s016
                                 + 1.17349 w_k1_DIS_s016
                                 + 1.21065 w_k1_FDX_s016
                                 + 1.08238 w_k1_IBM_s016
                                 + 1.01884 w_k1_INTC_s016
                                 + 1.19812 w_k1_KO_s016 + 1.56178 w_k1_MAT_s016
                                 + 1.28088 w_k1_MSFT_s016
                                 + 1.18305 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.904879 w_k1_ADM_s016
                                 + 0.836874 w_k1_AXP_s016
                                 + 0.711089 w_k1_DIS_s016
                                 + 0.769826 w_k1_FDX_s016
                                 + 0.747747 w_k1_IBM_s016
                                 + 1.00482 w_k1_INTC_s016
                                 + 0.77183 w_k1_KO_s016
                                 + 0.620466 w_k1_MAT_s016
                                 + 0.843326 w_k1_MSFT_s016
                                 + 0.84428 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.882199 w_k1_ADM_s017
                                 + 0.654046 w_k1_AXP_s017
                                 + 0.704179 w_k1_DIS_s017
                                 + 0.681271 w_k1_FDX_s017
                                 + 1.07911 w_k1_IBM_s017
                                 + 0.640851 w_k1_INTC_s017
                                 + 0.815853 w_k1_KO_s017
                                 + 1.08021 w_k1_MAT_s017
                                 + 0.609179 w_k1_MSFT_s017
                                 + 0.847538 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.79294 w_k1_ADM_s017
                                 + 1.01717 w_k1_AXP_s017
                                 + 1.25955 w_k1_DIS_s017
                                 + 0.99596 w_k1_FDX_s017
                                 + 1.27625 w_k1_IBM_s017
                                 + 0.955775 w_k1_INTC_s017
                                 + 1.23268 w_k1_KO_s017 + 1.74528 w_k1_MAT_s017
                                 + 1.06411 w_k1_MSFT_s017
                                 + 1.04111 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.547143 w_k1_ADM_s017
                                 + 0.648538 w_k1_AXP_s017
                                 + 0.646587 w_k1_DIS_s017
                                 + 0.764686 w_k1_FDX_s017
                                 + 0.694831 w_k1_IBM_s017
                                 + 0.949984 w_k1_INTC_s017
                                 + 0.688515 w_k1_KO_s017
                                 + 0.747854 w_k1_MAT_s017
                                 + 0.833658 w_k1_MSFT_s017
                                 + 0.750743 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.830223 w_k1_ADM_s017
                                 + 1.22941 w_k1_AXP_s017 + 1.2535 w_k1_DIS_s017
                                 + 1.24337 w_k1_FDX_s017
                                 + 0.936599 w_k1_IBM_s017
                                 + 1.24246 w_k1_INTC_s017
                                 + 1.05864 w_k1_KO_s017
                                 + 0.812168 w_k1_MAT_s017
                                 + 1.16102 w_k1_MSFT_s017
                                 + 1.12712 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.15275 w_k1_ADM_s017
                                 + 1.70408 w_k1_AXP_s017
                                 + 1.37082 w_k1_DIS_s017
                                 + 1.22801 w_k1_FDX_s017
                                 + 1.30215 w_k1_IBM_s017
                                 + 0.930047 w_k1_INTC_s017
                                 + 1.13692 w_k1_KO_s017 + 2.38899 w_k1_MAT_s017
                                 + 0.981457 w_k1_MSFT_s017
                                 + 1.12847 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.4238 w_k1_ADM_s017
                                 + 1.33568 w_k1_AXP_s017
                                 + 1.36371 w_k1_DIS_s017
                                 + 1.17046 w_k1_FDX_s017
                                 + 1.96698 w_k1_IBM_s017
                                 + 1.33113 w_k1_INTC_s017
                                 + 1.54919 w_k1_KO_s017 + 2.09672 w_k1_MAT_s017
                                 + 1.11903 w_k1_MSFT_s017 + 1.2649 w_k1_PG_s017
                                 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.847333 w_k1_ADM_s017
                                 + 1.22277 w_k1_AXP_s017
                                 + 0.899802 w_k1_DIS_s017
                                 + 0.931259 w_k1_FDX_s017
                                 + 0.999096 w_k1_IBM_s017
                                 + 0.858626 w_k1_INTC_s017
                                 + 0.989284 w_k1_KO_s017
                                 + 1.40421 w_k1_MAT_s017
                                 + 0.877017 w_k1_MSFT_s017
                                 + 1.03565 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 0.997447 w_k1_ADM_s017
                                 + 0.834689 w_k1_AXP_s017
                                 + 0.953026 w_k1_DIS_s017
                                 + 0.980763 w_k1_FDX_s017
                                 + 0.804639 w_k1_IBM_s017
                                 + 1.01697 w_k1_INTC_s017
                                 + 0.914698 w_k1_KO_s017
                                 + 0.716781 w_k1_MAT_s017
                                 + 1.06882 w_k1_MSFT_s017
                                 + 0.917004 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.857546 w_k1_ADM_s100
                                 + 0.84527 w_k1_AXP_s100
                                 + 0.89159 w_k1_DIS_s100
                                 + 0.984977 w_k1_FDX_s100
                                 + 0.925373 w_k1_IBM_s100
                                 + 0.933904 w_k1_INTC_s100
                                 + 1.01438 w_k1_KO_s100 + 1.09151 w_k1_MAT_s100
                                 + 0.965877 w_k1_MSFT_s100
                                 + 1.05411 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.27057 w_k1_ADM_s100
                                 + 1.3694 w_k1_AXP_s100 + 1.49945 w_k1_DIS_s100
                                 + 1.23462 w_k1_FDX_s100
                                 + 1.31435 w_k1_IBM_s100
                                 + 1.03378 w_k1_INTC_s100
                                 + 1.31505 w_k1_KO_s100 + 1.39381 w_k1_MAT_s100
                                 + 1.04026 w_k1_MSFT_s100
                                 + 1.20139 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.03245 w_k1_ADM_s100
                                 + 0.6924 w_k1_AXP_s100
                                 + 0.883494 w_k1_DIS_s100
                                 + 0.896151 w_k1_FDX_s100
                                 + 0.859246 w_k1_IBM_s100
                                 + 0.87403 w_k1_INTC_s100
                                 + 1.01258 w_k1_KO_s100 + 1.04134 w_k1_MAT_s100
                                 + 0.884863 w_k1_MSFT_s100
                                 + 0.980422 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.24197 w_k1_ADM_s100
                                 + 1.06206 w_k1_AXP_s100
                                 + 1.02431 w_k1_DIS_s100
                                 + 0.982871 w_k1_FDX_s100
                                 + 1.11975 w_k1_IBM_s100
                                 + 0.80219 w_k1_INTC_s100
                                 + 1.09239 w_k1_KO_s100 + 1.29939 w_k1_MAT_s100
                                 + 0.921678 w_k1_MSFT_s100
                                 + 1.07344 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.11257 w_k1_ADM_s100
                                 + 1.00316 w_k1_AXP_s100
                                 + 1.08326 w_k1_DIS_s100
                                 + 1.10852 w_k1_FDX_s100
                                 + 0.717274 w_k1_IBM_s100
                                 + 0.821956 w_k1_INTC_s100
                                 + 0.863261 w_k1_KO_s100
                                 + 1.12732 w_k1_MAT_s100
                                 + 0.929896 w_k1_MSFT_s100
                                 + 1.03176 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.826157 w_k1_ADM_s100
                                 + 0.701538 w_k1_AXP_s100
                                 + 0.667255 w_k1_DIS_s100
                                 + 0.754591 w_k1_FDX_s100
                                 + 0.709849 w_k1_IBM_s100
                                 + 0.681022 w_k1_INTC_s100
                                 + 0.736664 w_k1_KO_s100
                                 + 0.968007 w_k1_MAT_s100
                                 + 0.738362 w_k1_MSFT_s100
                                 + 0.783694 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.904353 w_k1_ADM_s100
                                 + 0.740564 w_k1_AXP_s100
                                 + 0.73119 w_k1_DIS_s100
                                 + 0.797703 w_k1_FDX_s100
                                 + 1.18366 w_k1_IBM_s100
                                 + 0.788593 w_k1_INTC_s100
                                 + 1.0863 w_k1_KO_s100 + 1.21615 w_k1_MAT_s100
                                 + 0.90126 w_k1_MSFT_s100
                                 + 0.950944 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.890817 w_k1_ADM_s100
                                 + 0.885015 w_k1_AXP_s100
                                 + 1.17046 w_k1_DIS_s100
                                 + 0.913554 w_k1_FDX_s100
                                 + 1.30334 w_k1_IBM_s100
                                 + 0.962218 w_k1_INTC_s100
                                 + 1.20157 w_k1_KO_s100 + 1.10217 w_k1_MAT_s100
                                 + 0.96892 w_k1_MSFT_s100
                                 + 1.06541 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.08757 w_k1_ADM_s101
                                 + 1.39211 w_k1_AXP_s101
                                 + 1.34682 w_k1_DIS_s101
                                 + 1.12122 w_k1_FDX_s101
                                 + 1.54501 w_k1_IBM_s101
                                 + 1.18976 w_k1_INTC_s101
                                 + 1.36968 w_k1_KO_s101 + 1.33381 w_k1_MAT_s101
                                 + 1.11605 w_k1_MSFT_s101
                                 + 1.27738 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.897807 w_k1_ADM_s101
                                 + 0.929018 w_k1_AXP_s101
                                 + 0.761677 w_k1_DIS_s101
                                 + 0.716129 w_k1_FDX_s101
                                 + 0.88401 w_k1_IBM_s101
                                 + 0.739297 w_k1_INTC_s101
                                 + 0.927025 w_k1_KO_s101
                                 + 0.708939 w_k1_MAT_s101
                                 + 0.770185 w_k1_MSFT_s101
                                 + 0.87844 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.2242 w_k1_ADM_s101
                                 + 1.20401 w_k1_AXP_s101
                                 + 0.900039 w_k1_DIS_s101
                                 + 1.07566 w_k1_FDX_s101
                                 + 0.644138 w_k1_IBM_s101
                                 + 0.965551 w_k1_INTC_s101
                                 + 0.801622 w_k1_KO_s101
                                 + 0.672896 w_k1_MAT_s101
                                 + 1.0587 w_k1_MSFT_s101
                                 + 0.964927 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.09719 w_k1_ADM_s101
                                 + 0.979106 w_k1_AXP_s101
                                 + 1.04678 w_k1_DIS_s101
                                 + 0.935867 w_k1_FDX_s101
                                 + 1.08527 w_k1_IBM_s101
                                 + 0.875564 w_k1_INTC_s101
                                 + 1.02464 w_k1_KO_s101
                                 + 0.975021 w_k1_MAT_s101
                                 + 0.956837 w_k1_MSFT_s101
                                 + 0.963231 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.05718 w_k1_ADM_s101
                                 + 1.14424 w_k1_AXP_s101 + 1.072 w_k1_DIS_s101
                                 + 0.908786 w_k1_FDX_s101
                                 + 1.34772 w_k1_IBM_s101
                                 + 1.00004 w_k1_INTC_s101
                                 + 1.08393 w_k1_KO_s101 + 1.67523 w_k1_MAT_s101
                                 + 1.03914 w_k1_MSFT_s101
                                 + 1.01422 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.889227 w_k1_ADM_s101
                                 + 0.657997 w_k1_AXP_s101
                                 + 0.768267 w_k1_DIS_s101
                                 + 0.768453 w_k1_FDX_s101
                                 + 0.709339 w_k1_IBM_s101
                                 + 0.753122 w_k1_INTC_s101
                                 + 0.841359 w_k1_KO_s101
                                 + 0.705964 w_k1_MAT_s101
                                 + 0.858065 w_k1_MSFT_s101
                                 + 0.789199 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.07451 w_k1_ADM_s101
                                 + 1.03791 w_k1_AXP_s101
                                 + 0.921533 w_k1_DIS_s101
                                 + 0.98924 w_k1_FDX_s101
                                 + 0.795059 w_k1_IBM_s101
                                 + 1.01549 w_k1_INTC_s101
                                 + 0.914395 w_k1_KO_s101
                                 + 0.750555 w_k1_MAT_s101
                                 + 0.990719 w_k1_MSFT_s101
                                 + 0.874491 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.601 w_k1_ADM_s101
                                 + 1.54609 w_k1_AXP_s101
                                 + 1.54911 w_k1_DIS_s101
                                 + 1.25649 w_k1_FDX_s101
                                 + 1.05723 w_k1_IBM_s101
                                 + 1.09741 w_k1_INTC_s101 + 1.1853 w_k1_KO_s101
                                 + 1.0976 w_k1_MAT_s101
                                 + 1.25846 w_k1_MSFT_s101
                                 + 1.07935 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.37636 w_k1_ADM_s102
                                 + 1.30722 w_k1_AXP_s102
                                 + 0.977933 w_k1_DIS_s102
                                 + 0.955868 w_k1_FDX_s102
                                 + 1.53571 w_k1_IBM_s102
                                 + 0.827089 w_k1_INTC_s102
                                 + 1.04259 w_k1_KO_s102 + 1.55976 w_k1_MAT_s102
                                 + 0.849828 w_k1_MSFT_s102
                                 + 1.09221 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.45964 w_k1_ADM_s102
                                 + 1.13565 w_k1_AXP_s102 + 1.241 w_k1_DIS_s102
                                 + 1.22221 w_k1_FDX_s102
                                 + 0.930303 w_k1_IBM_s102
                                 + 0.948862 w_k1_INTC_s102
                                 + 1.04056 w_k1_KO_s102
                                 + 0.990726 w_k1_MAT_s102
                                 + 1.14004 w_k1_MSFT_s102
                                 + 1.11612 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.43008 w_k1_ADM_s102
                                 + 1.1454 w_k1_AXP_s102 + 1.28 w_k1_DIS_s102
                                 + 1.06352 w_k1_FDX_s102
                                 + 1.17456 w_k1_IBM_s102
                                 + 1.02927 w_k1_INTC_s102
                                 + 1.06224 w_k1_KO_s102 + 1.07048 w_k1_MAT_s102
                                 + 1.12469 w_k1_MSFT_s102
                                 + 1.16686 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.16082 w_k1_ADM_s102
                                 + 1.0703 w_k1_AXP_s102
                                 + 0.937425 w_k1_DIS_s102
                                 + 0.952507 w_k1_FDX_s102
                                 + 0.658592 w_k1_IBM_s102
                                 + 0.826131 w_k1_INTC_s102
                                 + 0.770526 w_k1_KO_s102
                                 + 0.923251 w_k1_MAT_s102
                                 + 0.937265 w_k1_MSFT_s102
                                 + 0.939296 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.868085 w_k1_ADM_s102
                                 + 0.730458 w_k1_AXP_s102
                                 + 0.692985 w_k1_DIS_s102
                                 + 0.724523 w_k1_FDX_s102
                                 + 0.6767 w_k1_IBM_s102
                                 + 0.613947 w_k1_INTC_s102
                                 + 0.665508 w_k1_KO_s102
                                 + 0.652974 w_k1_MAT_s102
                                 + 0.737734 w_k1_MSFT_s102
                                 + 0.780353 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.05351 w_k1_ADM_s102
                                 + 0.865907 w_k1_AXP_s102
                                 + 0.873602 w_k1_DIS_s102
                                 + 0.889073 w_k1_FDX_s102
                                 + 0.872238 w_k1_IBM_s102
                                 + 0.841819 w_k1_INTC_s102
                                 + 0.900365 w_k1_KO_s102
                                 + 0.960709 w_k1_MAT_s102
                                 + 1.07706 w_k1_MSFT_s102
                                 + 1.04507 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.54895 w_k1_ADM_s102
                                 + 1.82625 w_k1_AXP_s102
                                 + 1.71793 w_k1_DIS_s102
                                 + 1.45449 w_k1_FDX_s102
                                 + 1.36448 w_k1_IBM_s102
                                 + 1.17451 w_k1_INTC_s102
                                 + 1.32707 w_k1_KO_s102 + 1.51583 w_k1_MAT_s102
                                 + 1.32412 w_k1_MSFT_s102
                                 + 1.29002 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.906579 w_k1_ADM_s102
                                 + 0.823254 w_k1_AXP_s102
                                 + 0.932281 w_k1_DIS_s102
                                 + 0.780122 w_k1_FDX_s102
                                 + 1.10315 w_k1_IBM_s102
                                 + 0.909195 w_k1_INTC_s102
                                 + 1.03564 w_k1_KO_s102
                                 + 0.885504 w_k1_MAT_s102
                                 + 0.929479 w_k1_MSFT_s102
                                 + 1.03249 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.18304 w_k1_ADM_s103
                                 + 1.48853 w_k1_AXP_s103
                                 + 1.15055 w_k1_DIS_s103
                                 + 1.27437 w_k1_FDX_s103
                                 + 0.980867 w_k1_IBM_s103
                                 + 1.04536 w_k1_INTC_s103
                                 + 0.919702 w_k1_KO_s103
                                 + 0.962087 w_k1_MAT_s103
                                 + 1.22202 w_k1_MSFT_s103
                                 + 0.940229 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.8893 w_k1_ADM_s103
                                 + 1.12548 w_k1_AXP_s103
                                 + 1.53334 w_k1_DIS_s103
                                 + 1.23363 w_k1_FDX_s103
                                 + 0.954091 w_k1_IBM_s103
                                 + 1.04448 w_k1_INTC_s103
                                 + 0.999483 w_k1_KO_s103
                                 + 0.947279 w_k1_MAT_s103
                                 + 1.28272 w_k1_MSFT_s103
                                 + 1.09523 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.818333 w_k1_ADM_s103
                                 + 1.03019 w_k1_AXP_s103
                                 + 0.71226 w_k1_DIS_s103
                                 + 0.806052 w_k1_FDX_s103
                                 + 0.914027 w_k1_IBM_s103
                                 + 0.803284 w_k1_INTC_s103
                                 + 0.71375 w_k1_KO_s103
                                 + 0.916764 w_k1_MAT_s103
                                 + 0.931711 w_k1_MSFT_s103
                                 + 0.768722 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.05491 w_k1_ADM_s103
                                 + 1.04609 w_k1_AXP_s103
                                 + 0.960741 w_k1_DIS_s103
                                 + 0.969192 w_k1_FDX_s103
                                 + 0.945446 w_k1_IBM_s103
                                 + 0.975771 w_k1_INTC_s103
                                 + 0.970011 w_k1_KO_s103
                                 + 0.727624 w_k1_MAT_s103
                                 + 1.07841 w_k1_MSFT_s103
                                 + 0.940306 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.06954 w_k1_ADM_s103
                                 + 1.35008 w_k1_AXP_s103
                                 + 1.47612 w_k1_DIS_s103
                                 + 1.26791 w_k1_FDX_s103
                                 + 1.46958 w_k1_IBM_s103
                                 + 1.13506 w_k1_INTC_s103
                                 + 1.26088 w_k1_KO_s103 + 1.1913 w_k1_MAT_s103
                                 + 1.23497 w_k1_MSFT_s103
                                 + 1.07826 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.985464 w_k1_ADM_s103
                                 + 1.08906 w_k1_AXP_s103
                                 + 0.949315 w_k1_DIS_s103
                                 + 0.907684 w_k1_FDX_s103
                                 + 1.31954 w_k1_IBM_s103
                                 + 0.94816 w_k1_INTC_s103
                                 + 1.00435 w_k1_KO_s103 + 1.25282 w_k1_MAT_s103
                                 + 0.914641 w_k1_MSFT_s103
                                 + 0.984732 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.846773 w_k1_ADM_s103
                                 + 0.768478 w_k1_AXP_s103
                                 + 1.03477 w_k1_DIS_s103
                                 + 0.897023 w_k1_FDX_s103
                                 + 1.13692 w_k1_IBM_s103
                                 + 0.936907 w_k1_INTC_s103
                                 + 0.967874 w_k1_KO_s103
                                 + 0.961382 w_k1_MAT_s103
                                 + 1.02389 w_k1_MSFT_s103
                                 + 0.843992 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.07948 w_k1_ADM_s103
                                 + 0.836583 w_k1_AXP_s103
                                 + 0.985776 w_k1_DIS_s103
                                 + 0.995947 w_k1_FDX_s103
                                 + 0.672977 w_k1_IBM_s103
                                 + 0.787963 w_k1_INTC_s103
                                 + 0.750331 w_k1_KO_s103
                                 + 0.794612 w_k1_MAT_s103
                                 + 1.00363 w_k1_MSFT_s103
                                 + 0.935193 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.935612 w_k1_ADM_s104
                                 + 0.974681 w_k1_AXP_s104
                                 + 0.893123 w_k1_DIS_s104
                                 + 0.952009 w_k1_FDX_s104
                                 + 0.906947 w_k1_IBM_s104
                                 + 0.937515 w_k1_INTC_s104
                                 + 0.933217 w_k1_KO_s104
                                 + 0.890893 w_k1_MAT_s104
                                 + 0.932553 w_k1_MSFT_s104
                                 + 0.897941 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.812082 w_k1_ADM_s104
                                 + 0.933488 w_k1_AXP_s104
                                 + 1.04161 w_k1_DIS_s104
                                 + 0.969407 w_k1_FDX_s104
                                 + 1.31113 w_k1_IBM_s104
                                 + 1.03683 w_k1_INTC_s104
                                 + 1.18091 w_k1_KO_s104 + 0.78384 w_k1_MAT_s104
                                 + 0.913851 w_k1_MSFT_s104
                                 + 0.872631 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.05376 w_k1_ADM_s104
                                 + 1.39147 w_k1_AXP_s104
                                 + 1.47416 w_k1_DIS_s104 + 1.293 w_k1_FDX_s104
                                 + 0.950662 w_k1_IBM_s104
                                 + 1.48807 w_k1_INTC_s104
                                 + 1.28652 w_k1_KO_s104
                                 + 0.982172 w_k1_MAT_s104
                                 + 1.33434 w_k1_MSFT_s104
                                 + 1.03906 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.07956 w_k1_ADM_s104
                                 + 1.27763 w_k1_AXP_s104
                                 + 1.07984 w_k1_DIS_s104 + 1.1378 w_k1_FDX_s104
                                 + 1.1162 w_k1_IBM_s104
                                 + 0.977691 w_k1_INTC_s104
                                 + 0.985839 w_k1_KO_s104
                                 + 1.25228 w_k1_MAT_s104
                                 + 0.936075 w_k1_MSFT_s104
                                 + 0.969063 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.644302 w_k1_ADM_s104
                                 + 0.707126 w_k1_AXP_s104
                                 + 0.786619 w_k1_DIS_s104
                                 + 0.904329 w_k1_FDX_s104
                                 + 0.522229 w_k1_IBM_s104
                                 + 1.05152 w_k1_INTC_s104
                                 + 0.733304 w_k1_KO_s104
                                 + 0.385653 w_k1_MAT_s104
                                 + 0.833562 w_k1_MSFT_s104
                                 + 0.865777 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.887424 w_k1_ADM_s104
                                 + 0.963267 w_k1_AXP_s104
                                 + 1.1716 w_k1_DIS_s104 + 1.19799 w_k1_FDX_s104
                                 + 0.756708 w_k1_IBM_s104
                                 + 1.21067 w_k1_INTC_s104
                                 + 0.859398 w_k1_KO_s104
                                 + 0.599092 w_k1_MAT_s104
                                 + 1.13107 w_k1_MSFT_s104
                                 + 0.970154 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.699713 w_k1_ADM_s104
                                 + 0.753638 w_k1_AXP_s104
                                 + 0.604978 w_k1_DIS_s104
                                 + 0.737156 w_k1_FDX_s104
                                 + 1.00098 w_k1_IBM_s104
                                 + 0.761715 w_k1_INTC_s104
                                 + 0.829401 w_k1_KO_s104
                                 + 0.822912 w_k1_MAT_s104
                                 + 0.71456 w_k1_MSFT_s104
                                 + 0.789943 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.41192 w_k1_ADM_s104
                                 + 1.33832 w_k1_AXP_s104
                                 + 1.58872 w_k1_DIS_s104
                                 + 1.36462 w_k1_FDX_s104
                                 + 1.28715 w_k1_IBM_s104
                                 + 1.41204 w_k1_INTC_s104
                                 + 1.47013 w_k1_KO_s104 + 1.3892 w_k1_MAT_s104
                                 + 1.39866 w_k1_MSFT_s104
                                 + 1.10387 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.14636 w_k1_ADM_s105
                                 + 1.45483 w_k1_AXP_s105
                                 + 1.35268 w_k1_DIS_s105
                                 + 1.26109 w_k1_FDX_s105
                                 + 1.10571 w_k1_IBM_s105
                                 + 1.25201 w_k1_INTC_s105
                                 + 1.02897 w_k1_KO_s105
                                 + 0.743364 w_k1_MAT_s105
                                 + 1.18725 w_k1_MSFT_s105 + 1.1233 w_k1_PG_s105
                                 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.918979 w_k1_ADM_s105
                                 + 0.905137 w_k1_AXP_s105
                                 + 0.849397 w_k1_DIS_s105
                                 + 0.824826 w_k1_FDX_s105
                                 + 1.33273 w_k1_IBM_s105
                                 + 0.973452 w_k1_INTC_s105
                                 + 1.00377 w_k1_KO_s105 + 1.01799 w_k1_MAT_s105
                                 + 0.995548 w_k1_MSFT_s105
                                 + 0.970184 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.875033 w_k1_ADM_s105
                                 + 0.89606 w_k1_AXP_s105
                                 + 1.05288 w_k1_DIS_s105
                                 + 0.993171 w_k1_FDX_s105
                                 + 0.936655 w_k1_IBM_s105
                                 + 1.15471 w_k1_INTC_s105
                                 + 0.957257 w_k1_KO_s105
                                 + 0.958905 w_k1_MAT_s105
                                 + 1.06042 w_k1_MSFT_s105
                                 + 0.973869 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.01912 w_k1_ADM_s105
                                 + 0.984575 w_k1_AXP_s105 + 1.033 w_k1_DIS_s105
                                 + 1.20579 w_k1_FDX_s105
                                 + 0.720093 w_k1_IBM_s105
                                 + 0.991824 w_k1_INTC_s105
                                 + 0.768911 w_k1_KO_s105
                                 + 0.766017 w_k1_MAT_s105
                                 + 1.23093 w_k1_MSFT_s105 + 1.0199 w_k1_PG_s105
                                 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.818336 w_k1_ADM_s105
                                 + 0.657547 w_k1_AXP_s105
                                 + 0.528902 w_k1_DIS_s105
                                 + 0.692964 w_k1_FDX_s105
                                 + 0.908476 w_k1_IBM_s105
                                 + 0.727573 w_k1_INTC_s105
                                 + 0.672131 w_k1_KO_s105
                                 + 0.76371 w_k1_MAT_s105
                                 + 0.719243 w_k1_MSFT_s105
                                 + 0.781412 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.2181 w_k1_ADM_s105
                                 + 1.22091 w_k1_AXP_s105
                                 + 1.15304 w_k1_DIS_s105
                                 + 1.07605 w_k1_FDX_s105 + 1.2771 w_k1_IBM_s105
                                 + 0.921886 w_k1_INTC_s105
                                 + 1.11169 w_k1_KO_s105 + 1.42029 w_k1_MAT_s105
                                 + 1.01814 w_k1_MSFT_s105
                                 + 1.01459 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.32621 w_k1_ADM_s105
                                 + 1.34003 w_k1_AXP_s105
                                 + 1.46426 w_k1_DIS_s105
                                 + 1.20711 w_k1_FDX_s105
                                 + 2.06699 w_k1_IBM_s105
                                 + 1.37507 w_k1_INTC_s105
                                 + 1.33078 w_k1_KO_s105 + 1.6208 w_k1_MAT_s105
                                 + 1.39712 w_k1_MSFT_s105
                                 + 1.14432 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.33213 w_k1_ADM_s105
                                 + 1.3865 w_k1_AXP_s105 + 1.90023 w_k1_DIS_s105
                                 + 1.68014 w_k1_FDX_s105
                                 + 0.908074 w_k1_IBM_s105
                                 + 1.22238 w_k1_INTC_s105 + 1.0845 w_k1_KO_s105
                                 + 0.689677 w_k1_MAT_s105
                                 + 1.36586 w_k1_MSFT_s105
                                 + 1.46283 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.02552 w_k1_ADM_s106
                                 + 0.930532 w_k1_AXP_s106
                                 + 1.56249 w_k1_DIS_s106
                                 + 1.22381 w_k1_FDX_s106
                                 + 1.22009 w_k1_IBM_s106
                                 + 1.53234 w_k1_INTC_s106
                                 + 1.42291 w_k1_KO_s106 + 0.99225 w_k1_MAT_s106
                                 + 1.33982 w_k1_MSFT_s106
                                 + 1.07458 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.956258 w_k1_ADM_s106
                                 + 1.00016 w_k1_AXP_s106
                                 + 0.770463 w_k1_DIS_s106
                                 + 0.908839 w_k1_FDX_s106
                                 + 0.995645 w_k1_IBM_s106
                                 + 0.93439 w_k1_INTC_s106
                                 + 0.951441 w_k1_KO_s106
                                 + 0.845251 w_k1_MAT_s106
                                 + 0.834397 w_k1_MSFT_s106
                                 + 0.814502 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.0696 w_k1_ADM_s106
                                 + 1.18515 w_k1_AXP_s106
                                 + 0.964627 w_k1_DIS_s106
                                 + 1.10497 w_k1_FDX_s106
                                 + 0.674479 w_k1_IBM_s106
                                 + 1.28933 w_k1_INTC_s106
                                 + 0.771532 w_k1_KO_s106
                                 + 0.445588 w_k1_MAT_s106
                                 + 0.993799 w_k1_MSFT_s106
                                 + 0.844358 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.743908 w_k1_ADM_s106
                                 + 0.786172 w_k1_AXP_s106
                                 + 0.862555 w_k1_DIS_s106
                                 + 0.775131 w_k1_FDX_s106
                                 + 1.08836 w_k1_IBM_s106
                                 + 0.939949 w_k1_INTC_s106
                                 + 0.992673 w_k1_KO_s106
                                 + 0.816702 w_k1_MAT_s106
                                 + 0.747161 w_k1_MSFT_s106
                                 + 0.857429 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.922672 w_k1_ADM_s106
                                 + 1.35472 w_k1_AXP_s106
                                 + 1.24336 w_k1_DIS_s106
                                 + 1.16053 w_k1_FDX_s106
                                 + 0.906167 w_k1_IBM_s106
                                 + 1.15991 w_k1_INTC_s106
                                 + 1.02352 w_k1_KO_s106
                                 + 0.834863 w_k1_MAT_s106
                                 + 0.994401 w_k1_MSFT_s106
                                 + 0.888768 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.882357 w_k1_ADM_s106
                                 + 1.23616 w_k1_AXP_s106
                                 + 1.27801 w_k1_DIS_s106
                                 + 1.05569 w_k1_FDX_s106
                                 + 1.58573 w_k1_IBM_s106
                                 + 1.48168 w_k1_INTC_s106
                                 + 1.39946 w_k1_KO_s106 + 1.31767 w_k1_MAT_s106
                                 + 1.19444 w_k1_MSFT_s106
                                 + 1.20064 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.23651 w_k1_ADM_s106
                                 + 1.72306 w_k1_AXP_s106
                                 + 1.47184 w_k1_DIS_s106 + 1.3902 w_k1_FDX_s106
                                 + 1.11284 w_k1_IBM_s106
                                 + 1.55397 w_k1_INTC_s106
                                 + 1.06748 w_k1_KO_s106
                                 + 0.942282 w_k1_MAT_s106
                                 + 1.16076 w_k1_MSFT_s106
                                 + 1.14917 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.07237 w_k1_ADM_s106
                                 + 0.944773 w_k1_AXP_s106
                                 + 0.934574 w_k1_DIS_s106
                                 + 0.943857 w_k1_FDX_s106
                                 + 0.928164 w_k1_IBM_s106
                                 + 0.971916 w_k1_INTC_s106
                                 + 0.979214 w_k1_KO_s106
                                 + 0.695647 w_k1_MAT_s106
                                 + 0.933053 w_k1_MSFT_s106
                                 + 0.922935 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.08115 w_k1_ADM_s107
                                 + 1.02513 w_k1_AXP_s107
                                 + 1.06434 w_k1_DIS_s107
                                 + 1.01402 w_k1_FDX_s107
                                 + 0.931337 w_k1_IBM_s107
                                 + 0.903452 w_k1_INTC_s107
                                 + 1.00392 w_k1_KO_s107
                                 + 0.790168 w_k1_MAT_s107
                                 + 0.927584 w_k1_MSFT_s107
                                 + 0.957228 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.2025 w_k1_ADM_s107
                                 + 0.958974 w_k1_AXP_s107
                                 + 1.28159 w_k1_DIS_s107
                                 + 1.09075 w_k1_FDX_s107
                                 + 1.20553 w_k1_IBM_s107
                                 + 1.20843 w_k1_INTC_s107
                                 + 1.16644 w_k1_KO_s107
                                 + 0.833243 w_k1_MAT_s107
                                 + 1.14209 w_k1_MSFT_s107
                                 + 0.999417 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.917918 w_k1_ADM_s107
                                 + 1.0466 w_k1_AXP_s107
                                 + 0.852025 w_k1_DIS_s107
                                 + 0.883156 w_k1_FDX_s107
                                 + 1.10215 w_k1_IBM_s107
                                 + 1.03172 w_k1_INTC_s107
                                 + 0.894481 w_k1_KO_s107
                                 + 0.787517 w_k1_MAT_s107
                                 + 0.90532 w_k1_MSFT_s107
                                 + 0.881973 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.947806 w_k1_ADM_s107
                                 + 0.68059 w_k1_AXP_s107
                                 + 0.826845 w_k1_DIS_s107
                                 + 0.86944 w_k1_FDX_s107
                                 + 1.15658 w_k1_IBM_s107
                                 + 0.684737 w_k1_INTC_s107
                                 + 0.887505 w_k1_KO_s107
                                 + 0.954298 w_k1_MAT_s107
                                 + 0.813559 w_k1_MSFT_s107
                                 + 0.969373 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.17128 w_k1_ADM_s107
                                 + 0.810197 w_k1_AXP_s107
                                 + 0.813352 w_k1_DIS_s107
                                 + 0.96487 w_k1_FDX_s107
                                 + 0.76828 w_k1_IBM_s107
                                 + 0.896527 w_k1_INTC_s107
                                 + 0.927234 w_k1_KO_s107
                                 + 0.648367 w_k1_MAT_s107
                                 + 1.01202 w_k1_MSFT_s107
                                 + 0.958221 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.43574 w_k1_ADM_s107
                                 + 1.25772 w_k1_AXP_s107 + 1.2671 w_k1_DIS_s107
                                 + 1.13732 w_k1_FDX_s107
                                 + 1.50853 w_k1_IBM_s107
                                 + 0.850958 w_k1_INTC_s107
                                 + 1.31033 w_k1_KO_s107 + 1.41159 w_k1_MAT_s107
                                 + 1.00882 w_k1_MSFT_s107
                                 + 1.05563 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.854092 w_k1_ADM_s107
                                 + 0.739349 w_k1_AXP_s107
                                 + 0.668952 w_k1_DIS_s107
                                 + 0.786684 w_k1_FDX_s107
                                 + 0.736981 w_k1_IBM_s107
                                 + 0.796181 w_k1_INTC_s107
                                 + 0.749761 w_k1_KO_s107
                                 + 0.495844 w_k1_MAT_s107
                                 + 0.860759 w_k1_MSFT_s107
                                 + 0.847911 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.22166 w_k1_ADM_s107
                                 + 1.48596 w_k1_AXP_s107
                                 + 1.53688 w_k1_DIS_s107
                                 + 1.34254 w_k1_FDX_s107
                                 + 1.03355 w_k1_IBM_s107
                                 + 1.14161 w_k1_INTC_s107
                                 + 1.21244 w_k1_KO_s107
                                 + 0.961279 w_k1_MAT_s107
                                 + 1.26604 w_k1_MSFT_s107
                                 + 1.19526 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.20918 w_k1_ADM_s110
                                 + 1.57212 w_k1_AXP_s110
                                 + 1.57499 w_k1_DIS_s110 + 1.3069 w_k1_FDX_s110
                                 + 1.42593 w_k1_IBM_s110
                                 + 1.13179 w_k1_INTC_s110
                                 + 1.40266 w_k1_KO_s110 + 1.55264 w_k1_MAT_s110
                                 + 1.23687 w_k1_MSFT_s110
                                 + 1.33025 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.27369 w_k1_ADM_s110
                                 + 0.960946 w_k1_AXP_s110
                                 + 1.14664 w_k1_DIS_s110 + 1.1806 w_k1_FDX_s110
                                 + 0.779627 w_k1_IBM_s110
                                 + 1.04097 w_k1_INTC_s110
                                 + 0.803491 w_k1_KO_s110
                                 + 0.450939 w_k1_MAT_s110
                                 + 1.17396 w_k1_MSFT_s110
                                 + 0.951547 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.05871 w_k1_ADM_s110
                                 + 1.05725 w_k1_AXP_s110
                                 + 1.65621 w_k1_DIS_s110
                                 + 1.15549 w_k1_FDX_s110
                                 + 1.17053 w_k1_IBM_s110
                                 + 1.04197 w_k1_INTC_s110
                                 + 1.20162 w_k1_KO_s110
                                 + 0.938916 w_k1_MAT_s110
                                 + 1.30436 w_k1_MSFT_s110
                                 + 1.05796 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.973499 w_k1_ADM_s110
                                 + 0.800304 w_k1_AXP_s110
                                 + 0.770399 w_k1_DIS_s110
                                 + 0.817641 w_k1_FDX_s110
                                 + 0.769143 w_k1_IBM_s110
                                 + 0.799094 w_k1_INTC_s110
                                 + 0.690403 w_k1_KO_s110
                                 + 0.730767 w_k1_MAT_s110
                                 + 0.834373 w_k1_MSFT_s110
                                 + 0.784404 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.981749 w_k1_ADM_s110
                                 + 1.12639 w_k1_AXP_s110
                                 + 0.802195 w_k1_DIS_s110
                                 + 0.870574 w_k1_FDX_s110
                                 + 1.09814 w_k1_IBM_s110
                                 + 0.655515 w_k1_INTC_s110
                                 + 0.781431 w_k1_KO_s110
                                 + 0.983774 w_k1_MAT_s110
                                 + 0.734251 w_k1_MSFT_s110
                                 + 0.890304 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.13518 w_k1_ADM_s110
                                 + 1.22078 w_k1_AXP_s110
                                 + 1.28281 w_k1_DIS_s110
                                 + 1.04217 w_k1_FDX_s110
                                 + 1.45981 w_k1_IBM_s110
                                 + 0.988637 w_k1_INTC_s110
                                 + 1.21956 w_k1_KO_s110 + 1.31106 w_k1_MAT_s110
                                 + 1.10765 w_k1_MSFT_s110 + 1.0687 w_k1_PG_s110
                                 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.01108 w_k1_ADM_s110
                                 + 1.31157 w_k1_AXP_s110
                                 + 1.04304 w_k1_DIS_s110
                                 + 1.15865 w_k1_FDX_s110
                                 + 1.01193 w_k1_IBM_s110
                                 + 1.08095 w_k1_INTC_s110
                                 + 0.952838 w_k1_KO_s110
                                 + 1.22395 w_k1_MAT_s110
                                 + 1.03693 w_k1_MSFT_s110
                                 + 1.04013 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.886575 w_k1_ADM_s110
                                 + 0.82216 w_k1_AXP_s110
                                 + 1.02874 w_k1_DIS_s110
                                 + 0.943464 w_k1_FDX_s110
                                 + 1.04651 w_k1_IBM_s110
                                 + 0.825705 w_k1_INTC_s110
                                 + 1.00601 w_k1_KO_s110
                                 + 0.951184 w_k1_MAT_s110
                                 + 0.976159 w_k1_MSFT_s110
                                 + 1.00555 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.07366 w_k1_ADM_s111
                                 + 1.04933 w_k1_AXP_s111
                                 + 1.00779 w_k1_DIS_s111
                                 + 0.997522 w_k1_FDX_s111
                                 + 0.987778 w_k1_IBM_s111
                                 + 1.02965 w_k1_INTC_s111
                                 + 0.966793 w_k1_KO_s111
                                 + 1.01784 w_k1_MAT_s111
                                 + 0.952815 w_k1_MSFT_s111
                                 + 1.06431 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.81792 w_k1_ADM_s111
                                 + 0.712758 w_k1_AXP_s111
                                 + 0.582197 w_k1_DIS_s111
                                 + 0.775015 w_k1_FDX_s111
                                 + 0.786663 w_k1_IBM_s111
                                 + 0.823848 w_k1_INTC_s111
                                 + 0.709711 w_k1_KO_s111
                                 + 0.703251 w_k1_MAT_s111
                                 + 0.728506 w_k1_MSFT_s111
                                 + 0.885199 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.807612 w_k1_ADM_s111
                                 + 1.13745 w_k1_AXP_s111
                                 + 1.24567 w_k1_DIS_s111
                                 + 0.972093 w_k1_FDX_s111
                                 + 1.51276 w_k1_IBM_s111
                                 + 1.22809 w_k1_INTC_s111 + 1.1352 w_k1_KO_s111
                                 + 1.7806 w_k1_MAT_s111
                                 + 0.954831 w_k1_MSFT_s111
                                 + 1.11395 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.04273 w_k1_ADM_s111
                                 + 1.29999 w_k1_AXP_s111
                                 + 1.06943 w_k1_DIS_s111
                                 + 1.01443 w_k1_FDX_s111
                                 + 0.846996 w_k1_IBM_s111
                                 + 1.07322 w_k1_INTC_s111
                                 + 0.869209 w_k1_KO_s111 + 1.1349 w_k1_MAT_s111
                                 + 1.06727 w_k1_MSFT_s111
                                 + 1.10282 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.15882 w_k1_ADM_s111
                                 + 1.64713 w_k1_AXP_s111
                                 + 1.55351 w_k1_DIS_s111
                                 + 1.32817 w_k1_FDX_s111
                                 + 1.23032 w_k1_IBM_s111
                                 + 1.2113 w_k1_INTC_s111 + 1.13352 w_k1_KO_s111
                                 + 1.23805 w_k1_MAT_s111
                                 + 1.2588 w_k1_MSFT_s111 + 1.2642 w_k1_PG_s111
                                 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.766467 w_k1_ADM_s111
                                 + 0.748772 w_k1_AXP_s111
                                 + 0.616209 w_k1_DIS_s111
                                 + 0.834731 w_k1_FDX_s111
                                 + 0.574104 w_k1_IBM_s111
                                 + 0.940192 w_k1_INTC_s111
                                 + 0.58925 w_k1_KO_s111
                                 + 0.474595 w_k1_MAT_s111
                                 + 0.888328 w_k1_MSFT_s111
                                 + 0.818522 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.0388 w_k1_ADM_s111
                                 + 1.02689 w_k1_AXP_s111
                                 + 1.27193 w_k1_DIS_s111 + 1.1866 w_k1_FDX_s111
                                 + 0.7477 w_k1_IBM_s111
                                 + 1.23492 w_k1_INTC_s111 + 0.9719 w_k1_KO_s111
                                 + 0.810048 w_k1_MAT_s111
                                 + 1.10234 w_k1_MSFT_s111
                                 + 1.07633 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.701179 w_k1_ADM_s111
                                 + 0.797564 w_k1_AXP_s111
                                 + 0.789749 w_k1_DIS_s111
                                 + 0.74574 w_k1_FDX_s111
                                 + 1.12056 w_k1_IBM_s111
                                 + 0.906382 w_k1_INTC_s111
                                 + 0.810471 w_k1_KO_s111
                                 + 0.912436 w_k1_MAT_s111
                                 + 0.755808 w_k1_MSFT_s111
                                 + 0.965297 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.03372 w_k1_ADM_s112
                                 + 1.22765 w_k1_AXP_s112
                                 + 1.20454 w_k1_DIS_s112
                                 + 1.00263 w_k1_FDX_s112
                                 + 1.48382 w_k1_IBM_s112
                                 + 0.805613 w_k1_INTC_s112
                                 + 1.14417 w_k1_KO_s112 + 1.77832 w_k1_MAT_s112
                                 + 0.951413 w_k1_MSFT_s112
                                 + 1.04932 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.05351 w_k1_ADM_s112
                                 + 0.784471 w_k1_AXP_s112
                                 + 1.17261 w_k1_DIS_s112
                                 + 1.01867 w_k1_FDX_s112
                                 + 0.813801 w_k1_IBM_s112
                                 + 0.84933 w_k1_INTC_s112 + 1.1153 w_k1_KO_s112
                                 + 0.869415 w_k1_MAT_s112
                                 + 1.00306 w_k1_MSFT_s112
                                 + 1.05906 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.22689 w_k1_ADM_s112
                                 + 1.21921 w_k1_AXP_s112
                                 + 1.66315 w_k1_DIS_s112 + 1.3738 w_k1_FDX_s112
                                 + 0.972253 w_k1_IBM_s112
                                 + 1.19644 w_k1_INTC_s112 + 1.3136 w_k1_KO_s112
                                 + 1.03146 w_k1_MAT_s112
                                 + 1.28295 w_k1_MSFT_s112
                                 + 1.28487 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.47742 w_k1_ADM_s112
                                 + 1.04847 w_k1_AXP_s112
                                 + 1.01387 w_k1_DIS_s112
                                 + 1.19935 w_k1_FDX_s112
                                 + 0.769002 w_k1_IBM_s112
                                 + 0.960727 w_k1_INTC_s112
                                 + 0.887336 w_k1_KO_s112
                                 + 0.83417 w_k1_MAT_s112
                                 + 0.962123 w_k1_MSFT_s112
                                 + 1.02014 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.55665 w_k1_ADM_s112
                                 + 1.48133 w_k1_AXP_s112
                                 + 1.81755 w_k1_DIS_s112
                                 + 1.45195 w_k1_FDX_s112
                                 + 1.39244 w_k1_IBM_s112
                                 + 1.18998 w_k1_INTC_s112
                                 + 1.47295 w_k1_KO_s112 + 1.75686 w_k1_MAT_s112
                                 + 1.19322 w_k1_MSFT_s112
                                 + 1.28684 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.26557 w_k1_ADM_s112
                                 + 1.26324 w_k1_AXP_s112
                                 + 1.00633 w_k1_DIS_s112
                                 + 1.18523 w_k1_FDX_s112
                                 + 1.02551 w_k1_IBM_s112
                                 + 0.926463 w_k1_INTC_s112
                                 + 0.904606 w_k1_KO_s112
                                 + 1.38496 w_k1_MAT_s112
                                 + 1.05999 w_k1_MSFT_s112
                                 + 1.17846 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.31472 w_k1_ADM_s112
                                 + 0.875446 w_k1_AXP_s112
                                 + 0.83234 w_k1_DIS_s112
                                 + 0.921015 w_k1_FDX_s112
                                 + 0.859173 w_k1_IBM_s112
                                 + 0.767794 w_k1_INTC_s112
                                 + 0.818908 w_k1_KO_s112
                                 + 0.80834 w_k1_MAT_s112
                                 + 0.857913 w_k1_MSFT_s112
                                 + 1.03569 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.871777 w_k1_ADM_s112
                                 + 0.704169 w_k1_AXP_s112
                                 + 0.650115 w_k1_DIS_s112
                                 + 0.771922 w_k1_FDX_s112
                                 + 0.654786 w_k1_IBM_s112
                                 + 0.652835 w_k1_INTC_s112
                                 + 0.671951 w_k1_KO_s112
                                 + 0.629133 w_k1_MAT_s112
                                 + 0.680823 w_k1_MSFT_s112
                                 + 0.813516 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.945069 w_k1_ADM_s113
                                 + 0.914386 w_k1_AXP_s113
                                 + 0.805715 w_k1_DIS_s113
                                 + 0.828948 w_k1_FDX_s113
                                 + 0.975139 w_k1_IBM_s113
                                 + 0.803094 w_k1_INTC_s113
                                 + 0.86001 w_k1_KO_s113 + 1.01655 w_k1_MAT_s113
                                 + 0.894821 w_k1_MSFT_s113
                                 + 0.792223 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.17642 w_k1_ADM_s113
                                 + 1.46493 w_k1_AXP_s113
                                 + 1.09927 w_k1_DIS_s113
                                 + 1.18148 w_k1_FDX_s113
                                 + 0.976125 w_k1_IBM_s113
                                 + 1.15915 w_k1_INTC_s113
                                 + 1.02696 w_k1_KO_s113
                                 + 0.886393 w_k1_MAT_s113
                                 + 1.08708 w_k1_MSFT_s113
                                 + 0.900245 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.0889 w_k1_ADM_s113
                                 + 0.876915 w_k1_AXP_s113
                                 + 0.758967 w_k1_DIS_s113
                                 + 0.913323 w_k1_FDX_s113
                                 + 0.663891 w_k1_IBM_s113
                                 + 0.804643 w_k1_INTC_s113
                                 + 0.863481 w_k1_KO_s113
                                 + 0.810968 w_k1_MAT_s113
                                 + 0.969908 w_k1_MSFT_s113
                                 + 0.865289 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.990358 w_k1_ADM_s113
                                 + 1.33759 w_k1_AXP_s113
                                 + 1.49969 w_k1_DIS_s113
                                 + 1.13235 w_k1_FDX_s113
                                 + 1.45834 w_k1_IBM_s113
                                 + 1.06461 w_k1_INTC_s113
                                 + 1.49249 w_k1_KO_s113 + 1.40718 w_k1_MAT_s113
                                 + 1.29606 w_k1_MSFT_s113
                                 + 1.18748 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.14669 w_k1_ADM_s113
                                 + 1.43929 w_k1_AXP_s113
                                 + 1.41714 w_k1_DIS_s113
                                 + 1.27454 w_k1_FDX_s113
                                 + 1.13705 w_k1_IBM_s113
                                 + 1.14124 w_k1_INTC_s113
                                 + 1.29741 w_k1_KO_s113 + 1.12957 w_k1_MAT_s113
                                 + 1.20203 w_k1_MSFT_s113 + 1.0979 w_k1_PG_s113
                                 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.985172 w_k1_ADM_s113
                                 + 0.99714 w_k1_AXP_s113
                                 + 1.14132 w_k1_DIS_s113
                                 + 0.98443 w_k1_FDX_s113
                                 + 0.976782 w_k1_IBM_s113
                                 + 0.97301 w_k1_INTC_s113
                                 + 1.13147 w_k1_KO_s113
                                 + 0.991252 w_k1_MAT_s113
                                 + 1.037 w_k1_MSFT_s113 + 0.999854 w_k1_PG_s113
                                 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.744467 w_k1_ADM_s113
                                 + 0.723162 w_k1_AXP_s113
                                 + 0.647152 w_k1_DIS_s113
                                 + 0.675939 w_k1_FDX_s113
                                 + 0.949097 w_k1_IBM_s113
                                 + 0.657772 w_k1_INTC_s113
                                 + 0.881529 w_k1_KO_s113
                                 + 0.643897 w_k1_MAT_s113
                                 + 0.669253 w_k1_MSFT_s113
                                 + 0.818585 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.860992 w_k1_ADM_s113
                                 + 0.968868 w_k1_AXP_s113
                                 + 0.96643 w_k1_DIS_s113
                                 + 0.848793 w_k1_FDX_s113
                                 + 1.37246 w_k1_IBM_s113
                                 + 0.894007 w_k1_INTC_s113
                                 + 1.19009 w_k1_KO_s113 + 1.32782 w_k1_MAT_s113
                                 + 1.01339 w_k1_MSFT_s113
                                 + 0.963021 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.680007 w_k1_ADM_s114
                                 + 0.802803 w_k1_AXP_s114
                                 + 0.609017 w_k1_DIS_s114
                                 + 0.739779 w_k1_FDX_s114
                                 + 0.752315 w_k1_IBM_s114
                                 + 0.98054 w_k1_INTC_s114
                                 + 0.728093 w_k1_KO_s114
                                 + 0.772259 w_k1_MAT_s114
                                 + 0.843335 w_k1_MSFT_s114
                                 + 0.831188 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.684487 w_k1_ADM_s114
                                 + 0.768946 w_k1_AXP_s114
                                 + 0.814625 w_k1_DIS_s114
                                 + 0.821746 w_k1_FDX_s114
                                 + 1.02067 w_k1_IBM_s114
                                 + 0.816613 w_k1_INTC_s114
                                 + 0.900733 w_k1_KO_s114
                                 + 0.822925 w_k1_MAT_s114
                                 + 0.840728 w_k1_MSFT_s114
                                 + 0.865335 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.952899 w_k1_ADM_s114
                                 + 0.91368 w_k1_AXP_s114
                                 + 0.843215 w_k1_DIS_s114
                                 + 0.979656 w_k1_FDX_s114
                                 + 1.02033 w_k1_IBM_s114
                                 + 1.1022 w_k1_INTC_s114 + 1.04698 w_k1_KO_s114
                                 + 1.01391 w_k1_MAT_s114
                                 + 1.03246 w_k1_MSFT_s114
                                 + 0.879513 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.959165 w_k1_ADM_s114
                                 + 1.06959 w_k1_AXP_s114
                                 + 1.08668 w_k1_DIS_s114
                                 + 1.18016 w_k1_FDX_s114
                                 + 0.668849 w_k1_IBM_s114
                                 + 1.21744 w_k1_INTC_s114
                                 + 0.805457 w_k1_KO_s114
                                 + 0.735167 w_k1_MAT_s114
                                 + 1.05346 w_k1_MSFT_s114
                                 + 0.966035 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.937649 w_k1_ADM_s114
                                 + 1.12775 w_k1_AXP_s114
                                 + 1.31934 w_k1_DIS_s114
                                 + 1.06127 w_k1_FDX_s114
                                 + 1.53429 w_k1_IBM_s114
                                 + 1.25638 w_k1_INTC_s114
                                 + 1.37194 w_k1_KO_s114 + 1.74923 w_k1_MAT_s114
                                 + 1.17112 w_k1_MSFT_s114
                                 + 1.13368 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.11213 w_k1_ADM_s114
                                 + 1.07461 w_k1_AXP_s114
                                 + 0.771849 w_k1_DIS_s114
                                 + 0.893566 w_k1_FDX_s114
                                 + 0.846827 w_k1_IBM_s114
                                 + 1.04153 w_k1_INTC_s114
                                 + 0.810521 w_k1_KO_s114
                                 + 0.873393 w_k1_MAT_s114
                                 + 0.888613 w_k1_MSFT_s114
                                 + 0.900292 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.17566 w_k1_ADM_s114
                                 + 1.46509 w_k1_AXP_s114
                                 + 1.33273 w_k1_DIS_s114
                                 + 1.23075 w_k1_FDX_s114
                                 + 0.987119 w_k1_IBM_s114
                                 + 1.34201 w_k1_INTC_s114
                                 + 1.20496 w_k1_KO_s114 + 1.04261 w_k1_MAT_s114
                                 + 1.16633 w_k1_MSFT_s114 + 1.0279 w_k1_PG_s114
                                 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.822383 w_k1_ADM_s114
                                 + 1.48308 w_k1_AXP_s114
                                 + 1.02309 w_k1_DIS_s114
                                 + 0.991492 w_k1_FDX_s114
                                 + 1.25939 w_k1_IBM_s114
                                 + 1.17339 w_k1_INTC_s114
                                 + 1.23038 w_k1_KO_s114 + 2.0101 w_k1_MAT_s114
                                 + 0.93776 w_k1_MSFT_s114
                                 + 1.10748 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.32817 w_k1_ADM_s115
                                 + 1.58404 w_k1_AXP_s115
                                 + 1.28388 w_k1_DIS_s115
                                 + 1.12252 w_k1_FDX_s115
                                 + 1.37918 w_k1_IBM_s115
                                 + 1.03751 w_k1_INTC_s115
                                 + 1.22519 w_k1_KO_s115 + 1.40348 w_k1_MAT_s115
                                 + 1.03956 w_k1_MSFT_s115
                                 + 1.22166 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.898203 w_k1_ADM_s115
                                 + 0.69319 w_k1_AXP_s115
                                 + 0.681616 w_k1_DIS_s115
                                 + 0.705129 w_k1_FDX_s115
                                 + 0.798438 w_k1_IBM_s115
                                 + 0.629701 w_k1_INTC_s115
                                 + 0.713165 w_k1_KO_s115
                                 + 0.724488 w_k1_MAT_s115
                                 + 0.756397 w_k1_MSFT_s115
                                 + 0.814642 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.00257 w_k1_ADM_s115
                                 + 1.02562 w_k1_AXP_s115
                                 + 0.983867 w_k1_DIS_s115
                                 + 0.935724 w_k1_FDX_s115
                                 + 1.11288 w_k1_IBM_s115
                                 + 0.881101 w_k1_INTC_s115
                                 + 0.876948 w_k1_KO_s115
                                 + 1.00672 w_k1_MAT_s115
                                 + 0.895464 w_k1_MSFT_s115
                                 + 1.05316 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.947768 w_k1_ADM_s115
                                 + 0.998788 w_k1_AXP_s115
                                 + 1.29352 w_k1_DIS_s115
                                 + 0.952873 w_k1_FDX_s115
                                 + 1.68684 w_k1_IBM_s115
                                 + 1.12188 w_k1_INTC_s115
                                 + 1.51118 w_k1_KO_s115 + 1.22946 w_k1_MAT_s115
                                 + 1.02322 w_k1_MSFT_s115
                                 + 1.22737 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.38416 w_k1_ADM_s115
                                 + 1.46516 w_k1_AXP_s115 + 1.5349 w_k1_DIS_s115
                                 + 1.40574 w_k1_FDX_s115
                                 + 1.08019 w_k1_IBM_s115
                                 + 1.28082 w_k1_INTC_s115
                                 + 1.26048 w_k1_KO_s115
                                 + 0.764164 w_k1_MAT_s115
                                 + 1.24721 w_k1_MSFT_s115
                                 + 1.21571 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.876924 w_k1_ADM_s115
                                 + 0.81699 w_k1_AXP_s115
                                 + 0.738124 w_k1_DIS_s115
                                 + 0.895362 w_k1_FDX_s115
                                 + 0.657558 w_k1_IBM_s115
                                 + 0.867454 w_k1_INTC_s115
                                 + 0.747759 w_k1_KO_s115
                                 + 0.645228 w_k1_MAT_s115
                                 + 0.896599 w_k1_MSFT_s115
                                 + 0.943979 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.35052 w_k1_ADM_s115
                                 + 1.04543 w_k1_AXP_s115
                                 + 1.20203 w_k1_DIS_s115
                                 + 1.23361 w_k1_FDX_s115
                                 + 0.913747 w_k1_IBM_s115
                                 + 0.927085 w_k1_INTC_s115
                                 + 0.936273 w_k1_KO_s115
                                 + 0.80184 w_k1_MAT_s115
                                 + 1.09788 w_k1_MSFT_s115
                                 + 1.05032 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 0.86686 w_k1_ADM_s115
                                 + 0.925378 w_k1_AXP_s115
                                 + 1.16329 w_k1_DIS_s115
                                 + 1.05402 w_k1_FDX_s115
                                 + 0.897686 w_k1_IBM_s115
                                 + 0.871294 w_k1_INTC_s115
                                 + 0.894192 w_k1_KO_s115
                                 + 1.20992 w_k1_MAT_s115
                                 + 1.10654 w_k1_MSFT_s115
                                 + 1.00904 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.06254 w_k1_ADM_s116
                                 + 1.12247 w_k1_AXP_s116
                                 + 1.59529 w_k1_DIS_s116 + 1.202 w_k1_FDX_s116
                                 + 1.4358 w_k1_IBM_s116
                                 + 1.67027 w_k1_INTC_s116
                                 + 1.58201 w_k1_KO_s116 + 1.34917 w_k1_MAT_s116
                                 + 1.31655 w_k1_MSFT_s116
                                 + 1.23037 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.2076 w_k1_ADM_s116
                                 + 1.30412 w_k1_AXP_s116
                                 + 1.52498 w_k1_DIS_s116
                                 + 1.37407 w_k1_FDX_s116
                                 + 1.08012 w_k1_IBM_s116
                                 + 1.39816 w_k1_INTC_s116
                                 + 1.24179 w_k1_KO_s116 + 1.26469 w_k1_MAT_s116
                                 + 1.23826 w_k1_MSFT_s116
                                 + 1.25169 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.679878 w_k1_ADM_s116
                                 + 0.821361 w_k1_AXP_s116
                                 + 1.04699 w_k1_DIS_s116
                                 + 0.873468 w_k1_FDX_s116
                                 + 1.37372 w_k1_IBM_s116
                                 + 1.06175 w_k1_INTC_s116
                                 + 1.12194 w_k1_KO_s116 + 1.63024 w_k1_MAT_s116
                                 + 0.914032 w_k1_MSFT_s116
                                 + 0.970734 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.932812 w_k1_ADM_s116
                                 + 1.01636 w_k1_AXP_s116
                                 + 1.08762 w_k1_DIS_s116
                                 + 1.08178 w_k1_FDX_s116
                                 + 0.794987 w_k1_IBM_s116
                                 + 1.12702 w_k1_INTC_s116
                                 + 0.96812 w_k1_KO_s116 + 1.16661 w_k1_MAT_s116
                                 + 1.04996 w_k1_MSFT_s116
                                 + 1.06595 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.04127 w_k1_ADM_s116
                                 + 1.13912 w_k1_AXP_s116
                                 + 1.12982 w_k1_DIS_s116
                                 + 0.985735 w_k1_FDX_s116
                                 + 1.80657 w_k1_IBM_s116
                                 + 1.2099 w_k1_INTC_s116 + 1.41955 w_k1_KO_s116
                                 + 1.70649 w_k1_MAT_s116
                                 + 1.0583 w_k1_MSFT_s116 + 1.16048 w_k1_PG_s116
                                 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.957911 w_k1_ADM_s116
                                 + 0.99098 w_k1_AXP_s116
                                 + 0.807693 w_k1_DIS_s116
                                 + 0.819298 w_k1_FDX_s116
                                 + 1.22296 w_k1_IBM_s116
                                 + 0.856677 w_k1_INTC_s116
                                 + 1.07399 w_k1_KO_s116 + 1.53718 w_k1_MAT_s116
                                 + 0.765713 w_k1_MSFT_s116
                                 + 0.906638 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.733484 w_k1_ADM_s116
                                 + 0.552008 w_k1_AXP_s116
                                 + 0.612301 w_k1_DIS_s116
                                 + 0.726529 w_k1_FDX_s116
                                 + 0.711255 w_k1_IBM_s116
                                 + 0.714928 w_k1_INTC_s116
                                 + 0.77355 w_k1_KO_s116
                                 + 0.993031 w_k1_MAT_s116
                                 + 0.745046 w_k1_MSFT_s116
                                 + 0.854273 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.71546 w_k1_ADM_s116
                                 + 0.663646 w_k1_AXP_s116
                                 + 0.981211 w_k1_DIS_s116
                                 + 0.872864 w_k1_FDX_s116
                                 + 0.931166 w_k1_IBM_s116
                                 + 1.14115 w_k1_INTC_s116
                                 + 0.961794 w_k1_KO_s116
                                 + 0.586398 w_k1_MAT_s116
                                 + 0.992603 w_k1_MSFT_s116
                                 + 0.919604 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.54603 w_k1_ADM_s117
                                 + 1.66716 w_k1_AXP_s117
                                 + 1.69596 w_k1_DIS_s117
                                 + 1.52572 w_k1_FDX_s117
                                 + 1.22707 w_k1_IBM_s117
                                 + 1.72371 w_k1_INTC_s117
                                 + 1.52115 w_k1_KO_s117 + 1.56462 w_k1_MAT_s117
                                 + 1.46181 w_k1_MSFT_s117
                                 + 1.33908 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.30634 w_k1_ADM_s117
                                 + 1.57658 w_k1_AXP_s117
                                 + 1.31188 w_k1_DIS_s117
                                 + 1.14077 w_k1_FDX_s117
                                 + 1.58568 w_k1_IBM_s117
                                 + 1.18277 w_k1_INTC_s117
                                 + 1.39049 w_k1_KO_s117 + 1.91669 w_k1_MAT_s117
                                 + 1.2435 w_k1_MSFT_s117 + 1.21353 w_k1_PG_s117
                                 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.12388 w_k1_ADM_s117
                                 + 1.06543 w_k1_AXP_s117
                                 + 1.08966 w_k1_DIS_s117 + 1.0789 w_k1_FDX_s117
                                 + 1.15107 w_k1_IBM_s117
                                 + 1.09873 w_k1_INTC_s117
                                 + 0.959053 w_k1_KO_s117
                                 + 1.15627 w_k1_MAT_s117
                                 + 1.04652 w_k1_MSFT_s117
                                 + 0.998946 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.778293 w_k1_ADM_s117
                                 + 0.775247 w_k1_AXP_s117
                                 + 0.675629 w_k1_DIS_s117
                                 + 0.731231 w_k1_FDX_s117
                                 + 1.04287 w_k1_IBM_s117
                                 + 1.09178 w_k1_INTC_s117
                                 + 0.846455 w_k1_KO_s117
                                 + 0.755687 w_k1_MAT_s117
                                 + 0.733827 w_k1_MSFT_s117
                                 + 0.886589 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.09874 w_k1_ADM_s117
                                 + 0.838043 w_k1_AXP_s117
                                 + 0.88397 w_k1_DIS_s117
                                 + 0.99002 w_k1_FDX_s117
                                 + 0.802827 w_k1_IBM_s117
                                 + 1.26342 w_k1_INTC_s117
                                 + 0.851608 w_k1_KO_s117
                                 + 0.744334 w_k1_MAT_s117
                                 + 0.983826 w_k1_MSFT_s117
                                 + 0.978412 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.09794 w_k1_ADM_s117
                                 + 0.826664 w_k1_AXP_s117
                                 + 0.868962 w_k1_DIS_s117
                                 + 0.894212 w_k1_FDX_s117
                                 + 1.35151 w_k1_IBM_s117
                                 + 1.13183 w_k1_INTC_s117
                                 + 1.01091 w_k1_KO_s117 + 1.01801 w_k1_MAT_s117
                                 + 0.87911 w_k1_MSFT_s117
                                 + 0.996165 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.22652 w_k1_ADM_s117
                                 + 1.13992 w_k1_AXP_s117
                                 + 1.22497 w_k1_DIS_s117 + 1.2312 w_k1_FDX_s117
                                 + 0.957223 w_k1_IBM_s117
                                 + 1.36678 w_k1_INTC_s117
                                 + 0.985173 w_k1_KO_s117
                                 + 0.631957 w_k1_MAT_s117
                                 + 1.12455 w_k1_MSFT_s117
                                 + 1.08233 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.78977 w_k1_ADM_s117
                                 + 0.563214 w_k1_AXP_s117
                                 + 0.686734 w_k1_DIS_s117
                                 + 0.774932 w_k1_FDX_s117
                                 + 0.763169 w_k1_IBM_s117
                                 + 1.07504 w_k1_INTC_s117
                                 + 0.836998 w_k1_KO_s117
                                 + 0.551461 w_k1_MAT_s117
                                 + 0.805206 w_k1_MSFT_s117
                                 + 0.825167 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0133389235155741 w_k1_ADM_s000
                                 + 0.0083881195927279 w_k1_ADM_s001
                                 + 0.00344863879934975 w_k1_ADM_s002
                                 + 0.00503254615492695 w_k1_ADM_s003
                                 + 0.0034071191953481 w_k1_ADM_s004
                                 + 0.00472809943973645 w_k1_ADM_s005
                                 + 0.00789391074736557 w_k1_ADM_s006
                                 + 0.00549887792256011 w_k1_ADM_s007
                                 + 0.0229054134302596 w_k1_ADM_s010
                                 + 0.0154862694381081 w_k1_ADM_s011
                                 + 0.02332374950027 w_k1_ADM_s012
                                 + 0.0122648329752484 w_k1_ADM_s013
                                 + 0.0274652419130798 w_k1_ADM_s014
                                 + 0.0124954863228183 w_k1_ADM_s015
                                 + 0.0302594185192619 w_k1_ADM_s016
                                 + 0.0172275548274005 w_k1_ADM_s017
                                 + 0.0366422627230947 w_k1_ADM_s100
                                 + 0.0422188233286551 w_k1_ADM_s101
                                 + 0.0509895372053145 w_k1_ADM_s102
                                 + 0.0282143453217118 w_k1_ADM_s103
                                 + 0.016792262983884 w_k1_ADM_s104
                                 + 0.0308334559199207 w_k1_ADM_s105
                                 + 0.0188148090430986 w_k1_ADM_s106
                                 + 0.0260689282899495 w_k1_ADM_s107
                                 + 0.0439815930179372 w_k1_ADM_s110
                                 + 0.0972791283731145 w_k1_ADM_s111
                                 + 0.0823762698202521 w_k1_ADM_s112
                                 + 0.0453362034795473 w_k1_ADM_s113
                                 + 0.070793400229203 w_k1_ADM_s114
                                 + 0.0508566732823056 w_k1_ADM_s115
                                 + 0.0693189217029216 w_k1_ADM_s116
                                 + 0.111282088664994 w_k1_ADM_s117
                                 + 0.0131020494872088 w_k1_AXP_s000
                                 + 0.00784913529323201 w_k1_AXP_s001
                                 + 0.00286188226950338 w_k1_AXP_s002
                                 + 0.00400442220281134 w_k1_AXP_s003
                                 + 0.00330795128663774 w_k1_AXP_s004
                                 + 0.00482588142530014 w_k1_AXP_s005
                                 + 0.00826745127942284 w_k1_AXP_s006
                                 + 0.0057132788204855 w_k1_AXP_s007
                                 + 0.0231428989943722 w_k1_AXP_s010
                                 + 0.0158454548237284 w_k1_AXP_s011
                                 + 0.0256435287780102 w_k1_AXP_s012
                                 + 0.0139736260031702 w_k1_AXP_s013
                                 + 0.0310910626262882 w_k1_AXP_s014
                                 + 0.0130557358242998 w_k1_AXP_s015
                                 + 0.0325843186398938 w_k1_AXP_s016
                                 + 0.0198457921442875 w_k1_AXP_s017
                                 + 0.0339628985946405 w_k1_AXP_s100
                                 + 0.0409479305750875 w_k1_AXP_s101
                                 + 0.0451515673783292 w_k1_AXP_s102
                                 + 0.0307836870009568 w_k1_AXP_s103
                                 + 0.0181891885419968 w_k1_AXP_s104
                                 + 0.0313007095506513 w_k1_AXP_s105
                                 + 0.020826002222446 w_k1_AXP_s106
                                 + 0.024538895112276 w_k1_AXP_s107
                                 + 0.0436395649572009 w_k1_AXP_s110
                                 + 0.115259075282075 w_k1_AXP_s111
                                 + 0.0727828274314184 w_k1_AXP_s112
                                 + 0.0503285966692482 w_k1_AXP_s113
                                 + 0.0844720960239321 w_k1_AXP_s114
                                 + 0.0513368946198772 w_k1_AXP_s115
                                 + 0.07206442281931 w_k1_AXP_s116
                                 + 0.0983488568913109 w_k1_AXP_s117
                                 + 0.0120298369509657 w_k1_DIS_s000
                                 + 0.00787474071107532 w_k1_DIS_s001
                                 + 0.0026027209887079 w_k1_DIS_s002
                                 + 0.00385179912979628 w_k1_DIS_s003
                                 + 0.00353875905863098 w_k1_DIS_s004
                                 + 0.00474380577255882 w_k1_DIS_s005
                                 + 0.0079999450069226 w_k1_DIS_s006
                                 + 0.00572665456837078 w_k1_DIS_s007
                                 + 0.0232131314369679 w_k1_DIS_s010
                                 + 0.0155754579489317 w_k1_DIS_s011
                                 + 0.023905152887849 w_k1_DIS_s012
                                 + 0.0145673969786557 w_k1_DIS_s013
                                 + 0.0316017720321654 w_k1_DIS_s014
                                 + 0.012812776835326 w_k1_DIS_s015
                                 + 0.0321643061955827 w_k1_DIS_s016
                                 + 0.0192612236555121 w_k1_DIS_s017
                                 + 0.0363885586725886 w_k1_DIS_s100
                                 + 0.0395312397792142 w_k1_DIS_s101
                                 + 0.0439248211409827 w_k1_DIS_s102
                                 + 0.0321609545472845 w_k1_DIS_s103
                                 + 0.0189879695588454 w_k1_DIS_s104
                                 + 0.032398869632974 w_k1_DIS_s105
                                 + 0.0202229085276197 w_k1_DIS_s106
                                 + 0.0254360114767562 w_k1_DIS_s107
                                 + 0.0468714566081262 w_k1_DIS_s110
                                 + 0.112890212472036 w_k1_DIS_s111
                                 + 0.0746915060991389 w_k1_DIS_s112
                                 + 0.0472323997089454 w_k1_DIS_s113
                                 + 0.0791674975556547 w_k1_DIS_s114
                                 + 0.0519165142848677 w_k1_DIS_s115
                                 + 0.0823142801416652 w_k1_DIS_s116
                                 + 0.101507656502867 w_k1_DIS_s117
                                 + 0.0123603444688619 w_k1_FDX_s000
                                 + 0.00789891036628214 w_k1_FDX_s001
                                 + 0.00285628954234289 w_k1_FDX_s002
                                 + 0.00386710379073742 w_k1_FDX_s003
                                 + 0.00342405469981521 w_k1_FDX_s004
                                 + 0.00495923797154876 w_k1_FDX_s005
                                 + 0.00779041617705332 w_k1_FDX_s006
                                 + 0.00607965555086285 w_k1_FDX_s007
                                 + 0.0238261552144095 w_k1_FDX_s010
                                 + 0.0152019919694486 w_k1_FDX_s011
                                 + 0.0235118921698087 w_k1_FDX_s012
                                 + 0.0130372348734776 w_k1_FDX_s013
                                 + 0.0309325324536717 w_k1_FDX_s014
                                 + 0.0127984443669918 w_k1_FDX_s015
                                 + 0.0305989252377172 w_k1_FDX_s016
                                 + 0.0186870153114845 w_k1_FDX_s017
                                 + 0.0345029389946816 w_k1_FDX_s100
                                 + 0.0360118073685528 w_k1_FDX_s101
                                 + 0.0410588590736524 w_k1_FDX_s102
                                 + 0.0301374169403519 w_k1_FDX_s103
                                 + 0.0186570346014735 w_k1_FDX_s104
                                 + 0.0322378024283091 w_k1_FDX_s105
                                 + 0.0195135801542636 w_k1_FDX_s106
                                 + 0.0241808853457004 w_k1_FDX_s107
                                 + 0.0430268429545342 w_k1_FDX_s110
                                 + 0.104217258911192 w_k1_FDX_s111
                                 + 0.0727418292318791 w_k1_FDX_s112
                                 + 0.0448513041123208 w_k1_FDX_s113
                                 + 0.0779373215296037 w_k1_FDX_s114
                                 + 0.0491397126104663 w_k1_FDX_s115
                                 + 0.0766095120074066 w_k1_FDX_s116
                                 + 0.104235733953903 w_k1_FDX_s117
                                 + 0.0129529862537363 w_k1_IBM_s000
                                 + 0.00801525658616624 w_k1_IBM_s001
                                 + 0.00298499997109259 w_k1_IBM_s002
                                 + 0.00415064249777461 w_k1_IBM_s003
                                 + 0.00378804730747181 w_k1_IBM_s004
                                 + 0.00543502338512394 w_k1_IBM_s005
                                 + 0.00802931760016095 w_k1_IBM_s006
                                 + 0.00571371185870837 w_k1_IBM_s007
                                 + 0.0230394706234718 w_k1_IBM_s010
                                 + 0.015209766846905 w_k1_IBM_s011
                                 + 0.0242762026245661 w_k1_IBM_s012
                                 + 0.0131775455505838 w_k1_IBM_s013
                                 + 0.0303547238203423 w_k1_IBM_s014
                                 + 0.0130318469263184 w_k1_IBM_s015
                                 + 0.0307938805082604 w_k1_IBM_s016
                                 + 0.0197626672182493 w_k1_IBM_s017
                                 + 0.0350378769549565 w_k1_IBM_s100
                                 + 0.0362981679153439 w_k1_IBM_s101
                                 + 0.0422147007941982 w_k1_IBM_s102
                                 + 0.0293231494466544 w_k1_IBM_s103
                                 + 0.0182272409450531 w_k1_IBM_s104
                                 + 0.0317514877424637 w_k1_IBM_s105
                                 + 0.0203108515118893 w_k1_IBM_s106
                                 + 0.0241116235426243 w_k1_IBM_s107
                                 + 0.0457176490152172 w_k1_IBM_s110
                                 + 0.105057272026199 w_k1_IBM_s111
                                 + 0.0698479262208565 w_k1_IBM_s112
                                 + 0.0469441694565777 w_k1_IBM_s113
                                 + 0.0745551136190849 w_k1_IBM_s114
                                 + 0.0494416763690724 w_k1_IBM_s115
                                 + 0.0782246718396465 w_k1_IBM_s116
                                 + 0.103583808542358 w_k1_IBM_s117
                                 + 0.0130480219297546 w_k1_INTC_s000
                                 + 0.00906304206986873 w_k1_INTC_s001
                                 + 0.00273737876713625 w_k1_INTC_s002
                                 + 0.00363929731177925 w_k1_INTC_s003
                                 + 0.00360710657647819 w_k1_INTC_s004
                                 + 0.00516282971717597 w_k1_INTC_s005
                                 + 0.00767473291082088 w_k1_INTC_s006
                                 + 0.00545538101670308 w_k1_INTC_s007
                                 + 0.0236353603496317 w_k1_INTC_s010
                                 + 0.0155662492291389 w_k1_INTC_s011
                                 + 0.0245766661854786 w_k1_INTC_s012
                                 + 0.0129428595294875 w_k1_INTC_s013
                                 + 0.0329947543076514 w_k1_INTC_s014
                                 + 0.0120369795643097 w_k1_INTC_s015
                                 + 0.0292089049482771 w_k1_INTC_s016
                                 + 0.0183876074195706 w_k1_INTC_s017
                                 + 0.0296477054455337 w_k1_INTC_s100
                                 + 0.0346955540228147 w_k1_INTC_s101
                                 + 0.0363758804923251 w_k1_INTC_s102
                                 + 0.0272939844297943 w_k1_INTC_s103
                                 + 0.0192049738709609 w_k1_INTC_s104
                                 + 0.0311215140238163 w_k1_INTC_s105
                                 + 0.0220216316754749 w_k1_INTC_s106
                                 + 0.0220082188163822 w_k1_INTC_s107
                                 + 0.039056934691551 w_k1_INTC_s110
                                 + 0.110556672622674 w_k1_INTC_s111
                                 + 0.0596695851039547 w_k1_INTC_s112
                                 + 0.0430793009684331 w_k1_INTC_s113
                                 + 0.0867162780620383 w_k1_INTC_s114
                                 + 0.044871876292312 w_k1_INTC_s115
                                 + 0.0853377972951162 w_k1_INTC_s116
                                 + 0.125437802116857 w_k1_INTC_s117
                                 + 0.0123984449015387 w_k1_KO_s000
                                 + 0.00734690107474942 w_k1_KO_s001
                                 + 0.0032999973371523 w_k1_KO_s002
                                 + 0.00372143433937784 w_k1_KO_s003
                                 + 0.00344389875510888 w_k1_KO_s004
                                 + 0.00542126258326084 w_k1_KO_s005
                                 + 0.00892146542251717 w_k1_KO_s006
                                 + 0.00561180188232053 w_k1_KO_s007
                                 + 0.0216424544698949 w_k1_KO_s010
                                 + 0.0145003324023856 w_k1_KO_s011
                                 + 0.0223270691482091 w_k1_KO_s012
                                 + 0.0126550801778363 w_k1_KO_s013
                                 + 0.031992077817247 w_k1_KO_s014
                                 + 0.0116905681365566 w_k1_KO_s015
                                 + 0.0311479037635916 w_k1_KO_s016
                                 + 0.0188731949602867 w_k1_KO_s017
                                 + 0.0359981276321596 w_k1_KO_s100
                                 + 0.0372386177000493 w_k1_KO_s101
                                 + 0.0395236883485481 w_k1_KO_s102
                                 + 0.0267970080506272 w_k1_KO_s103
                                 + 0.0186736263331698 w_k1_KO_s104
                                 + 0.0281726385480378 w_k1_KO_s105
                                 + 0.0202023486109486 w_k1_KO_s106
                                 + 0.0241955828483844 w_k1_KO_s107
                                 + 0.0420847565872183 w_k1_KO_s110
                                 + 0.0965127462778615 w_k1_KO_s111
                                 + 0.0679095046738424 w_k1_KO_s112
                                 + 0.0483620221668242 w_k1_KO_s113
                                 + 0.0774558554240546 w_k1_KO_s114
                                 + 0.0466518022631214 w_k1_KO_s115
                                 + 0.0814462963663153 w_k1_KO_s116
                                 + 0.0994901753841728 w_k1_KO_s117
                                 + 0.0111098788317598 w_k1_MAT_s000
                                 + 0.00640362712251485 w_k1_MAT_s001
                                 + 0.00347234310406375 w_k1_MAT_s002
                                 + 0.00402125612490225 w_k1_MAT_s003
                                 + 0.00410435470290811 w_k1_MAT_s004
                                 + 0.005047596001046 w_k1_MAT_s005
                                 + 0.00872020244515501 w_k1_MAT_s006
                                 + 0.00523142518968628 w_k1_MAT_s007
                                 + 0.0202230412990258 w_k1_MAT_s010
                                 + 0.0221355601307451 w_k1_MAT_s011
                                 + 0.0237717217708715 w_k1_MAT_s012
                                 + 0.0141334773881589 w_k1_MAT_s013
                                 + 0.0265985959608529 w_k1_MAT_s014
                                 + 0.0139977602921496 w_k1_MAT_s015
                                 + 0.0347404106156735 w_k1_MAT_s016
                                 + 0.0240015865030814 w_k1_MAT_s017
                                 + 0.0408210236704393 w_k1_MAT_s100
                                 + 0.0354334168243714 w_k1_MAT_s101
                                 + 0.0437209551325804 w_k1_MAT_s102
                                 + 0.0278927185725032 w_k1_MAT_s103
                                 + 0.015962642181772 w_k1_MAT_s104
                                 + 0.0279652783330048 w_k1_MAT_s105
                                 + 0.0161871181240365 w_k1_MAT_s106
                                 + 0.0201524287759796 w_k1_MAT_s107
                                 + 0.0425872989091712 w_k1_MAT_s110
                                 + 0.111487413187539 w_k1_MAT_s111
                                 + 0.0793403152808843 w_k1_MAT_s112
                                 + 0.0458464931148929 w_k1_MAT_s113
                                 + 0.0806473452458851 w_k1_MAT_s114
                                 + 0.0462893648403825 w_k1_MAT_s115
                                 + 0.0917462586589194 w_k1_MAT_s116
                                 + 0.0915650288735409 w_k1_MAT_s117
                                 + 0.0123733492732673 w_k1_MSFT_s000
                                 + 0.00808354250274674 w_k1_MSFT_s001
                                 + 0.00260694890019932 w_k1_MSFT_s002
                                 + 0.00321833488634741 w_k1_MSFT_s003
                                 + 0.00334146165121396 w_k1_MSFT_s004
                                 + 0.00444298591604542 w_k1_MSFT_s005
                                 + 0.00779477310125631 w_k1_MSFT_s006
                                 + 0.00607355184546605 w_k1_MSFT_s007
                                 + 0.0226915484361947 w_k1_MSFT_s010
                                 + 0.0158879319672546 w_k1_MSFT_s011
                                 + 0.024661797539292 w_k1_MSFT_s012
                                 + 0.0125331349879465 w_k1_MSFT_s013
                                 + 0.0304167723345949 w_k1_MSFT_s014
                                 + 0.0117571202160524 w_k1_MSFT_s015
                                 + 0.0325216910721408 w_k1_MSFT_s016
                                 + 0.0181798214826727 w_k1_MSFT_s017
                                 + 0.0317637877029416 w_k1_MSFT_s100
                                 + 0.0370614060539381 w_k1_MSFT_s101
                                 + 0.0414199734144662 w_k1_MSFT_s102
                                 + 0.0315208743156783 w_k1_MSFT_s103
                                 + 0.0180104797730343 w_k1_MSFT_s104
                                 + 0.0327404711163496 w_k1_MSFT_s105
                                 + 0.0185831837467332 w_k1_MSFT_s106
                                 + 0.0234365785072044 w_k1_MSFT_s107
                                 + 0.0434928577309105 w_k1_MSFT_s110
                                 + 0.101525429540368 w_k1_MSFT_s111
                                 + 0.06603338300429 w_k1_MSFT_s112
                                 + 0.0461002642160524 w_k1_MSFT_s113
                                 + 0.0772054710883219 w_k1_MSFT_s114
                                 + 0.047395874780251 w_k1_MSFT_s115
                                 + 0.0765155396146643 w_k1_MSFT_s116
                                 + 0.10237696495604 w_k1_MSFT_s117
                                 + 0.0125567456619859 w_k1_PG_s000
                                 + 0.0075705133883276 w_k1_PG_s001
                                 + 0.00288700146407586 w_k1_PG_s002
                                 + 0.00373042863342997 w_k1_PG_s003
                                 + 0.00343946409116568 w_k1_PG_s004
                                 + 0.00425352950720236 w_k1_PG_s005
                                 + 0.00851746221485788 w_k1_PG_s006
                                 + 0.00532334804079814 w_k1_PG_s007
                                 + 0.0221260655045824 w_k1_PG_s010
                                 + 0.0147054967419597 w_k1_PG_s011
                                 + 0.0235432789670024 w_k1_PG_s012
                                 + 0.0130161499205128 w_k1_PG_s013
                                 + 0.0289977308784679 w_k1_PG_s014
                                 + 0.0123127844877657 w_k1_PG_s015
                                 + 0.0309340568267261 w_k1_PG_s016
                                 + 0.0187670488888256 w_k1_PG_s017
                                 + 0.0354589091122066 w_k1_PG_s100
                                 + 0.0353096067944492 w_k1_PG_s101
                                 + 0.0431511927467129 w_k1_PG_s102
                                 + 0.0275874637349118 w_k1_PG_s103
                                 + 0.0164602868420198 w_k1_PG_s104
                                 + 0.0307419659134027 w_k1_PG_s105
                                 + 0.0182309956904852 w_k1_PG_s106
                                 + 0.0230185840097012 w_k1_PG_s107
                                 + 0.0424046925739993 w_k1_PG_s110
                                 + 0.10912074550069 w_k1_PG_s111
                                 + 0.073425532923087 w_k1_PG_s112
                                 + 0.0422186155687556 w_k1_PG_s113
                                 + 0.0731440470615461 w_k1_PG_s114
                                 + 0.0504606273887486 w_k1_PG_s115
                                 + 0.0787577236858031 w_k1_PG_s116
                                 + 0.102723759202775 w_k1_PG_s117 >= 1.08
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
