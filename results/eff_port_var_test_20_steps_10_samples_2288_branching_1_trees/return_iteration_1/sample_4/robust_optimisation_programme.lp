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
 w_k1_ADM_s0_return_constraint:  1.1589 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.968162 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.1401 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.841839 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.12261 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.877454 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.953278 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 0.967863 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.818773 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.793266 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.07667 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.55296 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.1465 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.05798 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.11285 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.961889 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.11241 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.26684 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.885384 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.37787 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.04982 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.695025 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 1.23494 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.1278 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.30308 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.06259 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.775592 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.845881 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.10601 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.887301 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.00735 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.01365 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.986899 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.790634 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.842148 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.664682 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.12472 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.30819 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.26101 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.84708 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.17694 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.767241 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.37206 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.972455 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.901788 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.03413 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.898092 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.742208 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.52785 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.49067 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.30839 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.20438 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.11232 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.848954 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.23271 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.39335 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.858102 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 1.46581 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.857468 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.688152 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.5226 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.26454 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.32516 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.914624 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.820835 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.731562 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.18178 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.28533 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.975785 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.14346 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.0461 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.724932 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.829263 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.583218 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.912699 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.37725 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.28362 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.845842 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.29602 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.834535 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.50087 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.942754 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.882195 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.01189 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.851602 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.666839 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.26856 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 1.7962 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.47155 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.28403 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.19193 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.788023 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.17272 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.04598 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.705245 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 1.47327 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.809423 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.551322 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.677 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.1598 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.30885 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.00862 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.708945 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.640988 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.07263 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.04356 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.07072 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.03589 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.08783 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.73702 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.694155 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.461598 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 0.844432 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.46203 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.1743 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.89716 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.15785 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.854515 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.21532 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.996761 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.979778 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.00744 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.771954 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.760056 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.05523 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.49719 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.29012 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.25579 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.1516 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 0.868893 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.10147 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 0.966886 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.825035 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 1.51323 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.04188 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.67429 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 1.53591 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.22848 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.29728 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.01581 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.714862 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.780884 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.92931 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.992215 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.945375 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.922299 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.931499 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.758898 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.756872 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.629542 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.946151 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.33127 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.21942 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.911894 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.18469 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.902316 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.464 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.901883 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.655116 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.924977 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.42225 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.71024 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.62199 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.03624 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.26923 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.719637 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.803653 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.0181 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.30304 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.72226 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.70644 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.925983 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.591056 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.984843 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 0.960237 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.777393 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.33337 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.958722 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.3281 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.815001 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.71926 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.19114 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.0094 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.35404 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.69454 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.795286 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.991384 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.681135 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.630094 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.03525 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.12272 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.924013 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.17928 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.939589 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.13977 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.951219 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.985183 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 0.879172 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.844789 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.824716 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.06482 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.41774 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.22703 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.06182 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.06479 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 0.921798 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.02618 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 0.843362 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.766659 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.31662 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.944181 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.784855 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 1.44838 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 1.21289 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.30161 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.991247 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.871938 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.881548 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.04078 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.1125 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.965857 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.984232 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.10905 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.865344 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.738356 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.743106 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 0.983466 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.18522 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.24001 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.886293 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.1626 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.82509 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.40727 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.92554 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.804466 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.960367 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.12564 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.704418 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.34844 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 1.37457 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.22452 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.863462 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.11349 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.899671 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.13779 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.31003 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.740114 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.29296 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.806985 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.844713 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.08405 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.927449 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.19894 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.910232 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.971516 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.791808 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.25146 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.932312 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.965106 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.14213 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.492 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.793845 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.889948 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.579447 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.747684 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.21475 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.26549 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.859985 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.1811 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.847747 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.55358 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.90387 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.600877 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.10318 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.28447 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.800041 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.91794 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.21913 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.3561 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.960383 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.812023 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.939015 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.17802 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.88948 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.655016 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.916482 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.617795 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.906015 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.14344 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.833674 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.38015 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.880891 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.23855 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.729646 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.59062 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.42137 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.96636 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 1.50781 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.68471 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.738718 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.04553 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.614482 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.67587 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.47956 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.10535 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.888749 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.18572 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.913441 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.15666 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.03965 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.976327 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.0329 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.835931 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.767582 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.14406 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.53699 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.35271 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.16288 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.17314 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.866688 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.0739 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.89985 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.761465 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.39183 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.944736 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.737558 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 1.61094 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.1867 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.19863 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.03583 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.909816 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.882996 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.01963 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.10648 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.966176 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.828707 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.04917 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.804352 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.725975 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.667116 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.919911 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.26825 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.13388 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.921656 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.11421 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.890065 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.20021 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.918159 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.91828 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.02667 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.922168 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.815994 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.11265 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.32191 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.20635 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.995335 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.08271 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.953795 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.15671 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.1199 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.826284 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.27485 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.967332 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.787684 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.18129 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 1.023 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.18825 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.963924 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.858345 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.848083 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.968753 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.09952 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.07345 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.05565 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.12846 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.89754 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.931861 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.756015 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 0.942381 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.25443 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.197000064627076 z_k1_s0000
                                 - 0.146000070592653 z_k1_s0001
                                 - 0.402999163824999 z_k1_s0002
                                 - 0.551000841477713 z_k1_s0003
                                 - 0.125000087660831 z_k1_s0004
                                 - 0.372999897922347 z_k1_s0005
                                 - 0.336000720509081 z_k1_s0006
                                 - 0.35300121532052 z_k1_s0007
                                 - 0.479998674220588 z_k1_s0010
                                 - 0.390000270105896 z_k1_s0011
                                 - 0.502000110457552 z_k1_s0012
                                 - 0.434000447976487 z_k1_s0013
                                 - 1.19299969888946 z_k1_s0014
                                 - 0.452001745561427 z_k1_s0015
                                 - 0.623001273252511 z_k1_s0016
                                 - 0.41700064463136 z_k1_s0017
                                 - 0.366999197512405 z_k1_s0020
                                 - 0.44399960849878 z_k1_s0021
                                 - 0.215999436308315 z_k1_s0022
                                 - 0.324998375521118 z_k1_s0023
                                 - 0.129999788607508 z_k1_s0024
                                 - 0.262999980385084 z_k1_s0025
                                 - 0.275998512528935 z_k1_s0026
                                 - 0.170999526344141 z_k1_s0027
                                 - 0.17600040182948 z_k1_s0030
                                 - 0.340000792016075 z_k1_s0031
                                 - 0.402999640038545 z_k1_s0032
                                 - 0.309000819705418 z_k1_s0033
                                 - 0.648001321917283 z_k1_s0034
                                 - 0.337000615804972 z_k1_s0035
                                 - 0.828001496560549 z_k1_s0036
                                 - 0.425001856522004 z_k1_s0037
                                 - 0.230001504616169 z_k1_s0040
                                 - 0.530001464196097 z_k1_s0041
                                 - 0.430002175671678 z_k1_s0042
                                 - 0.171000563282426 z_k1_s0043
                                 - 0.1900005328024 z_k1_s0044
                                 - 0.116000431092328 z_k1_s0045
                                 - 0.207000725952551 z_k1_s0046
                                 - 0.220000528760392 z_k1_s0047
                                 - 0.117999816312183 z_k1_s0050
                                 - 0.0690000439995127 z_k1_s0051
                                 - 0.0909999932950027 z_k1_s0052
                                 - 0.0680000109951685 z_k1_s0053
                                 - 0.0530000014300571 z_k1_s0054
                                 - 0.260999972232919 z_k1_s0055
                                 - 0.20500045438989 z_k1_s0056
                                 - 0.106000197060134 z_k1_s0057
                                 - 0.283001688846262 z_k1_s0060
                                 - 0.470000914157195 z_k1_s0061
                                 - 0.306001758879546 z_k1_s0062
                                 - 0.181000468418257 z_k1_s0063
                                 - 0.460001400665677 z_k1_s0064
                                 - 0.121000296061889 z_k1_s0065
                                 - 0.297999770080745 z_k1_s0066
                                 - 0.259000478461033 z_k1_s0067
                                 - 0.133000191203174 z_k1_s0070
                                 - 0.160000039133003 z_k1_s0071
                                 - 0.139000049743058 z_k1_s0072
                                 - 0.49500044204921 z_k1_s0073
                                 - 0.202999832836091 z_k1_s0074
                                 - 0.329000931976604 z_k1_s0075
                                 - 0.141000067189733 z_k1_s0076
                                 - 0.33800101048664 z_k1_s0077
                                 - 0.294000781203113 z_k1_s0100
                                 - 0.589001212736028 z_k1_s0101
                                 - 0.288000695219213 z_k1_s0102
                                 - 0.192000567479718 z_k1_s0103
                                 - 0.283000571565841 z_k1_s0104
                                 - 0.299000592855757 z_k1_s0105
                                 - 0.651001997236088 z_k1_s0106
                                 - 0.524002101248017 z_k1_s0107
                                 - 0.139999975287567 z_k1_s0110
                                 - 0.250999519098094 z_k1_s0111
                                 - 0.384999456166228 z_k1_s0112
                                 - 0.247000057617976 z_k1_s0113
                                 - 0.203999818962011 z_k1_s0114
                                 - 0.575000313948571 z_k1_s0115
                                 - 0.159999820685924 z_k1_s0116
                                 - 0.152999811346555 z_k1_s0117
                                 - 0.530999855846204 z_k1_s0120
                                 - 1.08999786290577 z_k1_s0121
                                 - 0.821000754029406 z_k1_s0122
                                 - 0.928998716734951 z_k1_s0123
                                 - 0.78700086290021 z_k1_s0124
                                 - 0.677001553423189 z_k1_s0125
                                 - 0.682999842741145 z_k1_s0126
                                 - 0.232999896673217 z_k1_s0127
                                 - 0.139999840912749 z_k1_s0130
                                 - 0.136999828112485 z_k1_s0131
                                 - 0.596998833854658 z_k1_s0132
                                 - 0.152999827613932 z_k1_s0133
                                 - 0.178999801016298 z_k1_s0134
                                 - 0.158999853214459 z_k1_s0135
                                 - 0.479999985043392 z_k1_s0136
                                 - 0.217999142220218 z_k1_s0137
                                 - 0.581999798259623 z_k1_s0140
                                 - 0.49100004614486 z_k1_s0141
                                 - 0.146999670439138 z_k1_s0142
                                 - 0.483998175369292 z_k1_s0143
                                 - 0.309000541915336 z_k1_s0144
                                 - 0.440998090018627 z_k1_s0145
                                 - 0.492999264511043 z_k1_s0146
                                 - 0.123999753339776 z_k1_s0147
                                 - 0.114999977471582 z_k1_s0150
                                 - 0.529999373669416 z_k1_s0151
                                 - 0.157999776841098 z_k1_s0152
                                 - 0.223999314185121 z_k1_s0153
                                 - 0.413999575537949 z_k1_s0154
                                 - 0.161999917880005 z_k1_s0155
                                 - 0.222999278925395 z_k1_s0156
                                 - 0.320000553120467 z_k1_s0157
                                 - 0.339999306310222 z_k1_s0160
                                 - 0.497999997947446 z_k1_s0161
                                 - 0.18899988738981 z_k1_s0162
                                 - 0.299999231920946 z_k1_s0163
                                 - 0.233999772177952 z_k1_s0164
                                 - 0.567999465129368 z_k1_s0165
                                 - 0.327999018793715 z_k1_s0166
                                 - 0.194999765948339 z_k1_s0167
                                 - 0.151999241084831 z_k1_s0170
                                 - 0.175999949728998 z_k1_s0171
                                 - 0.0679999090271824 z_k1_s0172
                                 - 0.252999122798718 z_k1_s0173
                                 - 0.172000308621172 z_k1_s0174
                                 - 0.289000185764968 z_k1_s0175
                                 - 0.207999940588816 z_k1_s0176
                                 - 0.112999949836249 z_k1_s0177
                                 - 0.218000539613338 z_k1_s1000
                                 - 0.266999758053997 z_k1_s1001
                                 - 0.0929999894423997 z_k1_s1002
                                 - 0.0710000067915087 z_k1_s1003
                                 - 0.0700000180437404 z_k1_s1004
                                 - 0.113999983345522 z_k1_s1005
                                 - 0.244999545203118 z_k1_s1006
                                 - 0.0729999842870453 z_k1_s1007
                                 - 0.147999831005681 z_k1_s1010
                                 - 0.687999984025831 z_k1_s1011
                                 - 0.151999802005853 z_k1_s1012
                                 - 0.138999839755336 z_k1_s1013
                                 - 0.210999769758091 z_k1_s1014
                                 - 0.184999845257059 z_k1_s1015
                                 - 0.361000829262925 z_k1_s1016
                                 - 0.376000438013782 z_k1_s1017
                                 - 0.108000079834252 z_k1_s1020
                                 - 0.241999746962079 z_k1_s1021
                                 - 0.0489999590914918 z_k1_s1022
                                 - 0.0769999952151815 z_k1_s1023
                                 - 0.0670000001638578 z_k1_s1024
                                 - 0.0499999752566186 z_k1_s1025
                                 - 0.108999679499518 z_k1_s1026
                                 - 0.131000035132307 z_k1_s1027
                                 - 0.140999588428755 z_k1_s1030
                                 - 0.195000058059146 z_k1_s1031
                                 - 0.437998842905977 z_k1_s1032
                                 - 0.32799890769283 z_k1_s1033
                                 - 0.36199895683123 z_k1_s1034
                                 - 0.147999497851671 z_k1_s1035
                                 - 0.158999491372986 z_k1_s1036
                                 - 0.130999663481896 z_k1_s1037
                                 - 0.0959996364359252 z_k1_s1040
                                 - 0.0649999975931643 z_k1_s1041
                                 - 0.134000229462918 z_k1_s1042
                                 - 0.058000041109301 z_k1_s1043
                                 - 0.0690000504696863 z_k1_s1044
                                 - 0.171000401120629 z_k1_s1045
                                 - 0.1400000366776 z_k1_s1046
                                 - 0.173999851227803 z_k1_s1047
                                 - 0.536999486963356 z_k1_s1050
                                 - 0.189000402069794 z_k1_s1051
                                 - 0.174000375448392 z_k1_s1052
                                 - 0.393000164359702 z_k1_s1053
                                 - 0.424001152339071 z_k1_s1054
                                 - 0.45299960444402 z_k1_s1055
                                 - 0.763002820224819 z_k1_s1056
                                 - 0.399001107970067 z_k1_s1057
                                 - 0.110999952630607 z_k1_s1060
                                 - 0.105999926537345 z_k1_s1061
                                 - 0.114999593045338 z_k1_s1062
                                 - 0.169000277692433 z_k1_s1063
                                 - 0.186000120229602 z_k1_s1064
                                 - 0.181999696515122 z_k1_s1065
                                 - 0.110999952630607 z_k1_s1066
                                 - 0.138999897332936 z_k1_s1067
                                 - 0.121999940871979 z_k1_s1070
                                 - 0.227000476048879 z_k1_s1071
                                 - 0.108000016181775 z_k1_s1072
                                 - 0.0780000361312903 z_k1_s1073
                                 - 0.150000450252609 z_k1_s1074
                                 - 0.223999675043562 z_k1_s1075
                                 - 0.128000442882246 z_k1_s1076
                                 - 0.0629999911060294 z_k1_s1077
                                 - 1.35200097899062 z_k1_s1100
                                 - 0.75100038364936 z_k1_s1101
                                 - 0.818000469533662 z_k1_s1102
                                 - 1.17700122596451 z_k1_s1103
                                 - 1.49000150599415 z_k1_s1104
                                 - 1.64399804073678 z_k1_s1105
                                 - 0.794999780233009 z_k1_s1106
                                 - 0.995000400177236 z_k1_s1107
                                 - 0.280001200036678 z_k1_s1110
                                 - 1.04700451001582 z_k1_s1111
                                 - 0.388001374921043 z_k1_s1112
                                 - 0.438001740785373 z_k1_s1113
                                 - 0.710002967164116 z_k1_s1114
                                 - 1.30200632287368 z_k1_s1115
                                 - 0.569000778931839 z_k1_s1116
                                 - 0.571000772346323 z_k1_s1117
                                 - 0.520998480893494 z_k1_s1120
                                 - 0.685000046804317 z_k1_s1121
                                 - 0.430001028388117 z_k1_s1122
                                 - 0.456000911532514 z_k1_s1123
                                 - 0.907998551042794 z_k1_s1124
                                 - 0.308999993016108 z_k1_s1125
                                 - 0.560000444110101 z_k1_s1126
                                 - 0.40800014026593 z_k1_s1127
                                 - 0.385000613091338 z_k1_s1130
                                 - 0.144000095769214 z_k1_s1131
                                 - 0.133999946301892 z_k1_s1132
                                 - 0.374000523917292 z_k1_s1133
                                 - 0.428000277680717 z_k1_s1134
                                 - 0.349000777249055 z_k1_s1135
                                 - 0.497000381045138 z_k1_s1136
                                 - 0.196999909825914 z_k1_s1137
                                 - 0.373998620755227 z_k1_s1140
                                 - 0.444999341556392 z_k1_s1141
                                 - 0.655998235880852 z_k1_s1142
                                 - 0.327999838540455 z_k1_s1143
                                 - 0.849000144495284 z_k1_s1144
                                 - 0.448998671714176 z_k1_s1145
                                 - 0.279999950046733 z_k1_s1146
                                 - 0.22199993465848 z_k1_s1147
                                 - 0.264000073367822 z_k1_s1150
                                 - 0.451998753914874 z_k1_s1151
                                 - 0.181999988229344 z_k1_s1152
                                 - 0.253000039135835 z_k1_s1153
                                 - 0.567000812248943 z_k1_s1154
                                 - 0.788999190494943 z_k1_s1155
                                 - 0.756998682801972 z_k1_s1156
                                 - 0.47699985198737 z_k1_s1157
                                 - 0.422001549611311 z_k1_s1160
                                 - 0.492003326342567 z_k1_s1161
                                 - 0.658000375297846 z_k1_s1162
                                 - 0.278001064849309 z_k1_s1163
                                 - 1.01400441150252 z_k1_s1164
                                 - 1.26700327065671 z_k1_s1165
                                 - 0.299000962426507 z_k1_s1166
                                 - 0.458001793002231 z_k1_s1167
                                 - 1.9439959684247 z_k1_s1170
                                 - 0.570998991403815 z_k1_s1171
                                 - 1.46299401965026 z_k1_s1172
                                 - 1.83099849968102 z_k1_s1173
                                 - 0.831998947991901 z_k1_s1174
                                 - 2.67700085799204 z_k1_s1175
                                 - 1.32499380076251 z_k1_s1176
                                 - 1.04599864368504 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.7484 w_k1_ADM_s000
                                 + 0.643722 w_k1_AXP_s000
                                 + 0.864514 w_k1_DIS_s000
                                 + 0.880009 w_k1_FDX_s000
                                 + 0.657718 w_k1_IBM_s000
                                 + 0.745096 w_k1_INTC_s000
                                 + 0.789404 w_k1_KO_s000
                                 + 0.813737 w_k1_MAT_s000
                                 + 0.859199 w_k1_MSFT_s000
                                 + 0.884142 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.15571 w_k1_ADM_s000
                                 + 1.38236 w_k1_AXP_s000
                                 + 1.38038 w_k1_DIS_s000 + 1.0767 w_k1_FDX_s000
                                 + 1.74565 w_k1_IBM_s000 + 1.024 w_k1_INTC_s000
                                 + 1.42702 w_k1_KO_s000 + 2.13196 w_k1_MAT_s000
                                 + 1.07834 w_k1_MSFT_s000 + 1.2068 w_k1_PG_s000
                                 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.33336 w_k1_ADM_s000
                                 + 1.47482 w_k1_AXP_s000
                                 + 1.07161 w_k1_DIS_s000
                                 + 1.25693 w_k1_FDX_s000
                                 + 0.930605 w_k1_IBM_s000
                                 + 0.936227 w_k1_INTC_s000
                                 + 0.949323 w_k1_KO_s000
                                 + 1.47157 w_k1_MAT_s000
                                 + 1.06719 w_k1_MSFT_s000
                                 + 1.14552 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.15314 w_k1_ADM_s000
                                 + 1.06854 w_k1_AXP_s000
                                 + 1.38747 w_k1_DIS_s000
                                 + 1.14429 w_k1_FDX_s000 + 0.9361 w_k1_IBM_s000
                                 + 1.0958 w_k1_INTC_s000 + 1.15238 w_k1_KO_s000
                                 + 1.292 w_k1_MAT_s000 + 1.26135 w_k1_MSFT_s000
                                 + 1.14606 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.994433 w_k1_ADM_s000
                                 + 1.06863 w_k1_AXP_s000
                                 + 0.829857 w_k1_DIS_s000
                                 + 0.860418 w_k1_FDX_s000
                                 + 1.24797 w_k1_IBM_s000
                                 + 0.680866 w_k1_INTC_s000
                                 + 0.851095 w_k1_KO_s000
                                 + 1.56366 w_k1_MAT_s000
                                 + 0.721621 w_k1_MSFT_s000
                                 + 0.892167 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 0.934338 w_k1_ADM_s000
                                 + 0.993085 w_k1_AXP_s000
                                 + 0.951945 w_k1_DIS_s000
                                 + 1.09622 w_k1_FDX_s000
                                 + 0.589607 w_k1_IBM_s000
                                 + 1.05447 w_k1_INTC_s000
                                 + 0.742245 w_k1_KO_s000
                                 + 0.65114 w_k1_MAT_s000
                                 + 0.991844 w_k1_MSFT_s000
                                 + 0.851922 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.874482 w_k1_ADM_s000
                                 + 0.845131 w_k1_AXP_s000
                                 + 1.03263 w_k1_DIS_s000
                                 + 0.950917 w_k1_FDX_s000
                                 + 1.23018 w_k1_IBM_s000
                                 + 0.932537 w_k1_INTC_s000
                                 + 1.11247 w_k1_KO_s000 + 1.07181 w_k1_MAT_s000
                                 + 1.07145 w_k1_MSFT_s000
                                 + 0.899637 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.95018 w_k1_ADM_s000
                                 + 0.744523 w_k1_AXP_s000
                                 + 0.596979 w_k1_DIS_s000
                                 + 0.801502 w_k1_FDX_s000
                                 + 0.784357 w_k1_IBM_s000
                                 + 0.632324 w_k1_INTC_s000
                                 + 0.703064 w_k1_KO_s000
                                 + 0.946695 w_k1_MAT_s000
                                 + 0.823477 w_k1_MSFT_s000
                                 + 0.802188 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.991254 w_k1_ADM_s001
                                 + 0.809418 w_k1_AXP_s001
                                 + 0.942065 w_k1_DIS_s001
                                 + 0.871073 w_k1_FDX_s001
                                 + 1.14541 w_k1_IBM_s001
                                 + 1.07335 w_k1_INTC_s001
                                 + 1.06796 w_k1_KO_s001 + 1.01413 w_k1_MAT_s001
                                 + 0.924558 w_k1_MSFT_s001
                                 + 0.850359 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.20435 w_k1_ADM_s001
                                 + 1.39778 w_k1_AXP_s001
                                 + 1.04713 w_k1_DIS_s001
                                 + 1.15706 w_k1_FDX_s001
                                 + 1.22527 w_k1_IBM_s001
                                 + 1.21312 w_k1_INTC_s001
                                 + 1.13342 w_k1_KO_s001 + 1.34299 w_k1_MAT_s001
                                 + 1.03665 w_k1_MSFT_s001
                                 + 1.07702 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.09653 w_k1_ADM_s001
                                 + 0.90247 w_k1_AXP_s001
                                 + 1.18614 w_k1_DIS_s001
                                 + 1.01997 w_k1_FDX_s001
                                 + 1.03808 w_k1_IBM_s001
                                 + 1.05883 w_k1_INTC_s001
                                 + 0.982112 w_k1_KO_s001
                                 + 0.650054 w_k1_MAT_s001
                                 + 1.02055 w_k1_MSFT_s001
                                 + 1.05849 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.970208 w_k1_ADM_s001
                                 + 1.27666 w_k1_AXP_s001
                                 + 1.32719 w_k1_DIS_s001 + 1.2791 w_k1_FDX_s001
                                 + 1.11773 w_k1_IBM_s001
                                 + 1.27744 w_k1_INTC_s001
                                 + 1.24359 w_k1_KO_s001 + 1.16949 w_k1_MAT_s001
                                 + 1.2912 w_k1_MSFT_s001 + 1.2743 w_k1_PG_s001
                                 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.908827 w_k1_ADM_s001
                                 + 0.959899 w_k1_AXP_s001
                                 + 0.860658 w_k1_DIS_s001
                                 + 1.01441 w_k1_FDX_s001
                                 + 0.800296 w_k1_IBM_s001
                                 + 1.02477 w_k1_INTC_s001
                                 + 0.81046 w_k1_KO_s001
                                 + 0.576967 w_k1_MAT_s001
                                 + 0.901622 w_k1_MSFT_s001
                                 + 0.844721 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.12965 w_k1_ADM_s001
                                 + 1.03762 w_k1_AXP_s001
                                 + 0.830566 w_k1_DIS_s001
                                 + 0.859044 w_k1_FDX_s001
                                 + 1.40665 w_k1_IBM_s001
                                 + 0.90912 w_k1_INTC_s001
                                 + 0.892848 w_k1_KO_s001
                                 + 1.16947 w_k1_MAT_s001
                                 + 0.813614 w_k1_MSFT_s001
                                 + 0.870894 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.772848 w_k1_ADM_s001
                                 + 0.72586 w_k1_AXP_s001
                                 + 0.632514 w_k1_DIS_s001
                                 + 0.723059 w_k1_FDX_s001
                                 + 0.818106 w_k1_IBM_s001
                                 + 1.03199 w_k1_INTC_s001
                                 + 0.769689 w_k1_KO_s001
                                 + 0.592755 w_k1_MAT_s001
                                 + 0.747978 w_k1_MSFT_s001
                                 + 0.755305 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.3768 w_k1_ADM_s001
                                 + 1.40867 w_k1_AXP_s001
                                 + 1.64211 w_k1_DIS_s001
                                 + 1.37083 w_k1_FDX_s001
                                 + 1.40705 w_k1_IBM_s001
                                 + 1.4219 w_k1_INTC_s001 + 1.47476 w_k1_KO_s001
                                 + 1.23415 w_k1_MAT_s001
                                 + 1.36496 w_k1_MSFT_s001
                                 + 1.16116 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.787629 w_k1_ADM_s002
                                 + 0.670856 w_k1_AXP_s002
                                 + 0.619386 w_k1_DIS_s002
                                 + 0.752363 w_k1_FDX_s002
                                 + 0.758925 w_k1_IBM_s002
                                 + 0.744725 w_k1_INTC_s002
                                 + 0.626885 w_k1_KO_s002
                                 + 0.865145 w_k1_MAT_s002
                                 + 0.765593 w_k1_MSFT_s002
                                 + 0.837292 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.00001 w_k1_ADM_s002
                                 + 0.939126 w_k1_AXP_s002
                                 + 0.849433 w_k1_DIS_s002
                                 + 0.997284 w_k1_FDX_s002
                                 + 0.675884 w_k1_IBM_s002
                                 + 0.983107 w_k1_INTC_s002
                                 + 0.705298 w_k1_KO_s002
                                 + 0.612926 w_k1_MAT_s002
                                 + 0.967362 w_k1_MSFT_s002
                                 + 0.925979 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.0854 w_k1_ADM_s002
                                 + 1.0673 w_k1_AXP_s002 + 0.7985 w_k1_DIS_s002
                                 + 0.960393 w_k1_FDX_s002
                                 + 1.20885 w_k1_IBM_s002
                                 + 0.957915 w_k1_INTC_s002
                                 + 0.924062 w_k1_KO_s002
                                 + 1.32192 w_k1_MAT_s002
                                 + 0.846784 w_k1_MSFT_s002
                                 + 1.00246 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.976403 w_k1_ADM_s002
                                 + 1.12371 w_k1_AXP_s002
                                 + 1.14259 w_k1_DIS_s002
                                 + 1.09556 w_k1_FDX_s002
                                 + 0.927422 w_k1_IBM_s002
                                 + 1.17482 w_k1_INTC_s002
                                 + 0.932572 w_k1_KO_s002
                                 + 1.02419 w_k1_MAT_s002
                                 + 1.16985 w_k1_MSFT_s002
                                 + 1.17379 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.875303 w_k1_ADM_s002
                                 + 1.08926 w_k1_AXP_s002
                                 + 1.12944 w_k1_DIS_s002
                                 + 1.03975 w_k1_FDX_s002
                                 + 1.42372 w_k1_IBM_s002
                                 + 1.05403 w_k1_INTC_s002
                                 + 1.12033 w_k1_KO_s002 + 1.31214 w_k1_MAT_s002
                                 + 1.12139 w_k1_MSFT_s002
                                 + 1.09276 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.23277 w_k1_ADM_s002
                                 + 1.21425 w_k1_AXP_s002 + 1.4375 w_k1_DIS_s002
                                 + 1.24723 w_k1_FDX_s002
                                 + 1.26225 w_k1_IBM_s002
                                 + 1.11082 w_k1_INTC_s002
                                 + 1.12458 w_k1_KO_s002 + 1.3009 w_k1_MAT_s002
                                 + 1.30912 w_k1_MSFT_s002 + 1.3629 w_k1_PG_s002
                                 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.868836 w_k1_ADM_s002
                                 + 0.880325 w_k1_AXP_s002
                                 + 0.836745 w_k1_DIS_s002
                                 + 0.845939 w_k1_FDX_s002
                                 + 0.984403 w_k1_IBM_s002
                                 + 1.1167 w_k1_INTC_s002 + 0.80254 w_k1_KO_s002
                                 + 1.16884 w_k1_MAT_s002
                                 + 0.821627 w_k1_MSFT_s002
                                 + 0.93848 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.05815 w_k1_ADM_s002
                                 + 1.14038 w_k1_AXP_s002
                                 + 1.01655 w_k1_DIS_s002
                                 + 0.995503 w_k1_FDX_s002
                                 + 1.7647 w_k1_IBM_s002
                                 + 1.00656 w_k1_INTC_s002
                                 + 1.26417 w_k1_KO_s002 + 1.87406 w_k1_MAT_s002
                                 + 1.07832 w_k1_MSFT_s002
                                 + 1.01242 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.734699 w_k1_ADM_s003
                                 + 0.748862 w_k1_AXP_s003
                                 + 0.53943 w_k1_DIS_s003
                                 + 0.69885 w_k1_FDX_s003
                                 + 1.05225 w_k1_IBM_s003
                                 + 0.796408 w_k1_INTC_s003
                                 + 0.750486 w_k1_KO_s003
                                 + 0.632447 w_k1_MAT_s003
                                 + 0.753398 w_k1_MSFT_s003
                                 + 0.738197 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.13247 w_k1_ADM_s003
                                 + 0.861397 w_k1_AXP_s003
                                 + 0.835506 w_k1_DIS_s003
                                 + 1.01152 w_k1_FDX_s003
                                 + 0.643524 w_k1_IBM_s003
                                 + 0.85346 w_k1_INTC_s003
                                 + 0.693005 w_k1_KO_s003
                                 + 0.448095 w_k1_MAT_s003
                                 + 0.942152 w_k1_MSFT_s003
                                 + 0.880736 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.9236 w_k1_ADM_s003
                                 + 0.906904 w_k1_AXP_s003
                                 + 0.792087 w_k1_DIS_s003
                                 + 0.901387 w_k1_FDX_s003
                                 + 0.901668 w_k1_IBM_s003
                                 + 0.825471 w_k1_INTC_s003
                                 + 0.874489 w_k1_KO_s003
                                 + 0.989383 w_k1_MAT_s003
                                 + 0.884589 w_k1_MSFT_s003
                                 + 0.871902 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.889254 w_k1_ADM_s003
                                 + 0.692266 w_k1_AXP_s003
                                 + 0.642473 w_k1_DIS_s003
                                 + 0.772984 w_k1_FDX_s003
                                 + 0.722953 w_k1_IBM_s003
                                 + 0.669733 w_k1_INTC_s003
                                 + 0.719599 w_k1_KO_s003
                                 + 0.707851 w_k1_MAT_s003
                                 + 0.745183 w_k1_MSFT_s003
                                 + 0.840194 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 0.993991 w_k1_ADM_s003
                                 + 1.0921 w_k1_AXP_s003 + 1.48947 w_k1_DIS_s003
                                 + 1.19164 w_k1_FDX_s003
                                 + 1.27324 w_k1_IBM_s003
                                 + 1.06765 w_k1_INTC_s003
                                 + 1.37037 w_k1_KO_s003 + 1.17409 w_k1_MAT_s003
                                 + 1.23414 w_k1_MSFT_s003
                                 + 1.24452 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.14576 w_k1_ADM_s003
                                 + 1.26369 w_k1_AXP_s003
                                 + 1.09693 w_k1_DIS_s003
                                 + 1.26175 w_k1_FDX_s003
                                 + 0.684611 w_k1_IBM_s003
                                 + 0.946923 w_k1_INTC_s003
                                 + 0.937174 w_k1_KO_s003 + 1.0646 w_k1_MAT_s003
                                 + 1.10504 w_k1_MSFT_s003
                                 + 1.05513 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.00376 w_k1_ADM_s003
                                 + 1.25441 w_k1_AXP_s003
                                 + 0.923805 w_k1_DIS_s003
                                 + 1.11948 w_k1_FDX_s003
                                 + 1.11228 w_k1_IBM_s003
                                 + 1.00982 w_k1_INTC_s003
                                 + 1.04535 w_k1_KO_s003 + 1.31615 w_k1_MAT_s003
                                 + 1.02849 w_k1_MSFT_s003
                                 + 1.06291 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.769029 w_k1_ADM_s003
                                 + 1.20125 w_k1_AXP_s003
                                 + 1.06419 w_k1_DIS_s003
                                 + 1.01179 w_k1_FDX_s003
                                 + 1.76976 w_k1_IBM_s003
                                 + 1.1361 w_k1_INTC_s003 + 1.37969 w_k1_KO_s003
                                 + 1.53878 w_k1_MAT_s003
                                 + 1.08907 w_k1_MSFT_s003
                                 + 1.12291 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.963397 w_k1_ADM_s004
                                 + 0.613515 w_k1_AXP_s004
                                 + 0.910621 w_k1_DIS_s004
                                 + 0.829873 w_k1_FDX_s004
                                 + 1.12004 w_k1_IBM_s004
                                 + 0.966548 w_k1_INTC_s004
                                 + 1.11573 w_k1_KO_s004
                                 + 0.783513 w_k1_MAT_s004
                                 + 1.01745 w_k1_MSFT_s004
                                 + 0.921092 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.16202 w_k1_ADM_s004
                                 + 1.08652 w_k1_AXP_s004
                                 + 1.47945 w_k1_DIS_s004
                                 + 1.27022 w_k1_FDX_s004
                                 + 1.55305 w_k1_IBM_s004
                                 + 1.09601 w_k1_INTC_s004
                                 + 1.48383 w_k1_KO_s004 + 1.75458 w_k1_MAT_s004
                                 + 1.24265 w_k1_MSFT_s004
                                 + 1.31462 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.989967 w_k1_ADM_s004
                                 + 0.922951 w_k1_AXP_s004
                                 + 0.873075 w_k1_DIS_s004
                                 + 1.06397 w_k1_FDX_s004
                                 + 1.06882 w_k1_IBM_s004
                                 + 1.1665 w_k1_INTC_s004 + 1.01532 w_k1_KO_s004
                                 + 1.27291 w_k1_MAT_s004
                                 + 0.946902 w_k1_MSFT_s004
                                 + 1.15148 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.31162 w_k1_ADM_s004
                                 + 0.898531 w_k1_AXP_s004
                                 + 0.999683 w_k1_DIS_s004
                                 + 1.10074 w_k1_FDX_s004
                                 + 0.858879 w_k1_IBM_s004
                                 + 0.911738 w_k1_INTC_s004
                                 + 0.889573 w_k1_KO_s004
                                 + 0.761768 w_k1_MAT_s004
                                 + 0.962161 w_k1_MSFT_s004
                                 + 1.06012 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.23268 w_k1_ADM_s004
                                 + 1.21869 w_k1_AXP_s004 + 1.09 w_k1_DIS_s004
                                 + 1.30564 w_k1_FDX_s004
                                 + 0.726985 w_k1_IBM_s004
                                 + 1.28709 w_k1_INTC_s004
                                 + 0.902167 w_k1_KO_s004
                                 + 0.963532 w_k1_MAT_s004
                                 + 1.18763 w_k1_MSFT_s004
                                 + 1.29822 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.903526 w_k1_ADM_s004
                                 + 0.57493 w_k1_AXP_s004
                                 + 0.546297 w_k1_DIS_s004
                                 + 0.687429 w_k1_FDX_s004
                                 + 0.596846 w_k1_IBM_s004
                                 + 0.779301 w_k1_INTC_s004
                                 + 0.716804 w_k1_KO_s004
                                 + 0.703769 w_k1_MAT_s004
                                 + 0.71773 w_k1_MSFT_s004
                                 + 0.776188 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.866507 w_k1_ADM_s004
                                 + 0.714586 w_k1_AXP_s004
                                 + 0.682116 w_k1_DIS_s004
                                 + 0.796507 w_k1_FDX_s004
                                 + 0.854172 w_k1_IBM_s004
                                 + 0.756702 w_k1_INTC_s004
                                 + 0.765297 w_k1_KO_s004
                                 + 0.795182 w_k1_MAT_s004
                                 + 0.706086 w_k1_MSFT_s004
                                 + 0.843936 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.931252 w_k1_ADM_s004
                                 + 0.764726 w_k1_AXP_s004
                                 + 0.739129 w_k1_DIS_s004
                                 + 0.894188 w_k1_FDX_s004
                                 + 0.680922 w_k1_IBM_s004
                                 + 0.898959 w_k1_INTC_s004
                                 + 0.738849 w_k1_KO_s004
                                 + 0.906397 w_k1_MAT_s004
                                 + 0.981947 w_k1_MSFT_s004
                                 + 0.994189 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.753628 w_k1_ADM_s005
                                 + 0.679467 w_k1_AXP_s005
                                 + 0.656027 w_k1_DIS_s005
                                 + 0.764037 w_k1_FDX_s005
                                 + 0.969508 w_k1_IBM_s005
                                 + 0.789278 w_k1_INTC_s005
                                 + 0.767771 w_k1_KO_s005 + 0.8846 w_k1_MAT_s005
                                 + 0.777282 w_k1_MSFT_s005
                                 + 0.855362 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.69302 w_k1_ADM_s005
                                 + 0.768313 w_k1_AXP_s005
                                 + 1.08915 w_k1_DIS_s005
                                 + 0.796942 w_k1_FDX_s005
                                 + 1.49254 w_k1_IBM_s005
                                 + 0.839093 w_k1_INTC_s005
                                 + 1.01687 w_k1_KO_s005 + 1.18781 w_k1_MAT_s005
                                 + 0.868321 w_k1_MSFT_s005
                                 + 0.879081 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.48417 w_k1_ADM_s005
                                 + 1.38489 w_k1_AXP_s005
                                 + 1.98615 w_k1_DIS_s005
                                 + 1.55628 w_k1_FDX_s005
                                 + 1.17502 w_k1_IBM_s005
                                 + 1.31001 w_k1_INTC_s005
                                 + 1.37072 w_k1_KO_s005 + 1.07364 w_k1_MAT_s005
                                 + 1.44893 w_k1_MSFT_s005
                                 + 1.17459 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.20397 w_k1_ADM_s005
                                 + 1.37717 w_k1_AXP_s005
                                 + 1.15341 w_k1_DIS_s005
                                 + 1.25647 w_k1_FDX_s005
                                 + 0.888768 w_k1_IBM_s005
                                 + 1.10554 w_k1_INTC_s005
                                 + 0.997452 w_k1_KO_s005
                                 + 0.816583 w_k1_MAT_s005
                                 + 1.15037 w_k1_MSFT_s005
                                 + 1.02771 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 0.939883 w_k1_ADM_s005
                                 + 1.06162 w_k1_AXP_s005
                                 + 1.16109 w_k1_DIS_s005
                                 + 0.992268 w_k1_FDX_s005
                                 + 1.32347 w_k1_IBM_s005
                                 + 1.20312 w_k1_INTC_s005
                                 + 1.17391 w_k1_KO_s005 + 1.33246 w_k1_MAT_s005
                                 + 1.16275 w_k1_MSFT_s005
                                 + 0.992814 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.826034 w_k1_ADM_s005
                                 + 0.619079 w_k1_AXP_s005
                                 + 0.771207 w_k1_DIS_s005
                                 + 0.861438 w_k1_FDX_s005
                                 + 0.736398 w_k1_IBM_s005
                                 + 0.80367 w_k1_INTC_s005
                                 + 0.760736 w_k1_KO_s005
                                 + 0.646039 w_k1_MAT_s005
                                 + 0.8343 w_k1_MSFT_s005 + 0.89423 w_k1_PG_s005
                                 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.94948 w_k1_ADM_s005
                                 + 1.02115 w_k1_AXP_s005
                                 + 1.05292 w_k1_DIS_s005
                                 + 1.06275 w_k1_FDX_s005
                                 + 1.08107 w_k1_IBM_s005
                                 + 0.983229 w_k1_INTC_s005
                                 + 1.04386 w_k1_KO_s005 + 1.19814 w_k1_MAT_s005
                                 + 1.03999 w_k1_MSFT_s005
                                 + 1.02585 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.982856 w_k1_ADM_s005
                                 + 1.05827 w_k1_AXP_s005
                                 + 1.54237 w_k1_DIS_s005
                                 + 1.18401 w_k1_FDX_s005
                                 + 1.01381 w_k1_IBM_s005
                                 + 1.29994 w_k1_INTC_s005
                                 + 1.05203 w_k1_KO_s005 + 1.04122 w_k1_MAT_s005
                                 + 1.14096 w_k1_MSFT_s005
                                 + 0.914958 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.40383 w_k1_ADM_s006
                                 + 1.58719 w_k1_AXP_s006
                                 + 1.24971 w_k1_DIS_s006
                                 + 1.30817 w_k1_FDX_s006
                                 + 0.875792 w_k1_IBM_s006
                                 + 1.02512 w_k1_INTC_s006
                                 + 1.03605 w_k1_KO_s006 + 1.16066 w_k1_MAT_s006
                                 + 1.05536 w_k1_MSFT_s006
                                 + 1.08865 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.905673 w_k1_ADM_s006
                                 + 0.887045 w_k1_AXP_s006
                                 + 0.965078 w_k1_DIS_s006
                                 + 0.936144 w_k1_FDX_s006
                                 + 0.92882 w_k1_IBM_s006
                                 + 0.697313 w_k1_INTC_s006
                                 + 0.897529 w_k1_KO_s006
                                 + 0.684872 w_k1_MAT_s006
                                 + 0.809896 w_k1_MSFT_s006
                                 + 0.826469 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.05512 w_k1_ADM_s006
                                 + 1.18978 w_k1_AXP_s006
                                 + 1.09559 w_k1_DIS_s006
                                 + 0.96739 w_k1_FDX_s006
                                 + 1.09234 w_k1_IBM_s006
                                 + 0.968279 w_k1_INTC_s006
                                 + 1.19593 w_k1_KO_s006 + 1.39022 w_k1_MAT_s006
                                 + 1.07124 w_k1_MSFT_s006
                                 + 1.00667 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.34319 w_k1_ADM_s006
                                 + 1.60472 w_k1_AXP_s006
                                 + 1.86935 w_k1_DIS_s006
                                 + 1.44559 w_k1_FDX_s006
                                 + 1.36992 w_k1_IBM_s006
                                 + 1.00795 w_k1_INTC_s006
                                 + 1.57151 w_k1_KO_s006 + 1.41062 w_k1_MAT_s006
                                 + 1.06229 w_k1_MSFT_s006 + 1.2204 w_k1_PG_s006
                                 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.886971 w_k1_ADM_s006
                                 + 0.793894 w_k1_AXP_s006
                                 + 0.68766 w_k1_DIS_s006
                                 + 0.778112 w_k1_FDX_s006
                                 + 0.782103 w_k1_IBM_s006
                                 + 0.71056 w_k1_INTC_s006
                                 + 0.830247 w_k1_KO_s006
                                 + 0.673919 w_k1_MAT_s006
                                 + 0.745188 w_k1_MSFT_s006
                                 + 0.813022 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.11125 w_k1_ADM_s006
                                 + 1.12931 w_k1_AXP_s006 + 1.1826 w_k1_DIS_s006
                                 + 1.05777 w_k1_FDX_s006
                                 + 1.67953 w_k1_IBM_s006
                                 + 1.02611 w_k1_INTC_s006
                                 + 1.13798 w_k1_KO_s006 + 1.53211 w_k1_MAT_s006
                                 + 0.947462 w_k1_MSFT_s006
                                 + 1.10685 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.766785 w_k1_ADM_s006
                                 + 1.01127 w_k1_AXP_s006
                                 + 0.716445 w_k1_DIS_s006
                                 + 0.74445 w_k1_FDX_s006
                                 + 1.06135 w_k1_IBM_s006
                                 + 0.751783 w_k1_INTC_s006
                                 + 0.909623 w_k1_KO_s006
                                 + 1.05971 w_k1_MAT_s006
                                 + 0.701926 w_k1_MSFT_s006
                                 + 0.83154 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.881526 w_k1_ADM_s006
                                 + 1.09575 w_k1_AXP_s006
                                 + 1.34577 w_k1_DIS_s006
                                 + 1.07936 w_k1_FDX_s006
                                 + 1.21456 w_k1_IBM_s006
                                 + 0.940077 w_k1_INTC_s006
                                 + 1.17227 w_k1_KO_s006 + 1.41549 w_k1_MAT_s006
                                 + 0.923728 w_k1_MSFT_s006
                                 + 0.951527 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.2963 w_k1_ADM_s007
                                 + 1.00511 w_k1_AXP_s007
                                 + 0.879878 w_k1_DIS_s007
                                 + 0.915484 w_k1_FDX_s007
                                 + 1.5067 w_k1_IBM_s007
                                 + 1.07564 w_k1_INTC_s007
                                 + 1.22411 w_k1_KO_s007 + 1.24889 w_k1_MAT_s007
                                 + 1.07024 w_k1_MSFT_s007
                                 + 1.07938 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.07579 w_k1_ADM_s007
                                 + 0.664798 w_k1_AXP_s007
                                 + 0.66864 w_k1_DIS_s007
                                 + 0.843948 w_k1_FDX_s007
                                 + 0.644632 w_k1_IBM_s007
                                 + 1.03904 w_k1_INTC_s007
                                 + 0.789301 w_k1_KO_s007
                                 + 0.301242 w_k1_MAT_s007
                                 + 0.873918 w_k1_MSFT_s007
                                 + 0.857172 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.04882 w_k1_ADM_s007
                                 + 0.680496 w_k1_AXP_s007
                                 + 0.988581 w_k1_DIS_s007
                                 + 0.915024 w_k1_FDX_s007
                                 + 0.944586 w_k1_IBM_s007
                                 + 0.98407 w_k1_INTC_s007
                                 + 1.04755 w_k1_KO_s007
                                 + 0.688917 w_k1_MAT_s007
                                 + 1.01796 w_k1_MSFT_s007
                                 + 0.988797 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.15777 w_k1_ADM_s007
                                 + 1.1692 w_k1_AXP_s007 + 1.37312 w_k1_DIS_s007
                                 + 1.14674 w_k1_FDX_s007
                                 + 1.19484 w_k1_IBM_s007
                                 + 1.07837 w_k1_INTC_s007
                                 + 1.37028 w_k1_KO_s007 + 1.22756 w_k1_MAT_s007
                                 + 1.18442 w_k1_MSFT_s007
                                 + 1.23058 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.962208 w_k1_ADM_s007
                                 + 0.559538 w_k1_AXP_s007
                                 + 0.663642 w_k1_DIS_s007
                                 + 0.791133 w_k1_FDX_s007
                                 + 0.91399 w_k1_IBM_s007
                                 + 0.829974 w_k1_INTC_s007
                                 + 0.927764 w_k1_KO_s007
                                 + 0.68257 w_k1_MAT_s007
                                 + 0.854663 w_k1_MSFT_s007
                                 + 0.835993 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.17732 w_k1_ADM_s007
                                 + 1.09096 w_k1_AXP_s007
                                 + 1.41065 w_k1_DIS_s007
                                 + 1.33733 w_k1_FDX_s007
                                 + 0.889803 w_k1_IBM_s007
                                 + 1.10435 w_k1_INTC_s007
                                 + 1.17311 w_k1_KO_s007
                                 + 0.906472 w_k1_MAT_s007
                                 + 1.1891 w_k1_MSFT_s007 + 1.22815 w_k1_PG_s007
                                 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.02612 w_k1_ADM_s007
                                 + 0.993666 w_k1_AXP_s007
                                 + 0.766519 w_k1_DIS_s007
                                 + 0.857677 w_k1_FDX_s007
                                 + 0.971776 w_k1_IBM_s007
                                 + 0.976495 w_k1_INTC_s007
                                 + 1.02792 w_k1_KO_s007 + 1.05518 w_k1_MAT_s007
                                 + 0.88302 w_k1_MSFT_s007
                                 + 0.996741 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.03629 w_k1_ADM_s007
                                 + 0.993286 w_k1_AXP_s007
                                 + 0.873557 w_k1_DIS_s007
                                 + 1.06424 w_k1_FDX_s007
                                 + 0.804224 w_k1_IBM_s007
                                 + 1.19901 w_k1_INTC_s007
                                 + 1.07977 w_k1_KO_s007
                                 + 0.746302 w_k1_MAT_s007
                                 + 1.12918 w_k1_MSFT_s007
                                 + 1.03646 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.831183 w_k1_ADM_s010
                                 + 0.831581 w_k1_AXP_s010
                                 + 0.719253 w_k1_DIS_s010
                                 + 0.919297 w_k1_FDX_s010
                                 + 0.697309 w_k1_IBM_s010
                                 + 0.904261 w_k1_INTC_s010
                                 + 0.672086 w_k1_KO_s010
                                 + 0.746088 w_k1_MAT_s010
                                 + 0.779544 w_k1_MSFT_s010
                                 + 0.853942 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.39775 w_k1_ADM_s010
                                 + 1.38194 w_k1_AXP_s010
                                 + 1.36605 w_k1_DIS_s010
                                 + 1.36651 w_k1_FDX_s010
                                 + 0.975954 w_k1_IBM_s010
                                 + 1.09329 w_k1_INTC_s010
                                 + 0.984087 w_k1_KO_s010
                                 + 1.17666 w_k1_MAT_s010
                                 + 1.25817 w_k1_MSFT_s010
                                 + 1.02803 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.15551 w_k1_ADM_s010
                                 + 1.40845 w_k1_AXP_s010
                                 + 1.38104 w_k1_DIS_s010
                                 + 1.15998 w_k1_FDX_s010
                                 + 1.50643 w_k1_IBM_s010
                                 + 0.948323 w_k1_INTC_s010
                                 + 1.28645 w_k1_KO_s010 + 2.09887 w_k1_MAT_s010
                                 + 1.08139 w_k1_MSFT_s010
                                 + 1.08606 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.682411 w_k1_ADM_s010
                                 + 0.604613 w_k1_AXP_s010
                                 + 0.566792 w_k1_DIS_s010
                                 + 0.688351 w_k1_FDX_s010
                                 + 0.881431 w_k1_IBM_s010
                                 + 0.796793 w_k1_INTC_s010
                                 + 0.720861 w_k1_KO_s010
                                 + 0.726614 w_k1_MAT_s010
                                 + 0.801653 w_k1_MSFT_s010
                                 + 0.830987 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.98159 w_k1_ADM_s010
                                 + 0.902449 w_k1_AXP_s010
                                 + 0.915102 w_k1_DIS_s010
                                 + 1.0458 w_k1_FDX_s010
                                 + 0.776688 w_k1_IBM_s010
                                 + 0.909079 w_k1_INTC_s010
                                 + 0.787455 w_k1_KO_s010
                                 + 0.719978 w_k1_MAT_s010
                                 + 0.943907 w_k1_MSFT_s010
                                 + 0.899243 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.02553 w_k1_ADM_s010
                                 + 0.717686 w_k1_AXP_s010
                                 + 0.926741 w_k1_DIS_s010
                                 + 0.975505 w_k1_FDX_s010
                                 + 0.714896 w_k1_IBM_s010
                                 + 0.886422 w_k1_INTC_s010
                                 + 0.813011 w_k1_KO_s010
                                 + 0.557696 w_k1_MAT_s010
                                 + 1.05456 w_k1_MSFT_s010
                                 + 0.911618 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.873218 w_k1_ADM_s010
                                 + 0.985186 w_k1_AXP_s010
                                 + 0.948801 w_k1_DIS_s010
                                 + 0.899253 w_k1_FDX_s010
                                 + 1.07371 w_k1_IBM_s010
                                 + 1.03992 w_k1_INTC_s010
                                 + 0.96493 w_k1_KO_s010 + 1.03964 w_k1_MAT_s010
                                 + 0.879202 w_k1_MSFT_s010
                                 + 0.967488 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.992231 w_k1_ADM_s010
                                 + 1.18335 w_k1_AXP_s010
                                 + 0.868233 w_k1_DIS_s010
                                 + 1.03366 w_k1_FDX_s010
                                 + 1.17551 w_k1_IBM_s010
                                 + 1.00452 w_k1_INTC_s010
                                 + 1.00331 w_k1_KO_s010 + 1.34341 w_k1_MAT_s010
                                 + 1.09222 w_k1_MSFT_s010
                                 + 0.962286 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.697285 w_k1_ADM_s011
                                 + 0.56736 w_k1_AXP_s011
                                 + 0.763449 w_k1_DIS_s011
                                 + 0.760003 w_k1_FDX_s011
                                 + 0.643337 w_k1_IBM_s011
                                 + 0.832779 w_k1_INTC_s011
                                 + 0.858339 w_k1_KO_s011
                                 + 0.628334 w_k1_MAT_s011
                                 + 0.841752 w_k1_MSFT_s011
                                 + 0.829176 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.868962 w_k1_ADM_s011
                                 + 0.707777 w_k1_AXP_s011
                                 + 0.716716 w_k1_DIS_s011
                                 + 0.824451 w_k1_FDX_s011
                                 + 0.727043 w_k1_IBM_s011
                                 + 0.744462 w_k1_INTC_s011
                                 + 0.922885 w_k1_KO_s011
                                 + 0.688214 w_k1_MAT_s011
                                 + 0.91315 w_k1_MSFT_s011
                                 + 0.933578 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.799141 w_k1_ADM_s011
                                 + 0.825265 w_k1_AXP_s011
                                 + 0.861232 w_k1_DIS_s011
                                 + 0.85413 w_k1_FDX_s011
                                 + 1.06019 w_k1_IBM_s011
                                 + 0.882339 w_k1_INTC_s011
                                 + 1.09944 w_k1_KO_s011
                                 + 0.876267 w_k1_MAT_s011
                                 + 0.990959 w_k1_MSFT_s011
                                 + 0.983403 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.997676 w_k1_ADM_s011
                                 + 0.865507 w_k1_AXP_s011
                                 + 1.13841 w_k1_DIS_s011
                                 + 1.06561 w_k1_FDX_s011
                                 + 0.764266 w_k1_IBM_s011
                                 + 1.01862 w_k1_INTC_s011
                                 + 0.996399 w_k1_KO_s011
                                 + 0.719166 w_k1_MAT_s011
                                 + 1.04755 w_k1_MSFT_s011
                                 + 0.923271 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.07109 w_k1_ADM_s011
                                 + 0.991909 w_k1_AXP_s011 + 1.43 w_k1_DIS_s011
                                 + 1.2057 w_k1_FDX_s011
                                 + 0.975105 w_k1_IBM_s011
                                 + 1.02419 w_k1_INTC_s011 + 1.3332 w_k1_KO_s011
                                 + 0.988677 w_k1_MAT_s011
                                 + 1.28683 w_k1_MSFT_s011
                                 + 1.11437 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.997071 w_k1_ADM_s011
                                 + 1.06526 w_k1_AXP_s011
                                 + 1.29165 w_k1_DIS_s011
                                 + 1.07013 w_k1_FDX_s011
                                 + 1.36271 w_k1_IBM_s011
                                 + 1.05795 w_k1_INTC_s011
                                 + 1.33372 w_k1_KO_s011 + 1.7143 w_k1_MAT_s011
                                 + 1.13679 w_k1_MSFT_s011
                                 + 1.16626 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.846358 w_k1_ADM_s011
                                 + 1.17081 w_k1_AXP_s011
                                 + 0.904111 w_k1_DIS_s011
                                 + 0.901478 w_k1_FDX_s011
                                 + 0.980936 w_k1_IBM_s011
                                 + 0.834216 w_k1_INTC_s011
                                 + 0.955147 w_k1_KO_s011 + 1.16 w_k1_MAT_s011
                                 + 0.926103 w_k1_MSFT_s011
                                 + 0.891156 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.1629 w_k1_ADM_s011
                                 + 1.28772 w_k1_AXP_s011
                                 + 1.06558 w_k1_DIS_s011
                                 + 1.01877 w_k1_FDX_s011
                                 + 1.03244 w_k1_IBM_s011
                                 + 0.950261 w_k1_INTC_s011
                                 + 1.25342 w_k1_KO_s011 + 1.32731 w_k1_MAT_s011
                                 + 1.2024 w_k1_MSFT_s011 + 1.10965 w_k1_PG_s011
                                 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 0.78562 w_k1_ADM_s012
                                 + 0.882535 w_k1_AXP_s012
                                 + 0.677418 w_k1_DIS_s012
                                 + 0.802199 w_k1_FDX_s012
                                 + 0.929934 w_k1_IBM_s012
                                 + 0.876554 w_k1_INTC_s012
                                 + 0.890622 w_k1_KO_s012
                                 + 0.90401 w_k1_MAT_s012
                                 + 0.852323 w_k1_MSFT_s012
                                 + 0.809017 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.925656 w_k1_ADM_s012
                                 + 0.898518 w_k1_AXP_s012
                                 + 0.954316 w_k1_DIS_s012
                                 + 0.925089 w_k1_FDX_s012
                                 + 1.00873 w_k1_IBM_s012
                                 + 1.15022 w_k1_INTC_s012 + 1.0914 w_k1_KO_s012
                                 + 1.0751 w_k1_MAT_s012
                                 + 1.09943 w_k1_MSFT_s012
                                 + 0.922293 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.11756 w_k1_ADM_s012
                                 + 1.11325 w_k1_AXP_s012
                                 + 0.826825 w_k1_DIS_s012
                                 + 0.991289 w_k1_FDX_s012
                                 + 0.69554 w_k1_IBM_s012
                                 + 0.747873 w_k1_INTC_s012
                                 + 0.795758 w_k1_KO_s012
                                 + 0.856949 w_k1_MAT_s012
                                 + 0.909282 w_k1_MSFT_s012
                                 + 0.844941 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.15009 w_k1_ADM_s012
                                 + 1.46067 w_k1_AXP_s012
                                 + 1.47677 w_k1_DIS_s012 + 1.3607 w_k1_FDX_s012
                                 + 1.14901 w_k1_IBM_s012
                                 + 1.0842 w_k1_INTC_s012 + 1.37631 w_k1_KO_s012
                                 + 1.66832 w_k1_MAT_s012
                                 + 1.19028 w_k1_MSFT_s012
                                 + 1.06249 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.880088 w_k1_ADM_s012
                                 + 0.660196 w_k1_AXP_s012
                                 + 0.655304 w_k1_DIS_s012
                                 + 0.735737 w_k1_FDX_s012
                                 + 0.762836 w_k1_IBM_s012
                                 + 0.834816 w_k1_INTC_s012
                                 + 0.809388 w_k1_KO_s012
                                 + 0.759385 w_k1_MAT_s012
                                 + 0.813399 w_k1_MSFT_s012
                                 + 0.828807 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.10089 w_k1_ADM_s012
                                 + 1.14813 w_k1_AXP_s012
                                 + 1.14367 w_k1_DIS_s012
                                 + 0.985848 w_k1_FDX_s012
                                 + 1.54015 w_k1_IBM_s012
                                 + 1.20223 w_k1_INTC_s012
                                 + 1.46075 w_k1_KO_s012 + 1.54585 w_k1_MAT_s012
                                 + 1.09896 w_k1_MSFT_s012
                                 + 1.10482 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.0382 w_k1_ADM_s012
                                 + 1.14516 w_k1_AXP_s012
                                 + 1.39535 w_k1_DIS_s012
                                 + 1.16003 w_k1_FDX_s012
                                 + 0.85762 w_k1_IBM_s012
                                 + 1.23774 w_k1_INTC_s012
                                 + 1.11772 w_k1_KO_s012 + 1.13483 w_k1_MAT_s012
                                 + 1.16815 w_k1_MSFT_s012
                                 + 1.05537 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.689958 w_k1_ADM_s012
                                 + 1.02046 w_k1_AXP_s012
                                 + 0.68259 w_k1_DIS_s012
                                 + 0.804694 w_k1_FDX_s012
                                 + 1.18634 w_k1_IBM_s012
                                 + 0.851284 w_k1_INTC_s012
                                 + 0.939004 w_k1_KO_s012
                                 + 1.51486 w_k1_MAT_s012
                                 + 0.858913 w_k1_MSFT_s012
                                 + 0.855041 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 0.92209 w_k1_ADM_s013
                                 + 1.20543 w_k1_AXP_s013
                                 + 1.31447 w_k1_DIS_s013
                                 + 1.20627 w_k1_FDX_s013
                                 + 1.06858 w_k1_IBM_s013
                                 + 1.42413 w_k1_INTC_s013
                                 + 1.13645 w_k1_KO_s013 + 1.3592 w_k1_MAT_s013
                                 + 1.30757 w_k1_MSFT_s013 + 1.1533 w_k1_PG_s013
                                 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.891755 w_k1_ADM_s013
                                 + 1.44668 w_k1_AXP_s013 + 1.0775 w_k1_DIS_s013
                                 + 1.39911 w_k1_FDX_s013
                                 + 0.778835 w_k1_IBM_s013
                                 + 1.41814 w_k1_INTC_s013
                                 + 0.849518 w_k1_KO_s013
                                 + 1.15933 w_k1_MAT_s013
                                 + 1.17392 w_k1_MSFT_s013
                                 + 0.981067 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.699044 w_k1_ADM_s013
                                 + 0.74365 w_k1_AXP_s013
                                 + 0.700762 w_k1_DIS_s013
                                 + 0.821737 w_k1_FDX_s013
                                 + 0.729783 w_k1_IBM_s013
                                 + 1.09319 w_k1_INTC_s013
                                 + 0.840906 w_k1_KO_s013
                                 + 0.750882 w_k1_MAT_s013
                                 + 0.920299 w_k1_MSFT_s013
                                 + 0.908486 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.02326 w_k1_ADM_s013
                                 + 1.10954 w_k1_AXP_s013
                                 + 1.12571 w_k1_DIS_s013
                                 + 0.988439 w_k1_FDX_s013
                                 + 1.8635 w_k1_IBM_s013
                                 + 1.35995 w_k1_INTC_s013
                                 + 1.31226 w_k1_KO_s013 + 1.4608 w_k1_MAT_s013
                                 + 1.16592 w_k1_MSFT_s013
                                 + 1.08482 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.18213 w_k1_ADM_s013
                                 + 1.33809 w_k1_AXP_s013
                                 + 1.30689 w_k1_DIS_s013
                                 + 1.17131 w_k1_FDX_s013
                                 + 1.37287 w_k1_IBM_s013
                                 + 1.52622 w_k1_INTC_s013 + 1.2703 w_k1_KO_s013
                                 + 1.19706 w_k1_MAT_s013
                                 + 1.41716 w_k1_MSFT_s013
                                 + 1.21508 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.11875 w_k1_ADM_s013
                                 + 1.51249 w_k1_AXP_s013
                                 + 1.85614 w_k1_DIS_s013
                                 + 1.45967 w_k1_FDX_s013
                                 + 1.45008 w_k1_IBM_s013
                                 + 1.48769 w_k1_INTC_s013
                                 + 1.27407 w_k1_KO_s013 + 1.45944 w_k1_MAT_s013
                                 + 1.51788 w_k1_MSFT_s013
                                 + 1.15034 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.665391 w_k1_ADM_s013
                                 + 0.814205 w_k1_AXP_s013
                                 + 0.869848 w_k1_DIS_s013
                                 + 0.881848 w_k1_FDX_s013
                                 + 1.11517 w_k1_IBM_s013
                                 + 1.01184 w_k1_INTC_s013
                                 + 1.05345 w_k1_KO_s013 + 1.18445 w_k1_MAT_s013
                                 + 0.926996 w_k1_MSFT_s013
                                 + 0.914433 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.926115 w_k1_ADM_s013
                                 + 1.06757 w_k1_AXP_s013
                                 + 1.11848 w_k1_DIS_s013
                                 + 1.07137 w_k1_FDX_s013
                                 + 0.823763 w_k1_IBM_s013
                                 + 1.09207 w_k1_INTC_s013 + 1.0296 w_k1_KO_s013
                                 + 1.2336 w_k1_MAT_s013
                                 + 1.31894 w_k1_MSFT_s013
                                 + 1.05638 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.16665 w_k1_ADM_s014
                                 + 1.01567 w_k1_AXP_s014
                                 + 0.964993 w_k1_DIS_s014
                                 + 1.05821 w_k1_FDX_s014
                                 + 0.722483 w_k1_IBM_s014
                                 + 1.10388 w_k1_INTC_s014
                                 + 0.836155 w_k1_KO_s014
                                 + 1.03346 w_k1_MAT_s014
                                 + 1.15773 w_k1_MSFT_s014
                                 + 0.98104 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.19061 w_k1_ADM_s014
                                 + 1.35244 w_k1_AXP_s014
                                 + 1.18886 w_k1_DIS_s014
                                 + 1.18572 w_k1_FDX_s014
                                 + 1.08953 w_k1_IBM_s014
                                 + 1.28644 w_k1_INTC_s014
                                 + 1.28677 w_k1_KO_s014 + 1.34761 w_k1_MAT_s014
                                 + 1.21029 w_k1_MSFT_s014
                                 + 1.14563 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.06087 w_k1_ADM_s014
                                 + 0.773193 w_k1_AXP_s014
                                 + 1.16448 w_k1_DIS_s014
                                 + 1.04424 w_k1_FDX_s014
                                 + 0.888611 w_k1_IBM_s014
                                 + 1.31901 w_k1_INTC_s014
                                 + 1.17018 w_k1_KO_s014
                                 + 0.952728 w_k1_MAT_s014
                                 + 1.27094 w_k1_MSFT_s014
                                 + 1.00581 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.12064 w_k1_ADM_s014
                                 + 1.25932 w_k1_AXP_s014
                                 + 1.18882 w_k1_DIS_s014
                                 + 1.17217 w_k1_FDX_s014
                                 + 1.55824 w_k1_IBM_s014
                                 + 0.987245 w_k1_INTC_s014
                                 + 1.2712 w_k1_KO_s014 + 1.54267 w_k1_MAT_s014
                                 + 1.20253 w_k1_MSFT_s014
                                 + 1.07727 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.984342 w_k1_ADM_s014
                                 + 0.790828 w_k1_AXP_s014
                                 + 0.622809 w_k1_DIS_s014
                                 + 0.851309 w_k1_FDX_s014
                                 + 0.729565 w_k1_IBM_s014
                                 + 0.998742 w_k1_INTC_s014
                                 + 0.686715 w_k1_KO_s014
                                 + 0.59147 w_k1_MAT_s014
                                 + 0.871627 w_k1_MSFT_s014
                                 + 0.878977 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.02769 w_k1_ADM_s014
                                 + 1.08909 w_k1_AXP_s014
                                 + 1.61036 w_k1_DIS_s014
                                 + 1.25955 w_k1_FDX_s014
                                 + 1.20339 w_k1_IBM_s014
                                 + 1.36444 w_k1_INTC_s014
                                 + 1.35324 w_k1_KO_s014
                                 + 0.834896 w_k1_MAT_s014
                                 + 1.35825 w_k1_MSFT_s014
                                 + 1.14887 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.754023 w_k1_ADM_s014
                                 + 0.671551 w_k1_AXP_s014
                                 + 0.677199 w_k1_DIS_s014
                                 + 0.747933 w_k1_FDX_s014
                                 + 0.875652 w_k1_IBM_s014
                                 + 0.884421 w_k1_INTC_s014
                                 + 0.762373 w_k1_KO_s014
                                 + 0.575044 w_k1_MAT_s014
                                 + 0.785203 w_k1_MSFT_s014
                                 + 0.800221 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.960594 w_k1_ADM_s014
                                 + 0.776593 w_k1_AXP_s014
                                 + 0.908183 w_k1_DIS_s014
                                 + 0.903686 w_k1_FDX_s014
                                 + 1.12886 w_k1_IBM_s014
                                 + 0.87785 w_k1_INTC_s014
                                 + 1.08101 w_k1_KO_s014 + 1.20861 w_k1_MAT_s014
                                 + 1.05767 w_k1_MSFT_s014
                                 + 0.976354 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.978834 w_k1_ADM_s015
                                 + 0.98069 w_k1_AXP_s015
                                 + 1.43798 w_k1_DIS_s015
                                 + 1.03005 w_k1_FDX_s015
                                 + 1.48718 w_k1_IBM_s015
                                 + 1.04666 w_k1_INTC_s015
                                 + 1.28439 w_k1_KO_s015 + 1.15397 w_k1_MAT_s015
                                 + 1.17169 w_k1_MSFT_s015
                                 + 1.01129 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.03518 w_k1_ADM_s015
                                 + 1.01631 w_k1_AXP_s015
                                 + 1.01733 w_k1_DIS_s015
                                 + 0.968493 w_k1_FDX_s015
                                 + 0.738812 w_k1_IBM_s015
                                 + 0.795433 w_k1_INTC_s015
                                 + 0.815667 w_k1_KO_s015
                                 + 0.813348 w_k1_MAT_s015
                                 + 0.892582 w_k1_MSFT_s015
                                 + 0.853714 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.31794 w_k1_ADM_s015
                                 + 1.55449 w_k1_AXP_s015
                                 + 1.07906 w_k1_DIS_s015
                                 + 1.18998 w_k1_FDX_s015
                                 + 1.25814 w_k1_IBM_s015
                                 + 0.916684 w_k1_INTC_s015
                                 + 1.03658 w_k1_KO_s015 + 1.94726 w_k1_MAT_s015
                                 + 0.973028 w_k1_MSFT_s015
                                 + 1.1088 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.951178 w_k1_ADM_s015
                                 + 1.04347 w_k1_AXP_s015
                                 + 1.06662 w_k1_DIS_s015
                                 + 0.894631 w_k1_FDX_s015
                                 + 1.45982 w_k1_IBM_s015
                                 + 0.815659 w_k1_INTC_s015
                                 + 1.13383 w_k1_KO_s015 + 2.19462 w_k1_MAT_s015
                                 + 0.870415 w_k1_MSFT_s015
                                 + 0.990015 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.892004 w_k1_ADM_s015
                                 + 0.698326 w_k1_AXP_s015
                                 + 0.636692 w_k1_DIS_s015
                                 + 0.744274 w_k1_FDX_s015
                                 + 0.847047 w_k1_IBM_s015
                                 + 0.799563 w_k1_INTC_s015
                                 + 0.739367 w_k1_KO_s015
                                 + 0.797175 w_k1_MAT_s015
                                 + 0.779412 w_k1_MSFT_s015
                                 + 0.812196 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.14446 w_k1_ADM_s015
                                 + 1.39825 w_k1_AXP_s015
                                 + 1.54944 w_k1_DIS_s015
                                 + 1.15436 w_k1_FDX_s015
                                 + 1.69094 w_k1_IBM_s015
                                 + 1.09501 w_k1_INTC_s015
                                 + 1.36507 w_k1_KO_s015 + 2.07518 w_k1_MAT_s015
                                 + 1.24605 w_k1_MSFT_s015
                                 + 1.03325 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.990423 w_k1_ADM_s015
                                 + 1.01973 w_k1_AXP_s015
                                 + 0.857062 w_k1_DIS_s015
                                 + 0.86739 w_k1_FDX_s015
                                 + 0.993199 w_k1_IBM_s015
                                 + 0.877601 w_k1_INTC_s015
                                 + 0.916538 w_k1_KO_s015
                                 + 1.32342 w_k1_MAT_s015
                                 + 0.882043 w_k1_MSFT_s015
                                 + 0.920657 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.39538 w_k1_ADM_s015
                                 + 1.31001 w_k1_AXP_s015
                                 + 1.59746 w_k1_DIS_s015
                                 + 1.18012 w_k1_FDX_s015
                                 + 0.97699 w_k1_IBM_s015 + 1.011 w_k1_INTC_s015
                                 + 1.10498 w_k1_KO_s015 + 1.33553 w_k1_MAT_s015
                                 + 1.19405 w_k1_MSFT_s015
                                 + 1.06668 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.06037 w_k1_ADM_s016
                                 + 0.910582 w_k1_AXP_s016
                                 + 1.04033 w_k1_DIS_s016
                                 + 1.02294 w_k1_FDX_s016
                                 + 0.961269 w_k1_IBM_s016
                                 + 1.1102 w_k1_INTC_s016 + 0.95337 w_k1_KO_s016
                                 + 0.697698 w_k1_MAT_s016
                                 + 1.06771 w_k1_MSFT_s016
                                 + 0.974195 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.16279 w_k1_ADM_s016
                                 + 1.20775 w_k1_AXP_s016
                                 + 1.06232 w_k1_DIS_s016
                                 + 1.05736 w_k1_FDX_s016
                                 + 1.05358 w_k1_IBM_s016
                                 + 1.00939 w_k1_INTC_s016
                                 + 1.00468 w_k1_KO_s016 + 1.0346 w_k1_MAT_s016
                                 + 1.05232 w_k1_MSFT_s016
                                 + 1.06132 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.64773 w_k1_ADM_s016
                                 + 0.707119 w_k1_AXP_s016
                                 + 1.01559 w_k1_DIS_s016
                                 + 0.792239 w_k1_FDX_s016
                                 + 1.12806 w_k1_IBM_s016
                                 + 0.98734 w_k1_INTC_s016
                                 + 0.983242 w_k1_KO_s016
                                 + 0.697781 w_k1_MAT_s016
                                 + 1.05492 w_k1_MSFT_s016
                                 + 0.867126 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 0.995626 w_k1_ADM_s016
                                 + 0.975826 w_k1_AXP_s016
                                 + 0.838427 w_k1_DIS_s016
                                 + 0.933134 w_k1_FDX_s016
                                 + 0.703539 w_k1_IBM_s016
                                 + 0.912712 w_k1_INTC_s016
                                 + 0.728412 w_k1_KO_s016
                                 + 0.796544 w_k1_MAT_s016
                                 + 0.954311 w_k1_MSFT_s016
                                 + 0.893734 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.954118 w_k1_ADM_s016
                                 + 1.51724 w_k1_AXP_s016
                                 + 1.44258 w_k1_DIS_s016
                                 + 1.13304 w_k1_FDX_s016
                                 + 1.78444 w_k1_IBM_s016
                                 + 1.3652 w_k1_INTC_s016 + 1.43466 w_k1_KO_s016
                                 + 1.95241 w_k1_MAT_s016
                                 + 1.01261 w_k1_MSFT_s016
                                 + 1.09668 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 0.860255 w_k1_ADM_s016
                                 + 0.753953 w_k1_AXP_s016
                                 + 0.639427 w_k1_DIS_s016
                                 + 0.743439 w_k1_FDX_s016
                                 + 0.86595 w_k1_IBM_s016
                                 + 0.780118 w_k1_INTC_s016
                                 + 0.716155 w_k1_KO_s016
                                 + 0.703168 w_k1_MAT_s016
                                 + 0.811689 w_k1_MSFT_s016
                                 + 0.738502 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.50909 w_k1_ADM_s016
                                 + 1.55502 w_k1_AXP_s016
                                 + 1.47032 w_k1_DIS_s016
                                 + 1.34834 w_k1_FDX_s016
                                 + 1.12137 w_k1_IBM_s016
                                 + 1.55254 w_k1_INTC_s016
                                 + 1.20461 w_k1_KO_s016
                                 + 0.875776 w_k1_MAT_s016
                                 + 1.21949 w_k1_MSFT_s016
                                 + 1.13135 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.847438 w_k1_ADM_s016
                                 + 1.24842 w_k1_AXP_s016
                                 + 0.79368 w_k1_DIS_s016
                                 + 0.879076 w_k1_FDX_s016
                                 + 1.25967 w_k1_IBM_s016
                                 + 0.873425 w_k1_INTC_s016
                                 + 0.968997 w_k1_KO_s016
                                 + 1.64063 w_k1_MAT_s016
                                 + 0.78333 w_k1_MSFT_s016
                                 + 0.953279 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.21868 w_k1_ADM_s017
                                 + 1.44266 w_k1_AXP_s017
                                 + 1.19868 w_k1_DIS_s017
                                 + 1.07889 w_k1_FDX_s017
                                 + 0.925989 w_k1_IBM_s017
                                 + 1.05323 w_k1_INTC_s017
                                 + 1.16027 w_k1_KO_s017 + 1.21307 w_k1_MAT_s017
                                 + 1.03318 w_k1_MSFT_s017
                                 + 1.05953 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.13796 w_k1_ADM_s017
                                 + 1.32808 w_k1_AXP_s017
                                 + 1.35411 w_k1_DIS_s017
                                 + 1.26301 w_k1_FDX_s017
                                 + 0.947213 w_k1_IBM_s017
                                 + 1.00157 w_k1_INTC_s017
                                 + 1.13488 w_k1_KO_s017 + 1.28139 w_k1_MAT_s017
                                 + 1.24444 w_k1_MSFT_s017 + 1.1915 w_k1_PG_s017
                                 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.583486 w_k1_ADM_s017
                                 + 0.990982 w_k1_AXP_s017
                                 + 0.892083 w_k1_DIS_s017
                                 + 0.860609 w_k1_FDX_s017
                                 + 1.87045 w_k1_IBM_s017
                                 + 0.96788 w_k1_INTC_s017
                                 + 1.29375 w_k1_KO_s017 + 1.70634 w_k1_MAT_s017
                                 + 0.92238 w_k1_MSFT_s017 + 1.0602 w_k1_PG_s017
                                 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.938311 w_k1_ADM_s017
                                 + 1.02758 w_k1_AXP_s017
                                 + 0.871253 w_k1_DIS_s017
                                 + 0.919096 w_k1_FDX_s017
                                 + 0.791308 w_k1_IBM_s017
                                 + 0.999036 w_k1_INTC_s017
                                 + 0.809866 w_k1_KO_s017
                                 + 0.958846 w_k1_MAT_s017
                                 + 1.035 w_k1_MSFT_s017 + 0.965838 w_k1_PG_s017
                                 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.39893 w_k1_ADM_s017
                                 + 1.38805 w_k1_AXP_s017
                                 + 1.55276 w_k1_DIS_s017
                                 + 1.17032 w_k1_FDX_s017
                                 + 1.53547 w_k1_IBM_s017
                                 + 1.06664 w_k1_INTC_s017
                                 + 1.37201 w_k1_KO_s017 + 1.57747 w_k1_MAT_s017
                                 + 1.08375 w_k1_MSFT_s017 + 1.2825 w_k1_PG_s017
                                 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.97138 w_k1_ADM_s017
                                 + 0.652624 w_k1_AXP_s017
                                 + 0.682032 w_k1_DIS_s017
                                 + 0.763916 w_k1_FDX_s017
                                 + 0.725594 w_k1_IBM_s017
                                 + 0.656075 w_k1_INTC_s017
                                 + 0.684574 w_k1_KO_s017
                                 + 0.582017 w_k1_MAT_s017
                                 + 0.718959 w_k1_MSFT_s017
                                 + 0.816069 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.806154 w_k1_ADM_s017
                                 + 0.905319 w_k1_AXP_s017
                                 + 0.962485 w_k1_DIS_s017
                                 + 0.853827 w_k1_FDX_s017
                                 + 1.25121 w_k1_IBM_s017
                                 + 0.827744 w_k1_INTC_s017
                                 + 0.963461 w_k1_KO_s017
                                 + 1.26065 w_k1_MAT_s017
                                 + 0.863606 w_k1_MSFT_s017
                                 + 1.06364 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.00239 w_k1_ADM_s017
                                 + 1.10312 w_k1_AXP_s017
                                 + 0.948379 w_k1_DIS_s017
                                 + 0.949594 w_k1_FDX_s017
                                 + 1.08421 w_k1_IBM_s017
                                 + 0.936389 w_k1_INTC_s017
                                 + 1.0015 w_k1_KO_s017 + 0.786591 w_k1_MAT_s017
                                 + 0.961575 w_k1_MSFT_s017
                                 + 1.03417 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.02067 w_k1_ADM_s100
                                 + 0.670208 w_k1_AXP_s100
                                 + 0.72527 w_k1_DIS_s100
                                 + 0.753647 w_k1_FDX_s100
                                 + 0.79987 w_k1_IBM_s100
                                 + 0.827244 w_k1_INTC_s100
                                 + 0.821785 w_k1_KO_s100
                                 + 0.440404 w_k1_MAT_s100
                                 + 0.700663 w_k1_MSFT_s100
                                 + 0.818657 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.29982 w_k1_ADM_s100
                                 + 0.894305 w_k1_AXP_s100
                                 + 0.930202 w_k1_DIS_s100
                                 + 0.944124 w_k1_FDX_s100
                                 + 0.889701 w_k1_IBM_s100
                                 + 0.83165 w_k1_INTC_s100
                                 + 1.08621 w_k1_KO_s100
                                 + 0.802714 w_k1_MAT_s100
                                 + 0.868342 w_k1_MSFT_s100
                                 + 0.982057 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.08683 w_k1_ADM_s100
                                 + 1.18146 w_k1_AXP_s100
                                 + 0.777708 w_k1_DIS_s100
                                 + 0.832868 w_k1_FDX_s100
                                 + 1.03246 w_k1_IBM_s100
                                 + 0.936415 w_k1_INTC_s100
                                 + 0.993662 w_k1_KO_s100
                                 + 1.11691 w_k1_MAT_s100
                                 + 0.771061 w_k1_MSFT_s100
                                 + 1.08701 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.11165 w_k1_ADM_s100
                                 + 0.863318 w_k1_AXP_s100
                                 + 1.13922 w_k1_DIS_s100
                                 + 1.02682 w_k1_FDX_s100
                                 + 0.672951 w_k1_IBM_s100
                                 + 1.12851 w_k1_INTC_s100
                                 + 1.01158 w_k1_KO_s100
                                 + 0.550885 w_k1_MAT_s100
                                 + 1.09124 w_k1_MSFT_s100
                                 + 0.983369 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 1.04726 w_k1_ADM_s100
                                 + 0.970584 w_k1_AXP_s100
                                 + 1.13147 w_k1_DIS_s100
                                 + 0.956243 w_k1_FDX_s100
                                 + 1.46286 w_k1_IBM_s100
                                 + 0.780451 w_k1_INTC_s100
                                 + 1.38136 w_k1_KO_s100 + 1.47047 w_k1_MAT_s100
                                 + 0.919834 w_k1_MSFT_s100
                                 + 1.18283 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.1229 w_k1_ADM_s100
                                 + 1.39449 w_k1_AXP_s100
                                 + 1.76376 w_k1_DIS_s100
                                 + 1.35194 w_k1_FDX_s100
                                 + 1.47468 w_k1_IBM_s100
                                 + 1.12739 w_k1_INTC_s100
                                 + 1.54565 w_k1_KO_s100 + 1.76198 w_k1_MAT_s100
                                 + 1.12665 w_k1_MSFT_s100
                                 + 1.28366 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.1706 w_k1_ADM_s100
                                 + 1.28292 w_k1_AXP_s100
                                 + 1.31291 w_k1_DIS_s100
                                 + 1.10922 w_k1_FDX_s100
                                 + 0.979095 w_k1_IBM_s100
                                 + 1.13397 w_k1_INTC_s100
                                 + 1.17119 w_k1_KO_s100
                                 + 0.960857 w_k1_MAT_s100
                                 + 1.03915 w_k1_MSFT_s100
                                 + 1.14575 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.821752 w_k1_ADM_s100
                                 + 0.997919 w_k1_AXP_s100
                                 + 0.957142 w_k1_DIS_s100
                                 + 0.717534 w_k1_FDX_s100
                                 + 1.54117 w_k1_IBM_s100
                                 + 0.866867 w_k1_INTC_s100
                                 + 1.24135 w_k1_KO_s100 + 1.4898 w_k1_MAT_s100
                                 + 0.811746 w_k1_MSFT_s100
                                 + 0.967142 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.696889 w_k1_ADM_s101
                                 + 0.811753 w_k1_AXP_s101
                                 + 0.627774 w_k1_DIS_s101
                                 + 0.764836 w_k1_FDX_s101
                                 + 0.93859 w_k1_IBM_s101
                                 + 0.81398 w_k1_INTC_s101
                                 + 0.882463 w_k1_KO_s101
                                 + 0.972728 w_k1_MAT_s101
                                 + 0.82842 w_k1_MSFT_s101
                                 + 0.890784 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.31038 w_k1_ADM_s101
                                 + 1.17925 w_k1_AXP_s101
                                 + 1.27435 w_k1_DIS_s101
                                 + 1.16175 w_k1_FDX_s101
                                 + 1.17826 w_k1_IBM_s101
                                 + 1.10284 w_k1_INTC_s101
                                 + 1.15664 w_k1_KO_s101 + 1.2147 w_k1_MAT_s101
                                 + 1.16364 w_k1_MSFT_s101
                                 + 1.10103 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.585807 w_k1_ADM_s101
                                 + 0.698529 w_k1_AXP_s101
                                 + 0.731864 w_k1_DIS_s101
                                 + 0.749904 w_k1_FDX_s101
                                 + 1.22285 w_k1_IBM_s101
                                 + 0.858441 w_k1_INTC_s101
                                 + 1.05819 w_k1_KO_s101 + 1.11745 w_k1_MAT_s101
                                 + 0.899891 w_k1_MSFT_s101
                                 + 0.928178 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.888188 w_k1_ADM_s101
                                 + 0.620938 w_k1_AXP_s101
                                 + 0.596235 w_k1_DIS_s101
                                 + 0.731388 w_k1_FDX_s101
                                 + 0.602207 w_k1_IBM_s101
                                 + 0.7935 w_k1_INTC_s101
                                 + 0.706764 w_k1_KO_s101
                                 + 0.599294 w_k1_MAT_s101
                                 + 0.768549 w_k1_MSFT_s101
                                 + 0.813305 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.845 w_k1_ADM_s101
                                 + 0.643804 w_k1_AXP_s101
                                 + 0.726671 w_k1_DIS_s101
                                 + 0.899694 w_k1_FDX_s101
                                 + 0.675023 w_k1_IBM_s101
                                 + 0.842631 w_k1_INTC_s101
                                 + 0.811213 w_k1_KO_s101
                                 + 0.85504 w_k1_MAT_s101
                                 + 0.985235 w_k1_MSFT_s101
                                 + 0.844434 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.92131 w_k1_ADM_s101
                                 + 1.03218 w_k1_AXP_s101
                                 + 0.723256 w_k1_DIS_s101
                                 + 0.892998 w_k1_FDX_s101
                                 + 0.94001 w_k1_IBM_s101
                                 + 0.800168 w_k1_INTC_s101
                                 + 0.901881 w_k1_KO_s101
                                 + 1.07914 w_k1_MAT_s101
                                 + 0.857652 w_k1_MSFT_s101
                                 + 0.994207 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 0.980167 w_k1_ADM_s101
                                 + 1.29862 w_k1_AXP_s101
                                 + 1.06579 w_k1_DIS_s101
                                 + 1.09716 w_k1_FDX_s101
                                 + 1.39927 w_k1_IBM_s101
                                 + 1.17058 w_k1_INTC_s101
                                 + 1.16676 w_k1_KO_s101 + 1.9734 w_k1_MAT_s101
                                 + 0.997358 w_k1_MSFT_s101
                                 + 1.18761 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.09112 w_k1_ADM_s101
                                 + 0.89114 w_k1_AXP_s101
                                 + 0.995961 w_k1_DIS_s101
                                 + 1.06672 w_k1_FDX_s101
                                 + 0.802393 w_k1_IBM_s101
                                 + 0.946483 w_k1_INTC_s101
                                 + 0.896077 w_k1_KO_s101
                                 + 0.777664 w_k1_MAT_s101
                                 + 0.990239 w_k1_MSFT_s101
                                 + 0.952007 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.15177 w_k1_ADM_s102
                                 + 1.50303 w_k1_AXP_s102
                                 + 1.15646 w_k1_DIS_s102
                                 + 1.13214 w_k1_FDX_s102
                                 + 0.923448 w_k1_IBM_s102
                                 + 1.1025 w_k1_INTC_s102 + 1.08526 w_k1_KO_s102
                                 + 0.795916 w_k1_MAT_s102
                                 + 1.18012 w_k1_MSFT_s102
                                 + 1.14562 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.09351 w_k1_ADM_s102
                                 + 0.970916 w_k1_AXP_s102
                                 + 1.20081 w_k1_DIS_s102
                                 + 1.02794 w_k1_FDX_s102
                                 + 1.18316 w_k1_IBM_s102
                                 + 0.831332 w_k1_INTC_s102
                                 + 1.12536 w_k1_KO_s102 + 1.18881 w_k1_MAT_s102
                                 + 1.06601 w_k1_MSFT_s102
                                 + 1.07292 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.09618 w_k1_ADM_s102
                                 + 0.969001 w_k1_AXP_s102
                                 + 0.75669 w_k1_DIS_s102
                                 + 0.801152 w_k1_FDX_s102
                                 + 0.9154 w_k1_IBM_s102
                                 + 0.686639 w_k1_INTC_s102
                                 + 0.98621 w_k1_KO_s102 + 1.35913 w_k1_MAT_s102
                                 + 0.890355 w_k1_MSFT_s102
                                 + 1.00999 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.695555 w_k1_ADM_s102
                                 + 0.876852 w_k1_AXP_s102
                                 + 0.621819 w_k1_DIS_s102
                                 + 0.783633 w_k1_FDX_s102
                                 + 0.743329 w_k1_IBM_s102
                                 + 0.786866 w_k1_INTC_s102
                                 + 0.729912 w_k1_KO_s102
                                 + 0.655844 w_k1_MAT_s102
                                 + 0.770175 w_k1_MSFT_s102
                                 + 0.791291 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.870645 w_k1_ADM_s102
                                 + 0.89323 w_k1_AXP_s102
                                 + 0.769812 w_k1_DIS_s102
                                 + 0.952092 w_k1_FDX_s102
                                 + 0.600281 w_k1_IBM_s102
                                 + 0.674013 w_k1_INTC_s102
                                 + 0.727544 w_k1_KO_s102
                                 + 0.719045 w_k1_MAT_s102
                                 + 0.838154 w_k1_MSFT_s102
                                 + 0.913912 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.724735 w_k1_ADM_s102
                                 + 0.733722 w_k1_AXP_s102
                                 + 0.760607 w_k1_DIS_s102
                                 + 0.734429 w_k1_FDX_s102
                                 + 0.901377 w_k1_IBM_s102
                                 + 0.765926 w_k1_INTC_s102
                                 + 0.906721 w_k1_KO_s102
                                 + 0.812463 w_k1_MAT_s102
                                 + 0.849984 w_k1_MSFT_s102
                                 + 0.843047 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 1.12991 w_k1_ADM_s102
                                 + 1.44266 w_k1_AXP_s102
                                 + 1.21799 w_k1_DIS_s102
                                 + 1.12581 w_k1_FDX_s102
                                 + 1.57194 w_k1_IBM_s102
                                 + 1.16268 w_k1_INTC_s102
                                 + 1.42594 w_k1_KO_s102 + 2.32814 w_k1_MAT_s102
                                 + 1.2922 w_k1_MSFT_s102 + 1.18354 w_k1_PG_s102
                                 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.01925 w_k1_ADM_s102
                                 + 1.02645 w_k1_AXP_s102
                                 + 0.969571 w_k1_DIS_s102
                                 + 0.986197 w_k1_FDX_s102
                                 + 0.819633 w_k1_IBM_s102
                                 + 0.789656 w_k1_INTC_s102
                                 + 0.959327 w_k1_KO_s102
                                 + 0.94386 w_k1_MAT_s102
                                 + 1.17324 w_k1_MSFT_s102
                                 + 0.924735 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.23422 w_k1_ADM_s103
                                 + 1.28701 w_k1_AXP_s103
                                 + 1.59218 w_k1_DIS_s103
                                 + 1.31336 w_k1_FDX_s103
                                 + 1.15217 w_k1_IBM_s103
                                 + 1.35747 w_k1_INTC_s103
                                 + 1.54069 w_k1_KO_s103
                                 + 0.830432 w_k1_MAT_s103
                                 + 1.44748 w_k1_MSFT_s103
                                 + 1.28415 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.813171 w_k1_ADM_s103
                                 + 0.696366 w_k1_AXP_s103
                                 + 0.614129 w_k1_DIS_s103
                                 + 0.756394 w_k1_FDX_s103
                                 + 0.686552 w_k1_IBM_s103
                                 + 0.747911 w_k1_INTC_s103
                                 + 0.81871 w_k1_KO_s103
                                 + 0.755183 w_k1_MAT_s103
                                 + 0.823592 w_k1_MSFT_s103
                                 + 0.883402 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.01854 w_k1_ADM_s103
                                 + 0.935745 w_k1_AXP_s103
                                 + 1.04219 w_k1_DIS_s103
                                 + 1.03068 w_k1_FDX_s103
                                 + 0.784061 w_k1_IBM_s103
                                 + 1.03499 w_k1_INTC_s103
                                 + 0.936039 w_k1_KO_s103
                                 + 0.791967 w_k1_MAT_s103
                                 + 0.998998 w_k1_MSFT_s103
                                 + 0.948809 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.971384 w_k1_ADM_s103
                                 + 1.00816 w_k1_AXP_s103
                                 + 1.13755 w_k1_DIS_s103
                                 + 0.961569 w_k1_FDX_s103
                                 + 1.2234 w_k1_IBM_s103
                                 + 1.02374 w_k1_INTC_s103
                                 + 1.14448 w_k1_KO_s103 + 1.08911 w_k1_MAT_s103
                                 + 0.965487 w_k1_MSFT_s103
                                 + 0.97293 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.901808 w_k1_ADM_s103
                                 + 0.868215 w_k1_AXP_s103
                                 + 0.763855 w_k1_DIS_s103
                                 + 0.840636 w_k1_FDX_s103
                                 + 0.916862 w_k1_IBM_s103
                                 + 0.868309 w_k1_INTC_s103
                                 + 0.927769 w_k1_KO_s103
                                 + 0.881421 w_k1_MAT_s103
                                 + 0.853242 w_k1_MSFT_s103
                                 + 0.987754 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 1.12959 w_k1_ADM_s103
                                 + 1.3746 w_k1_AXP_s103 + 1.54374 w_k1_DIS_s103
                                 + 1.21368 w_k1_FDX_s103
                                 + 1.59561 w_k1_IBM_s103
                                 + 1.17041 w_k1_INTC_s103
                                 + 1.64903 w_k1_KO_s103 + 1.55657 w_k1_MAT_s103
                                 + 1.13285 w_k1_MSFT_s103
                                 + 1.19976 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 1.07095 w_k1_ADM_s103
                                 + 1.30541 w_k1_AXP_s103
                                 + 0.914583 w_k1_DIS_s103
                                 + 1.05045 w_k1_FDX_s103
                                 + 1.15212 w_k1_IBM_s103
                                 + 1.21364 w_k1_INTC_s103
                                 + 1.12662 w_k1_KO_s103 + 1.19436 w_k1_MAT_s103
                                 + 1.12272 w_k1_MSFT_s103
                                 + 1.20135 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.820891 w_k1_ADM_s103
                                 + 0.705174 w_k1_AXP_s103
                                 + 0.849574 w_k1_DIS_s103
                                 + 0.738526 w_k1_FDX_s103
                                 + 1.23071 w_k1_IBM_s103
                                 + 0.83944 w_k1_INTC_s103
                                 + 0.984633 w_k1_KO_s103
                                 + 0.933395 w_k1_MAT_s103
                                 + 0.963884 w_k1_MSFT_s103
                                 + 0.983676 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 0.928737 w_k1_ADM_s104
                                 + 0.721505 w_k1_AXP_s104
                                 + 1.03086 w_k1_DIS_s104
                                 + 0.94258 w_k1_FDX_s104
                                 + 1.34895 w_k1_IBM_s104
                                 + 0.951033 w_k1_INTC_s104
                                 + 1.04108 w_k1_KO_s104 + 1.24182 w_k1_MAT_s104
                                 + 0.88138 w_k1_MSFT_s104
                                 + 0.981363 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.886194 w_k1_ADM_s104
                                 + 0.962176 w_k1_AXP_s104
                                 + 0.768442 w_k1_DIS_s104
                                 + 0.851692 w_k1_FDX_s104
                                 + 1.09507 w_k1_IBM_s104
                                 + 0.895781 w_k1_INTC_s104
                                 + 0.9891 w_k1_KO_s104 + 1.28599 w_k1_MAT_s104
                                 + 0.884832 w_k1_MSFT_s104
                                 + 0.982375 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.17227 w_k1_ADM_s104
                                 + 1.1288 w_k1_AXP_s104 + 1.14855 w_k1_DIS_s104
                                 + 1.20708 w_k1_FDX_s104
                                 + 1.08467 w_k1_IBM_s104
                                 + 1.14618 w_k1_INTC_s104
                                 + 1.08479 w_k1_KO_s104 + 1.0177 w_k1_MAT_s104
                                 + 1.04337 w_k1_MSFT_s104
                                 + 1.20162 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.868837 w_k1_ADM_s104
                                 + 1.5696 w_k1_AXP_s104 + 1.45949 w_k1_DIS_s104
                                 + 1.22874 w_k1_FDX_s104
                                 + 1.68438 w_k1_IBM_s104
                                 + 1.27635 w_k1_INTC_s104
                                 + 1.31749 w_k1_KO_s104 + 1.99555 w_k1_MAT_s104
                                 + 1.16671 w_k1_MSFT_s104
                                 + 1.20804 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 1.32571 w_k1_ADM_s104
                                 + 0.996213 w_k1_AXP_s104
                                 + 1.38624 w_k1_DIS_s104
                                 + 1.17938 w_k1_FDX_s104
                                 + 1.16312 w_k1_IBM_s104
                                 + 1.13881 w_k1_INTC_s104
                                 + 1.31585 w_k1_KO_s104
                                 + 0.972374 w_k1_MAT_s104
                                 + 1.16406 w_k1_MSFT_s104
                                 + 1.21556 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.885284 w_k1_ADM_s104
                                 + 1.12369 w_k1_AXP_s104 + 1.0495 w_k1_DIS_s104
                                 + 1.17555 w_k1_FDX_s104
                                 + 0.902072 w_k1_IBM_s104
                                 + 1.07168 w_k1_INTC_s104
                                 + 0.936181 w_k1_KO_s104
                                 + 1.02287 w_k1_MAT_s104
                                 + 1.02083 w_k1_MSFT_s104
                                 + 1.00511 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.704737 w_k1_ADM_s104
                                 + 0.725439 w_k1_AXP_s104
                                 + 0.567796 w_k1_DIS_s104
                                 + 0.781785 w_k1_FDX_s104
                                 + 0.842843 w_k1_IBM_s104
                                 + 0.789674 w_k1_INTC_s104
                                 + 0.673078 w_k1_KO_s104
                                 + 0.899331 w_k1_MAT_s104
                                 + 0.734138 w_k1_MSFT_s104
                                 + 0.91313 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.03401 w_k1_ADM_s104
                                 + 1.02152 w_k1_AXP_s104
                                 + 0.920904 w_k1_DIS_s104
                                 + 1.06482 w_k1_FDX_s104
                                 + 0.715884 w_k1_IBM_s104
                                 + 0.856893 w_k1_INTC_s104
                                 + 0.805034 w_k1_KO_s104
                                 + 0.715828 w_k1_MAT_s104
                                 + 0.875612 w_k1_MSFT_s104
                                 + 1.04187 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.19241 w_k1_ADM_s105
                                 + 1.17364 w_k1_AXP_s105
                                 + 1.28814 w_k1_DIS_s105
                                 + 1.13557 w_k1_FDX_s105
                                 + 1.13539 w_k1_IBM_s105
                                 + 1.24929 w_k1_INTC_s105
                                 + 1.17777 w_k1_KO_s105 + 1.23873 w_k1_MAT_s105
                                 + 1.14855 w_k1_MSFT_s105
                                 + 1.12047 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.19192 w_k1_ADM_s105
                                 + 1.47382 w_k1_AXP_s105
                                 + 1.66415 w_k1_DIS_s105
                                 + 1.51645 w_k1_FDX_s105
                                 + 0.978885 w_k1_IBM_s105 + 1.13 w_k1_INTC_s105
                                 + 1.25457 w_k1_KO_s105 + 1.29281 w_k1_MAT_s105
                                 + 1.35903 w_k1_MSFT_s105
                                 + 1.09912 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.36287 w_k1_ADM_s105
                                 + 1.1048 w_k1_AXP_s105 + 1.11217 w_k1_DIS_s105
                                 + 1.24391 w_k1_FDX_s105
                                 + 0.654045 w_k1_IBM_s105
                                 + 1.01434 w_k1_INTC_s105
                                 + 0.728403 w_k1_KO_s105
                                 + 0.639694 w_k1_MAT_s105
                                 + 1.01143 w_k1_MSFT_s105
                                 + 0.943081 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.0214 w_k1_ADM_s105
                                 + 1.03813 w_k1_AXP_s105
                                 + 1.28801 w_k1_DIS_s105
                                 + 1.10577 w_k1_FDX_s105
                                 + 0.91703 w_k1_IBM_s105
                                 + 1.15327 w_k1_INTC_s105
                                 + 0.885172 w_k1_KO_s105
                                 + 0.729015 w_k1_MAT_s105
                                 + 1.16223 w_k1_MSFT_s105 + 1.126 w_k1_PG_s105
                                 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.986322 w_k1_ADM_s105
                                 + 1.33464 w_k1_AXP_s105
                                 + 1.07442 w_k1_DIS_s105
                                 + 1.07905 w_k1_FDX_s105
                                 + 1.31792 w_k1_IBM_s105
                                 + 1.08361 w_k1_INTC_s105
                                 + 1.08907 w_k1_KO_s105 + 1.4308 w_k1_MAT_s105
                                 + 0.928545 w_k1_MSFT_s105
                                 + 1.10975 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.680711 w_k1_ADM_s105
                                 + 0.984499 w_k1_AXP_s105
                                 + 0.858254 w_k1_DIS_s105
                                 + 0.848713 w_k1_FDX_s105
                                 + 1.51033 w_k1_IBM_s105
                                 + 0.923209 w_k1_INTC_s105
                                 + 0.952634 w_k1_KO_s105
                                 + 1.32643 w_k1_MAT_s105
                                 + 0.870001 w_k1_MSFT_s105
                                 + 0.936637 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.674605 w_k1_ADM_s105
                                 + 0.811763 w_k1_AXP_s105
                                 + 0.638023 w_k1_DIS_s105
                                 + 0.804175 w_k1_FDX_s105
                                 + 0.763362 w_k1_IBM_s105
                                 + 0.814488 w_k1_INTC_s105
                                 + 0.734186 w_k1_KO_s105
                                 + 0.729533 w_k1_MAT_s105
                                 + 0.835205 w_k1_MSFT_s105
                                 + 0.816459 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.86617 w_k1_ADM_s105
                                 + 1.2046 w_k1_AXP_s105 + 1.04384 w_k1_DIS_s105
                                 + 1.05886 w_k1_FDX_s105
                                 + 0.832882 w_k1_IBM_s105
                                 + 0.963542 w_k1_INTC_s105
                                 + 0.868918 w_k1_KO_s105
                                 + 1.04308 w_k1_MAT_s105
                                 + 1.02814 w_k1_MSFT_s105
                                 + 0.998433 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.867003 w_k1_ADM_s106
                                 + 1.08037 w_k1_AXP_s106
                                 + 1.07991 w_k1_DIS_s106
                                 + 1.17431 w_k1_FDX_s106
                                 + 0.787557 w_k1_IBM_s106
                                 + 1.10375 w_k1_INTC_s106
                                 + 0.920738 w_k1_KO_s106
                                 + 0.806751 w_k1_MAT_s106
                                 + 1.09187 w_k1_MSFT_s106 + 1.1087 w_k1_PG_s106
                                 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.961993 w_k1_ADM_s106
                                 + 0.945483 w_k1_AXP_s106
                                 + 1.11636 w_k1_DIS_s106
                                 + 1.06576 w_k1_FDX_s106
                                 + 0.851526 w_k1_IBM_s106
                                 + 1.11214 w_k1_INTC_s106
                                 + 0.936344 w_k1_KO_s106
                                 + 0.737206 w_k1_MAT_s106
                                 + 1.12527 w_k1_MSFT_s106
                                 + 1.10324 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.18999 w_k1_ADM_s106
                                 + 1.56505 w_k1_AXP_s106 + 1.505 w_k1_DIS_s106
                                 + 1.25478 w_k1_FDX_s106
                                 + 1.00498 w_k1_IBM_s106
                                 + 1.24087 w_k1_INTC_s106
                                 + 1.20992 w_k1_KO_s106 + 1.28201 w_k1_MAT_s106
                                 + 1.34962 w_k1_MSFT_s106
                                 + 1.17337 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.877622 w_k1_ADM_s106
                                 + 0.863991 w_k1_AXP_s106
                                 + 0.709775 w_k1_DIS_s106
                                 + 0.81133 w_k1_FDX_s106
                                 + 1.04929 w_k1_IBM_s106
                                 + 0.99111 w_k1_INTC_s106 + 1.0443 w_k1_KO_s106
                                 + 1.28889 w_k1_MAT_s106
                                 + 0.98043 w_k1_MSFT_s106
                                 + 1.17764 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.710586 w_k1_ADM_s106
                                 + 1.04186 w_k1_AXP_s106
                                 + 1.19468 w_k1_DIS_s106
                                 + 1.01018 w_k1_FDX_s106
                                 + 1.21473 w_k1_IBM_s106
                                 + 0.926098 w_k1_INTC_s106
                                 + 1.11481 w_k1_KO_s106 + 1.58588 w_k1_MAT_s106
                                 + 1.11256 w_k1_MSFT_s106
                                 + 1.19246 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.583034 w_k1_ADM_s106
                                 + 0.665163 w_k1_AXP_s106
                                 + 0.593084 w_k1_DIS_s106
                                 + 0.762069 w_k1_FDX_s106
                                 + 0.698839 w_k1_IBM_s106
                                 + 0.888629 w_k1_INTC_s106
                                 + 0.738844 w_k1_KO_s106
                                 + 0.718108 w_k1_MAT_s106
                                 + 0.787755 w_k1_MSFT_s106
                                 + 0.976548 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.954878 w_k1_ADM_s106
                                 + 1.22907 w_k1_AXP_s106
                                 + 1.40452 w_k1_DIS_s106
                                 + 1.11418 w_k1_FDX_s106
                                 + 1.75304 w_k1_IBM_s106
                                 + 1.05254 w_k1_INTC_s106
                                 + 1.45329 w_k1_KO_s106 + 1.50336 w_k1_MAT_s106
                                 + 1.13371 w_k1_MSFT_s106
                                 + 1.26072 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.768691 w_k1_ADM_s106
                                 + 0.897189 w_k1_AXP_s106
                                 + 0.891967 w_k1_DIS_s106
                                 + 0.896638 w_k1_FDX_s106
                                 + 0.871757 w_k1_IBM_s106
                                 + 0.936138 w_k1_INTC_s106
                                 + 0.935561 w_k1_KO_s106
                                 + 0.865727 w_k1_MAT_s106
                                 + 0.912302 w_k1_MSFT_s106
                                 + 0.966633 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.14255 w_k1_ADM_s107
                                 + 0.995176 w_k1_AXP_s107
                                 + 0.740466 w_k1_DIS_s107
                                 + 0.89544 w_k1_FDX_s107
                                 + 1.16597 w_k1_IBM_s107
                                 + 0.7366 w_k1_INTC_s107
                                 + 0.983344 w_k1_KO_s107
                                 + 1.12196 w_k1_MAT_s107
                                 + 0.710875 w_k1_MSFT_s107
                                 + 0.85992 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.21937 w_k1_ADM_s107
                                 + 1.12692 w_k1_AXP_s107
                                 + 1.17305 w_k1_DIS_s107
                                 + 1.10107 w_k1_FDX_s107
                                 + 1.03316 w_k1_IBM_s107
                                 + 0.94552 w_k1_INTC_s107
                                 + 1.27322 w_k1_KO_s107 + 1.05172 w_k1_MAT_s107
                                 + 0.913413 w_k1_MSFT_s107
                                 + 0.989466 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.10844 w_k1_ADM_s107
                                 + 0.85431 w_k1_AXP_s107 + 1.5002 w_k1_DIS_s107
                                 + 1.07959 w_k1_FDX_s107
                                 + 1.17957 w_k1_IBM_s107
                                 + 1.10842 w_k1_INTC_s107
                                 + 1.47738 w_k1_KO_s107 + 0.78372 w_k1_MAT_s107
                                 + 1.29074 w_k1_MSFT_s107
                                 + 1.13626 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.59751 w_k1_ADM_s107
                                 + 1.42653 w_k1_AXP_s107
                                 + 1.85729 w_k1_DIS_s107
                                 + 1.55432 w_k1_FDX_s107
                                 + 1.15605 w_k1_IBM_s107
                                 + 1.25319 w_k1_INTC_s107
                                 + 1.54341 w_k1_KO_s107
                                 + 0.884848 w_k1_MAT_s107
                                 + 1.41159 w_k1_MSFT_s107
                                 + 1.14163 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.10112 w_k1_ADM_s107
                                 + 0.965761 w_k1_AXP_s107
                                 + 0.912541 w_k1_DIS_s107
                                 + 1.0417 w_k1_FDX_s107
                                 + 0.730178 w_k1_IBM_s107
                                 + 0.897261 w_k1_INTC_s107
                                 + 0.997905 w_k1_KO_s107
                                 + 0.819723 w_k1_MAT_s107
                                 + 0.928771 w_k1_MSFT_s107
                                 + 0.835918 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.766998 w_k1_ADM_s107
                                 + 0.734569 w_k1_AXP_s107
                                 + 0.677621 w_k1_DIS_s107
                                 + 0.770746 w_k1_FDX_s107
                                 + 0.817335 w_k1_IBM_s107
                                 + 0.881858 w_k1_INTC_s107
                                 + 0.863984 w_k1_KO_s107
                                 + 0.655834 w_k1_MAT_s107
                                 + 0.801096 w_k1_MSFT_s107
                                 + 0.76125 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.904238 w_k1_ADM_s107
                                 + 1.08895 w_k1_AXP_s107
                                 + 1.18611 w_k1_DIS_s107
                                 + 1.06582 w_k1_FDX_s107
                                 + 1.41151 w_k1_IBM_s107
                                 + 0.980186 w_k1_INTC_s107
                                 + 1.3869 w_k1_KO_s107 + 1.47479 w_k1_MAT_s107
                                 + 1.03269 w_k1_MSFT_s107
                                 + 0.956434 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.22641 w_k1_ADM_s107
                                 + 0.815012 w_k1_AXP_s107
                                 + 1.26416 w_k1_DIS_s107
                                 + 1.20285 w_k1_FDX_s107
                                 + 0.835372 w_k1_IBM_s107
                                 + 1.11862 w_k1_INTC_s107
                                 + 1.13362 w_k1_KO_s107
                                 + 0.732162 w_k1_MAT_s107
                                 + 1.04985 w_k1_MSFT_s107
                                 + 0.943493 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.977605 w_k1_ADM_s110
                                 + 0.86087 w_k1_AXP_s110
                                 + 0.810235 w_k1_DIS_s110
                                 + 0.926445 w_k1_FDX_s110
                                 + 0.939365 w_k1_IBM_s110
                                 + 0.874064 w_k1_INTC_s110
                                 + 0.941291 w_k1_KO_s110
                                 + 0.868127 w_k1_MAT_s110
                                 + 0.881966 w_k1_MSFT_s110
                                 + 0.912953 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.780145 w_k1_ADM_s110
                                 + 0.69567 w_k1_AXP_s110
                                 + 0.555189 w_k1_DIS_s110
                                 + 0.77936 w_k1_FDX_s110
                                 + 0.671551 w_k1_IBM_s110
                                 + 0.897652 w_k1_INTC_s110
                                 + 0.789276 w_k1_KO_s110
                                 + 0.761099 w_k1_MAT_s110
                                 + 0.855954 w_k1_MSFT_s110
                                 + 0.826446 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.03637 w_k1_ADM_s110
                                 + 1.27039 w_k1_AXP_s110
                                 + 1.42871 w_k1_DIS_s110
                                 + 1.11215 w_k1_FDX_s110
                                 + 1.64044 w_k1_IBM_s110
                                 + 1.1914 w_k1_INTC_s110 + 1.39398 w_k1_KO_s110
                                 + 1.34941 w_k1_MAT_s110
                                 + 1.14256 w_k1_MSFT_s110
                                 + 1.14291 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.910054 w_k1_ADM_s110
                                 + 0.927681 w_k1_AXP_s110
                                 + 0.987812 w_k1_DIS_s110
                                 + 1.01416 w_k1_FDX_s110 + 1.1931 w_k1_IBM_s110
                                 + 0.878493 w_k1_INTC_s110
                                 + 1.11621 w_k1_KO_s110 + 1.14507 w_k1_MAT_s110
                                 + 0.92863 w_k1_MSFT_s110
                                 + 0.993574 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 1.29587 w_k1_ADM_s110
                                 + 1.28178 w_k1_AXP_s110
                                 + 1.39226 w_k1_DIS_s110
                                 + 1.36854 w_k1_FDX_s110
                                 + 1.09119 w_k1_IBM_s110
                                 + 1.37887 w_k1_INTC_s110
                                 + 1.26075 w_k1_KO_s110
                                 + 0.859904 w_k1_MAT_s110
                                 + 1.24383 w_k1_MSFT_s110
                                 + 1.12491 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.02266 w_k1_ADM_s110
                                 + 0.970946 w_k1_AXP_s110
                                 + 1.04251 w_k1_DIS_s110
                                 + 1.12288 w_k1_FDX_s110
                                 + 0.878775 w_k1_IBM_s110
                                 + 1.22351 w_k1_INTC_s110
                                 + 1.08663 w_k1_KO_s110
                                 + 0.843612 w_k1_MAT_s110
                                 + 1.11445 w_k1_MSFT_s110
                                 + 0.964199 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.31352 w_k1_ADM_s110
                                 + 1.46133 w_k1_AXP_s110
                                 + 1.18104 w_k1_DIS_s110
                                 + 1.20496 w_k1_FDX_s110 + 1.1761 w_k1_IBM_s110
                                 + 1.14537 w_k1_INTC_s110
                                 + 1.29026 w_k1_KO_s110 + 1.71919 w_k1_MAT_s110
                                 + 1.0084 w_k1_MSFT_s110 + 1.12104 w_k1_PG_s110
                                 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.822721 w_k1_ADM_s110
                                 + 0.729266 w_k1_AXP_s110
                                 + 0.731382 w_k1_DIS_s110
                                 + 0.855122 w_k1_FDX_s110
                                 + 0.795892 w_k1_IBM_s110
                                 + 0.866369 w_k1_INTC_s110
                                 + 0.899909 w_k1_KO_s110
                                 + 0.710055 w_k1_MAT_s110
                                 + 0.8517 w_k1_MSFT_s110
                                 + 0.873428 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.870227 w_k1_ADM_s111
                                 + 0.740226 w_k1_AXP_s111
                                 + 0.651139 w_k1_DIS_s111
                                 + 0.728876 w_k1_FDX_s111
                                 + 1.1278 w_k1_IBM_s111
                                 + 0.837335 w_k1_INTC_s111
                                 + 0.765286 w_k1_KO_s111
                                 + 0.951656 w_k1_MAT_s111
                                 + 0.839691 w_k1_MSFT_s111
                                 + 0.842974 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.17946 w_k1_ADM_s111
                                 + 1.01913 w_k1_AXP_s111
                                 + 1.13241 w_k1_DIS_s111
                                 + 1.14094 w_k1_FDX_s111
                                 + 0.90318 w_k1_IBM_s111
                                 + 1.11861 w_k1_INTC_s111 + 1.0163 w_k1_KO_s111
                                 + 0.953419 w_k1_MAT_s111
                                 + 1.20219 w_k1_MSFT_s111
                                 + 1.08816 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.799107 w_k1_ADM_s111
                                 + 0.713824 w_k1_AXP_s111
                                 + 0.537147 w_k1_DIS_s111
                                 + 0.732391 w_k1_FDX_s111
                                 + 0.781637 w_k1_IBM_s111
                                 + 0.815417 w_k1_INTC_s111
                                 + 0.734182 w_k1_KO_s111
                                 + 0.736077 w_k1_MAT_s111
                                 + 0.832097 w_k1_MSFT_s111
                                 + 0.874252 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.992036 w_k1_ADM_s111
                                 + 0.677282 w_k1_AXP_s111
                                 + 0.592479 w_k1_DIS_s111
                                 + 0.827693 w_k1_FDX_s111
                                 + 0.629004 w_k1_IBM_s111
                                 + 0.853157 w_k1_INTC_s111
                                 + 0.751216 w_k1_KO_s111
                                 + 0.480418 w_k1_MAT_s111
                                 + 0.941567 w_k1_MSFT_s111
                                 + 0.887364 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.23103 w_k1_ADM_s111
                                 + 1.5444 w_k1_AXP_s111 + 1.47436 w_k1_DIS_s111
                                 + 1.33337 w_k1_FDX_s111
                                 + 1.22197 w_k1_IBM_s111
                                 + 1.3548 w_k1_INTC_s111 + 1.21247 w_k1_KO_s111
                                 + 1.45539 w_k1_MAT_s111
                                 + 1.46429 w_k1_MSFT_s111
                                 + 1.27248 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.833335 w_k1_ADM_s111
                                 + 0.924495 w_k1_AXP_s111
                                 + 0.831268 w_k1_DIS_s111
                                 + 0.880839 w_k1_FDX_s111
                                 + 0.959226 w_k1_IBM_s111
                                 + 1.07779 w_k1_INTC_s111
                                 + 0.820692 w_k1_KO_s111
                                 + 0.925183 w_k1_MAT_s111
                                 + 0.960211 w_k1_MSFT_s111
                                 + 1.00561 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.39064 w_k1_ADM_s111
                                 + 1.407 w_k1_AXP_s111 + 1.40759 w_k1_DIS_s111
                                 + 1.22743 w_k1_FDX_s111
                                 + 1.57979 w_k1_IBM_s111
                                 + 0.952172 w_k1_INTC_s111
                                 + 1.5805 w_k1_KO_s111 + 1.88724 w_k1_MAT_s111
                                 + 1.35199 w_k1_MSFT_s111
                                 + 1.38684 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.06002 w_k1_ADM_s111
                                 + 1.04545 w_k1_AXP_s111
                                 + 1.08187 w_k1_DIS_s111
                                 + 1.09358 w_k1_FDX_s111
                                 + 1.19023 w_k1_IBM_s111
                                 + 0.984947 w_k1_INTC_s111
                                 + 1.08627 w_k1_KO_s111 + 1.27676 w_k1_MAT_s111
                                 + 1.06515 w_k1_MSFT_s111
                                 + 1.05022 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.14674 w_k1_ADM_s112
                                 + 0.8246 w_k1_AXP_s112
                                 + 0.814065 w_k1_DIS_s112
                                 + 0.74626 w_k1_FDX_s112
                                 + 1.35483 w_k1_IBM_s112
                                 + 0.847496 w_k1_INTC_s112
                                 + 0.940315 w_k1_KO_s112
                                 + 1.13055 w_k1_MAT_s112
                                 + 0.756363 w_k1_MSFT_s112
                                 + 0.954637 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.78949 w_k1_ADM_s112
                                 + 0.714456 w_k1_AXP_s112
                                 + 0.698658 w_k1_DIS_s112
                                 + 0.771421 w_k1_FDX_s112
                                 + 0.887473 w_k1_IBM_s112
                                 + 0.850097 w_k1_INTC_s112
                                 + 0.701156 w_k1_KO_s112
                                 + 0.641382 w_k1_MAT_s112
                                 + 0.778573 w_k1_MSFT_s112
                                 + 0.839277 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.06676 w_k1_ADM_s112
                                 + 1.10207 w_k1_AXP_s112
                                 + 1.30475 w_k1_DIS_s112
                                 + 1.11212 w_k1_FDX_s112
                                 + 0.81097 w_k1_IBM_s112
                                 + 0.992402 w_k1_INTC_s112
                                 + 0.942403 w_k1_KO_s112
                                 + 0.824126 w_k1_MAT_s112
                                 + 1.07685 w_k1_MSFT_s112
                                 + 1.10171 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.21463 w_k1_ADM_s112
                                 + 1.33578 w_k1_AXP_s112
                                 + 1.62135 w_k1_DIS_s112
                                 + 1.38161 w_k1_FDX_s112
                                 + 1.16572 w_k1_IBM_s112
                                 + 1.02743 w_k1_INTC_s112
                                 + 1.13723 w_k1_KO_s112 + 1.21992 w_k1_MAT_s112
                                 + 1.1099 w_k1_MSFT_s112 + 1.35446 w_k1_PG_s112
                                 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.16931 w_k1_ADM_s112
                                 + 1.05354 w_k1_AXP_s112
                                 + 1.14936 w_k1_DIS_s112
                                 + 1.00353 w_k1_FDX_s112
                                 + 1.11449 w_k1_IBM_s112
                                 + 0.932071 w_k1_INTC_s112
                                 + 1.0178 w_k1_KO_s112 + 1.13918 w_k1_MAT_s112
                                 + 0.96714 w_k1_MSFT_s112
                                 + 1.19291 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.637922 w_k1_ADM_s112
                                 + 0.631908 w_k1_AXP_s112
                                 + 0.50364 w_k1_DIS_s112
                                 + 0.617325 w_k1_FDX_s112
                                 + 0.629901 w_k1_IBM_s112
                                 + 0.7423 w_k1_INTC_s112
                                 + 0.582086 w_k1_KO_s112
                                 + 0.709527 w_k1_MAT_s112
                                 + 0.612357 w_k1_MSFT_s112
                                 + 0.708133 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.890161 w_k1_ADM_s112
                                 + 0.944668 w_k1_AXP_s112
                                 + 0.792525 w_k1_DIS_s112
                                 + 0.910208 w_k1_FDX_s112
                                 + 0.784972 w_k1_IBM_s112
                                 + 0.87604 w_k1_INTC_s112
                                 + 0.729405 w_k1_KO_s112
                                 + 0.684449 w_k1_MAT_s112
                                 + 0.864722 w_k1_MSFT_s112
                                 + 0.944565 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.52324 w_k1_ADM_s112
                                 + 1.59125 w_k1_AXP_s112
                                 + 1.81874 w_k1_DIS_s112
                                 + 1.26937 w_k1_FDX_s112
                                 + 1.61577 w_k1_IBM_s112
                                 + 1.25316 w_k1_INTC_s112
                                 + 1.50478 w_k1_KO_s112 + 1.82125 w_k1_MAT_s112
                                 + 1.27177 w_k1_MSFT_s112
                                 + 1.23003 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.961154 w_k1_ADM_s113
                                 + 0.99119 w_k1_AXP_s113
                                 + 1.15044 w_k1_DIS_s113
                                 + 1.08588 w_k1_FDX_s113
                                 + 0.961911 w_k1_IBM_s113
                                 + 1.0622 w_k1_INTC_s113 + 1.21182 w_k1_KO_s113
                                 + 1.31131 w_k1_MAT_s113
                                 + 1.11686 w_k1_MSFT_s113
                                 + 1.09693 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.93252 w_k1_ADM_s113
                                 + 0.858468 w_k1_AXP_s113
                                 + 0.811408 w_k1_DIS_s113
                                 + 0.865162 w_k1_FDX_s113
                                 + 1.16055 w_k1_IBM_s113
                                 + 0.955251 w_k1_INTC_s113
                                 + 1.04466 w_k1_KO_s113 + 1.18738 w_k1_MAT_s113
                                 + 0.93626 w_k1_MSFT_s113
                                 + 1.05115 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.997655 w_k1_ADM_s113
                                 + 1.07656 w_k1_AXP_s113 + 1.016 w_k1_DIS_s113
                                 + 0.951756 w_k1_FDX_s113
                                 + 1.56071 w_k1_IBM_s113
                                 + 1.09622 w_k1_INTC_s113
                                 + 1.22613 w_k1_KO_s113 + 2.04415 w_k1_MAT_s113
                                 + 1.04727 w_k1_MSFT_s113
                                 + 1.18577 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.34319 w_k1_ADM_s113
                                 + 1.27581 w_k1_AXP_s113
                                 + 1.45092 w_k1_DIS_s113
                                 + 1.28033 w_k1_FDX_s113
                                 + 0.92786 w_k1_IBM_s113
                                 + 1.31901 w_k1_INTC_s113
                                 + 1.23365 w_k1_KO_s113 + 1.01116 w_k1_MAT_s113
                                 + 1.27604 w_k1_MSFT_s113
                                 + 1.06157 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.22265 w_k1_ADM_s113
                                 + 1.16053 w_k1_AXP_s113 + 1.2711 w_k1_DIS_s113
                                 + 1.08129 w_k1_FDX_s113
                                 + 1.34883 w_k1_IBM_s113
                                 + 1.16892 w_k1_INTC_s113 + 1.3857 w_k1_KO_s113
                                 + 1.07908 w_k1_MAT_s113
                                 + 1.13241 w_k1_MSFT_s113 + 1.0665 w_k1_PG_s113
                                 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.760661 w_k1_ADM_s113
                                 + 0.851486 w_k1_AXP_s113
                                 + 0.755405 w_k1_DIS_s113
                                 + 0.861556 w_k1_FDX_s113
                                 + 0.884254 w_k1_IBM_s113
                                 + 0.895271 w_k1_INTC_s113
                                 + 0.912821 w_k1_KO_s113
                                 + 0.81655 w_k1_MAT_s113
                                 + 0.854103 w_k1_MSFT_s113
                                 + 0.965401 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.881129 w_k1_ADM_s113
                                 + 0.797159 w_k1_AXP_s113
                                 + 0.775173 w_k1_DIS_s113
                                 + 0.890022 w_k1_FDX_s113
                                 + 0.648783 w_k1_IBM_s113
                                 + 0.911834 w_k1_INTC_s113
                                 + 0.860719 w_k1_KO_s113
                                 + 0.703146 w_k1_MAT_s113
                                 + 0.987724 w_k1_MSFT_s113
                                 + 0.96258 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.665053 w_k1_ADM_s113
                                 + 0.743451 w_k1_AXP_s113
                                 + 0.962173 w_k1_DIS_s113
                                 + 0.875819 w_k1_FDX_s113
                                 + 1.04392 w_k1_IBM_s113
                                 + 0.854165 w_k1_INTC_s113
                                 + 1.04568 w_k1_KO_s113 + 1.08336 w_k1_MAT_s113
                                 + 1.00784 w_k1_MSFT_s113
                                 + 0.992666 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.11125 w_k1_ADM_s114
                                 + 1.10185 w_k1_AXP_s114
                                 + 1.13895 w_k1_DIS_s114
                                 + 1.06174 w_k1_FDX_s114
                                 + 0.987053 w_k1_IBM_s114
                                 + 1.08708 w_k1_INTC_s114
                                 + 0.983738 w_k1_KO_s114 + 1.1039 w_k1_MAT_s114
                                 + 1.079 w_k1_MSFT_s114 + 1.03227 w_k1_PG_s114
                                 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.930212 w_k1_ADM_s114
                                 + 0.670339 w_k1_AXP_s114
                                 + 0.735227 w_k1_DIS_s114
                                 + 0.68764 w_k1_FDX_s114
                                 + 0.89379 w_k1_IBM_s114
                                 + 0.809438 w_k1_INTC_s114
                                 + 0.780014 w_k1_KO_s114
                                 + 0.681313 w_k1_MAT_s114
                                 + 0.787459 w_k1_MSFT_s114
                                 + 0.850286 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.96712 w_k1_ADM_s114
                                 + 1.05815 w_k1_AXP_s114
                                 + 1.05235 w_k1_DIS_s114
                                 + 0.917357 w_k1_FDX_s114
                                 + 1.48347 w_k1_IBM_s114
                                 + 1.19985 w_k1_INTC_s114
                                 + 1.19187 w_k1_KO_s114 + 1.2072 w_k1_MAT_s114
                                 + 0.977951 w_k1_MSFT_s114
                                 + 0.972276 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.980482 w_k1_ADM_s114
                                 + 0.715092 w_k1_AXP_s114
                                 + 0.701227 w_k1_DIS_s114
                                 + 0.796915 w_k1_FDX_s114
                                 + 1.00443 w_k1_IBM_s114
                                 + 0.951052 w_k1_INTC_s114
                                 + 0.798799 w_k1_KO_s114
                                 + 0.813783 w_k1_MAT_s114
                                 + 0.871939 w_k1_MSFT_s114
                                 + 0.891954 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.0236 w_k1_ADM_s114
                                 + 0.882548 w_k1_AXP_s114
                                 + 0.943094 w_k1_DIS_s114
                                 + 0.994465 w_k1_FDX_s114
                                 + 0.741715 w_k1_IBM_s114
                                 + 1.02113 w_k1_INTC_s114
                                 + 0.872028 w_k1_KO_s114
                                 + 0.721073 w_k1_MAT_s114
                                 + 0.905763 w_k1_MSFT_s114
                                 + 1.01157 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.47491 w_k1_ADM_s114
                                 + 1.29161 w_k1_AXP_s114
                                 + 1.23151 w_k1_DIS_s114 + 1.1865 w_k1_FDX_s114
                                 + 1.07035 w_k1_IBM_s114
                                 + 1.19398 w_k1_INTC_s114
                                 + 1.11386 w_k1_KO_s114 + 1.04144 w_k1_MAT_s114
                                 + 1.24684 w_k1_MSFT_s114
                                 + 1.15984 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 1.38259 w_k1_ADM_s114
                                 + 1.25968 w_k1_AXP_s114
                                 + 1.41442 w_k1_DIS_s114
                                 + 1.19736 w_k1_FDX_s114
                                 + 1.30575 w_k1_IBM_s114
                                 + 1.37406 w_k1_INTC_s114
                                 + 1.27326 w_k1_KO_s114 + 1.26274 w_k1_MAT_s114
                                 + 1.46476 w_k1_MSFT_s114
                                 + 1.15507 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.945026 w_k1_ADM_s114
                                 + 0.978591 w_k1_AXP_s114
                                 + 0.902354 w_k1_DIS_s114
                                 + 0.879692 w_k1_FDX_s114
                                 + 0.999087 w_k1_IBM_s114
                                 + 1.03369 w_k1_INTC_s114
                                 + 0.847147 w_k1_KO_s114
                                 + 0.929058 w_k1_MAT_s114
                                 + 0.95379 w_k1_MSFT_s114
                                 + 0.82642 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.23984 w_k1_ADM_s115
                                 + 1.19544 w_k1_AXP_s115
                                 + 1.01629 w_k1_DIS_s115
                                 + 1.12583 w_k1_FDX_s115
                                 + 0.967076 w_k1_IBM_s115
                                 + 0.96175 w_k1_INTC_s115
                                 + 1.03642 w_k1_KO_s115
                                 + 0.946953 w_k1_MAT_s115
                                 + 0.85891 w_k1_MSFT_s115
                                 + 1.09858 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.39659 w_k1_ADM_s115
                                 + 1.24398 w_k1_AXP_s115
                                 + 1.40907 w_k1_DIS_s115
                                 + 1.18828 w_k1_FDX_s115
                                 + 1.20033 w_k1_IBM_s115
                                 + 0.914022 w_k1_INTC_s115
                                 + 1.22139 w_k1_KO_s115 + 1.52309 w_k1_MAT_s115
                                 + 0.983062 w_k1_MSFT_s115
                                 + 1.13697 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.76679 w_k1_ADM_s115
                                 + 1.59975 w_k1_AXP_s115 + 1.5894 w_k1_DIS_s115
                                 + 1.50646 w_k1_FDX_s115
                                 + 0.944222 w_k1_IBM_s115
                                 + 1.28103 w_k1_INTC_s115
                                 + 1.18246 w_k1_KO_s115 + 1.11511 w_k1_MAT_s115
                                 + 1.42229 w_k1_MSFT_s115
                                 + 1.19628 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.28759 w_k1_ADM_s115
                                 + 1.57143 w_k1_AXP_s115
                                 + 1.54086 w_k1_DIS_s115 + 1.2473 w_k1_FDX_s115
                                 + 1.72921 w_k1_IBM_s115
                                 + 1.17547 w_k1_INTC_s115
                                 + 1.84689 w_k1_KO_s115 + 2.39435 w_k1_MAT_s115
                                 + 1.2056 w_k1_MSFT_s115 + 1.28662 w_k1_PG_s115
                                 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.09254 w_k1_ADM_s115
                                 + 0.895246 w_k1_AXP_s115
                                 + 0.944387 w_k1_DIS_s115
                                 + 0.804755 w_k1_FDX_s115
                                 + 1.45013 w_k1_IBM_s115
                                 + 0.954585 w_k1_INTC_s115
                                 + 1.2273 w_k1_KO_s115 + 0.967358 w_k1_MAT_s115
                                 + 0.856768 w_k1_MSFT_s115
                                 + 1.06186 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.12328 w_k1_ADM_s115
                                 + 0.824502 w_k1_AXP_s115
                                 + 0.908441 w_k1_DIS_s115
                                 + 1.04857 w_k1_FDX_s115
                                 + 0.619594 w_k1_IBM_s115
                                 + 0.991747 w_k1_INTC_s115
                                 + 0.995812 w_k1_KO_s115
                                 + 0.778125 w_k1_MAT_s115
                                 + 0.960766 w_k1_MSFT_s115
                                 + 0.966415 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.924629 w_k1_ADM_s115
                                 + 0.838456 w_k1_AXP_s115
                                 + 0.818413 w_k1_DIS_s115
                                 + 0.823574 w_k1_FDX_s115
                                 + 0.914568 w_k1_IBM_s115
                                 + 0.845685 w_k1_INTC_s115
                                 + 0.95385 w_k1_KO_s115
                                 + 0.816201 w_k1_MAT_s115
                                 + 0.723266 w_k1_MSFT_s115
                                 + 0.972202 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.16019 w_k1_ADM_s115
                                 + 0.980334 w_k1_AXP_s115
                                 + 1.18193 w_k1_DIS_s115
                                 + 1.06718 w_k1_FDX_s115
                                 + 0.981338 w_k1_IBM_s115
                                 + 1.00167 w_k1_INTC_s115
                                 + 1.17498 w_k1_KO_s115
                                 + 0.928315 w_k1_MAT_s115
                                 + 1.05525 w_k1_MSFT_s115
                                 + 1.09321 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.18464 w_k1_ADM_s116
                                 + 1.70948 w_k1_AXP_s116
                                 + 1.55057 w_k1_DIS_s116
                                 + 1.36121 w_k1_FDX_s116
                                 + 1.62081 w_k1_IBM_s116
                                 + 1.31156 w_k1_INTC_s116
                                 + 1.42536 w_k1_KO_s116 + 1.1929 w_k1_MAT_s116
                                 + 1.23541 w_k1_MSFT_s116 + 1.1599 w_k1_PG_s116
                                 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.0429 w_k1_ADM_s116
                                 + 1.17229 w_k1_AXP_s116
                                 + 0.944728 w_k1_DIS_s116
                                 + 1.0724 w_k1_FDX_s116 + 1.41695 w_k1_IBM_s116
                                 + 1.09239 w_k1_INTC_s116
                                 + 1.20905 w_k1_KO_s116 + 1.52903 w_k1_MAT_s116
                                 + 0.962917 w_k1_MSFT_s116
                                 + 1.12779 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.77292 w_k1_ADM_s116
                                 + 0.649405 w_k1_AXP_s116
                                 + 0.678808 w_k1_DIS_s116
                                 + 0.753878 w_k1_FDX_s116
                                 + 0.790398 w_k1_IBM_s116
                                 + 0.778391 w_k1_INTC_s116
                                 + 0.83819 w_k1_KO_s116 + 0.74223 w_k1_MAT_s116
                                 + 0.775589 w_k1_MSFT_s116
                                 + 0.825483 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.13045 w_k1_ADM_s116
                                 + 1.60787 w_k1_AXP_s116
                                 + 1.08251 w_k1_DIS_s116
                                 + 1.20312 w_k1_FDX_s116
                                 + 1.10385 w_k1_IBM_s116
                                 + 0.985112 w_k1_INTC_s116
                                 + 0.93307 w_k1_KO_s116 + 1.27482 w_k1_MAT_s116
                                 + 0.946467 w_k1_MSFT_s116
                                 + 1.06511 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.89842 w_k1_ADM_s116
                                 + 0.911236 w_k1_AXP_s116
                                 + 1.02503 w_k1_DIS_s116 + 1.0299 w_k1_FDX_s116
                                 + 0.784434 w_k1_IBM_s116
                                 + 0.949856 w_k1_INTC_s116
                                 + 0.949221 w_k1_KO_s116
                                 + 0.919738 w_k1_MAT_s116
                                 + 0.983771 w_k1_MSFT_s116
                                 + 0.990746 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.816168 w_k1_ADM_s116
                                 + 0.86076 w_k1_AXP_s116
                                 + 0.842582 w_k1_DIS_s116
                                 + 0.929123 w_k1_FDX_s116
                                 + 1.06593 w_k1_IBM_s116
                                 + 0.784923 w_k1_INTC_s116
                                 + 1.09742 w_k1_KO_s116
                                 + 0.742547 w_k1_MAT_s116
                                 + 0.889484 w_k1_MSFT_s116
                                 + 0.946619 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.00846 w_k1_ADM_s116
                                 + 1.12852 w_k1_AXP_s116
                                 + 0.971296 w_k1_DIS_s116
                                 + 1.01176 w_k1_FDX_s116
                                 + 1.69413 w_k1_IBM_s116
                                 + 1.03173 w_k1_INTC_s116
                                 + 1.37962 w_k1_KO_s116 + 1.63609 w_k1_MAT_s116
                                 + 0.958023 w_k1_MSFT_s116
                                 + 1.05732 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.19449 w_k1_ADM_s116
                                 + 1.08134 w_k1_AXP_s116
                                 + 1.31605 w_k1_DIS_s116
                                 + 1.26541 w_k1_FDX_s116
                                 + 0.995595 w_k1_IBM_s116
                                 + 1.10299 w_k1_INTC_s116
                                 + 1.16487 w_k1_KO_s116 + 1.07426 w_k1_MAT_s116
                                 + 1.23388 w_k1_MSFT_s116
                                 + 1.08021 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.937514 w_k1_ADM_s117
                                 + 0.849783 w_k1_AXP_s117
                                 + 0.702684 w_k1_DIS_s117
                                 + 0.908543 w_k1_FDX_s117
                                 + 0.596731 w_k1_IBM_s117
                                 + 1.09328 w_k1_INTC_s117
                                 + 0.721762 w_k1_KO_s117
                                 + 0.427431 w_k1_MAT_s117
                                 + 0.758647 w_k1_MSFT_s117
                                 + 0.854095 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.798163 w_k1_ADM_s117
                                 + 0.548679 w_k1_AXP_s117
                                 + 0.478672 w_k1_DIS_s117
                                 + 0.697313 w_k1_FDX_s117
                                 + 0.604781 w_k1_IBM_s117
                                 + 0.850169 w_k1_INTC_s117
                                 + 0.632878 w_k1_KO_s117
                                 + 0.361905 w_k1_MAT_s117
                                 + 0.72093 w_k1_MSFT_s117
                                 + 0.726999 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.978775 w_k1_ADM_s117
                                 + 1.0329 w_k1_AXP_s117 + 1.19624 w_k1_DIS_s117
                                 + 0.945167 w_k1_FDX_s117
                                 + 1.23335 w_k1_IBM_s117
                                 + 1.0646 w_k1_INTC_s117 + 1.09602 w_k1_KO_s117
                                 + 1.11871 w_k1_MAT_s117
                                 + 1.01825 w_k1_MSFT_s117
                                 + 1.04689 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.97784 w_k1_ADM_s117
                                 + 0.822485 w_k1_AXP_s117
                                 + 0.9297 w_k1_DIS_s117
                                 + 0.951495 w_k1_FDX_s117
                                 + 0.915988 w_k1_IBM_s117
                                 + 0.961558 w_k1_INTC_s117
                                 + 1.06629 w_k1_KO_s117
                                 + 0.911185 w_k1_MAT_s117
                                 + 0.945156 w_k1_MSFT_s117
                                 + 0.96006 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.02146 w_k1_ADM_s117
                                 + 0.754827 w_k1_AXP_s117
                                 + 0.69804 w_k1_DIS_s117
                                 + 0.744573 w_k1_FDX_s117
                                 + 0.922828 w_k1_IBM_s117
                                 + 0.920488 w_k1_INTC_s117
                                 + 0.907605 w_k1_KO_s117
                                 + 0.629293 w_k1_MAT_s117
                                 + 0.756068 w_k1_MSFT_s117
                                 + 0.869433 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.31089 w_k1_ADM_s117
                                 + 1.3149 w_k1_AXP_s117 + 1.41715 w_k1_DIS_s117
                                 + 1.28683 w_k1_FDX_s117
                                 + 0.925517 w_k1_IBM_s117
                                 + 1.19481 w_k1_INTC_s117
                                 + 1.24407 w_k1_KO_s117
                                 + 0.863774 w_k1_MAT_s117
                                 + 1.07212 w_k1_MSFT_s117
                                 + 1.14553 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.1153 w_k1_ADM_s117
                                 + 1.1151 w_k1_AXP_s117
                                 + 0.907007 w_k1_DIS_s117
                                 + 1.00818 w_k1_FDX_s117
                                 + 1.13191 w_k1_IBM_s117
                                 + 0.949367 w_k1_INTC_s117
                                 + 1.05957 w_k1_KO_s117
                                 + 0.979433 w_k1_MAT_s117
                                 + 0.955857 w_k1_MSFT_s117
                                 + 1.13452 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.20498 w_k1_ADM_s117
                                 + 1.45527 w_k1_AXP_s117 + 1.6024 w_k1_DIS_s117
                                 + 1.22543 w_k1_FDX_s117
                                 + 1.83246 w_k1_IBM_s117
                                 + 1.21462 w_k1_INTC_s117
                                 + 1.59067 w_k1_KO_s117 + 1.33177 w_k1_MAT_s117
                                 + 1.16396 w_k1_MSFT_s117
                                 + 1.30864 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0259094613301412 w_k1_ADM_s000
                                 + 0.0456747220364347 w_k1_ADM_s001
                                 + 0.0214358671967484 w_k1_ADM_s002
                                 + 0.0334952154106955 w_k1_ADM_s003
                                 + 0.0221069355635364 w_k1_ADM_s004
                                 + 0.0091791122187014 w_k1_ADM_s005
                                 + 0.0238822688461006 w_k1_ADM_s006
                                 + 0.0214103478040857 w_k1_ADM_s007
                                 + 0.0320427827899229 w_k1_ADM_s010
                                 + 0.0197498271973916 w_k1_ADM_s011
                                 + 0.0571986174035749 w_k1_ADM_s012
                                 + 0.0173591185764737 w_k1_ADM_s013
                                 + 0.0321013262725008 w_k1_ADM_s014
                                 + 0.0230458577518092 w_k1_ADM_s015
                                 + 0.0273282043981789 w_k1_ADM_s016
                                 + 0.0146488358875948 w_k1_ADM_s017
                                 + 0.0129766372325415 w_k1_ADM_s100
                                 + 0.0233002112160486 w_k1_ADM_s101
                                 + 0.00847542462962178 w_k1_ADM_s102
                                 + 0.0186877439472784 w_k1_ADM_s103
                                 + 0.0087567682095141 w_k1_ADM_s104
                                 + 0.0309103896259563 w_k1_ADM_s105
                                 + 0.00934495682037277 w_k1_ADM_s106
                                 + 0.0119048846068414 w_k1_ADM_s107
                                 + 0.0930145305515116 w_k1_ADM_s110
                                 + 0.0557872252697467 w_k1_ADM_s111
                                 + 0.0452965313740223 w_k1_ADM_s112
                                 + 0.0249806997823893 w_k1_ADM_s113
                                 + 0.0391376666638802 w_k1_ADM_s114
                                 + 0.0436498234176032 w_k1_ADM_s115
                                 + 0.0462957657968764 w_k1_ADM_s116
                                 + 0.125979195806358 w_k1_ADM_s117
                                 + 0.0256253719519516 w_k1_AXP_s000
                                 + 0.0459455846750686 w_k1_AXP_s001
                                 + 0.0215783943922168 w_k1_AXP_s002
                                 + 0.0368680181145884 w_k1_AXP_s003
                                 + 0.0188189063142922 w_k1_AXP_s004
                                 + 0.00892221673723346 w_k1_AXP_s005
                                 + 0.0260761645386728 w_k1_AXP_s006
                                 + 0.0186174264457068 w_k1_AXP_s007
                                 + 0.0331158848030338 w_k1_AXP_s010
                                 + 0.0198781332565775 w_k1_AXP_s011
                                 + 0.0603572009224571 w_k1_AXP_s012
                                 + 0.0208422379490453 w_k1_AXP_s013
                                 + 0.0313035991584567 w_k1_AXP_s014
                                 + 0.022929946608898 w_k1_AXP_s015
                                 + 0.0287421456480107 w_k1_AXP_s016
                                 + 0.015207010513262 w_k1_AXP_s017
                                 + 0.011701318978097 w_k1_AXP_s100
                                 + 0.0225461740344995 w_k1_AXP_s101
                                 + 0.00900534351158677 w_k1_AXP_s102
                                 + 0.0187546001355029 w_k1_AXP_s103
                                 + 0.00914297981561187 w_k1_AXP_s104
                                 + 0.0362089827442856 w_k1_AXP_s105
                                 + 0.0112211841944865 w_k1_AXP_s106
                                 + 0.0108089730815425 w_k1_AXP_s107
                                 + 0.0921087125316981 w_k1_AXP_s110
                                 + 0.0554567703440493 w_k1_AXP_s111
                                 + 0.0433214295553421 w_k1_AXP_s112
                                 + 0.0246316536730154 w_k1_AXP_s113
                                 + 0.0353825741676663 w_k1_AXP_s114
                                 + 0.0382706587048065 w_k1_AXP_s115
                                 + 0.0501973937846021 w_k1_AXP_s116
                                 + 0.121300799370626 w_k1_AXP_s117
                                 + 0.0258470628490106 w_k1_DIS_s000
                                 + 0.0451301437199006 w_k1_DIS_s001
                                 + 0.0207793690509481 w_k1_DIS_s002
                                 + 0.0344878732068423 w_k1_DIS_s003
                                 + 0.0211420391823347 w_k1_DIS_s004
                                 + 0.0105389737240481 w_k1_DIS_s005
                                 + 0.0250233830792812 w_k1_DIS_s006
                                 + 0.020432809425387 w_k1_DIS_s007
                                 + 0.0313132867612649 w_k1_DIS_s010
                                 + 0.0224164943317124 w_k1_DIS_s011
                                 + 0.0585271105435193 w_k1_DIS_s012
                                 + 0.0211264418350489 w_k1_DIS_s013
                                 + 0.0324100801292435 w_k1_DIS_s014
                                 + 0.0233087698020456 w_k1_DIS_s015
                                 + 0.0266400920469764 w_k1_DIS_s016
                                 + 0.0147315572596719 w_k1_DIS_s017
                                 + 0.0123148965872755 w_k1_DIS_s100
                                 + 0.0221014548594071 w_k1_DIS_s101
                                 + 0.00849833417547338 w_k1_DIS_s102
                                 + 0.0193554468479966 w_k1_DIS_s103
                                 + 0.00902311527674754 w_k1_DIS_s104
                                 + 0.0345361238302122 w_k1_DIS_s105
                                 + 0.0114126730300391 w_k1_DIS_s106
                                 + 0.0118364314676935 w_k1_DIS_s107
                                 + 0.09298732868373 w_k1_DIS_s110
                                 + 0.0538366326515682 w_k1_DIS_s111
                                 + 0.0458819080893621 w_k1_DIS_s112
                                 + 0.0262102907634812 w_k1_DIS_s113
                                 + 0.0362347670117235 w_k1_DIS_s114
                                 + 0.0401985234807358 w_k1_DIS_s115
                                 + 0.0486685790293549 w_k1_DIS_s116
                                 + 0.123440745539585 w_k1_DIS_s117
                                 + 0.0258648884851216 w_k1_FDX_s000
                                 + 0.0455710650672942 w_k1_FDX_s001
                                 + 0.0214930708262522 w_k1_FDX_s002
                                 + 0.0362336537413467 w_k1_FDX_s003
                                 + 0.0219924078939937 w_k1_FDX_s004
                                 + 0.00993001727824425 w_k1_FDX_s005
                                 + 0.0235520214205249 w_k1_FDX_s006
                                 + 0.0203284584164005 w_k1_FDX_s007
                                 + 0.0325608299451308 w_k1_FDX_s010
                                 + 0.0206677812647809 w_k1_FDX_s011
                                 + 0.0573849036441523 w_k1_FDX_s012
                                 + 0.0210095880543136 w_k1_FDX_s013
                                 + 0.032181990049337 w_k1_FDX_s014
                                 + 0.0208637145767545 w_k1_FDX_s015
                                 + 0.0260511460076777 w_k1_FDX_s016
                                 + 0.0138429927992931 w_k1_FDX_s017
                                 + 0.0111193380542549 w_k1_FDX_s100
                                 + 0.0228033023025122 w_k1_FDX_s101
                                 + 0.00823044949560754 w_k1_FDX_s102
                                 + 0.0184721138126478 w_k1_FDX_s103
                                 + 0.00955988534073314 w_k1_FDX_s104
                                 + 0.0342547796875521 w_k1_FDX_s105
                                 + 0.0109962946608562 w_k1_FDX_s106
                                 + 0.0113812708875073 w_k1_FDX_s107
                                 + 0.0963518983988364 w_k1_FDX_s110
                                 + 0.0546175342461919 w_k1_FDX_s111
                                 + 0.0415502913942942 w_k1_FDX_s112
                                 + 0.0252738089133839 w_k1_FDX_s113
                                 + 0.0347385136722383 w_k1_FDX_s114
                                 + 0.0384016968805628 w_k1_FDX_s115
                                 + 0.050361800304629 w_k1_FDX_s116
                                 + 0.119712878696348 w_k1_FDX_s117
                                 + 0.0234140048710852 w_k1_IBM_s000
                                 + 0.0472084548687269 w_k1_IBM_s001
                                 + 0.0223165031151243 w_k1_IBM_s002
                                 + 0.0371965636344676 w_k1_IBM_s003
                                 + 0.022211727321609 w_k1_IBM_s004
                                 + 0.00976177807931702 w_k1_IBM_s005
                                 + 0.0246045676696004 w_k1_IBM_s006
                                 + 0.019134112019173 w_k1_IBM_s007
                                 + 0.0313145016005679 w_k1_IBM_s010
                                 + 0.0216689355755436 w_k1_IBM_s011
                                 + 0.057369839994757 w_k1_IBM_s012
                                 + 0.0216826324454011 w_k1_IBM_s013
                                 + 0.03168057906891 w_k1_IBM_s014
                                 + 0.0224710932264752 w_k1_IBM_s015
                                 + 0.0279863930878122 w_k1_IBM_s016
                                 + 0.0149141512128365 w_k1_IBM_s017
                                 + 0.0117861727536073 w_k1_IBM_s100
                                 + 0.0234230280888773 w_k1_IBM_s101
                                 + 0.00852148333117407 w_k1_IBM_s102
                                 + 0.0195348650626944 w_k1_IBM_s103
                                 + 0.00920790168021628 w_k1_IBM_s104
                                 + 0.0342665721329777 w_k1_IBM_s105
                                 + 0.011394729779554 w_k1_IBM_s106
                                 + 0.011202537406195 w_k1_IBM_s107
                                 + 0.0931800953281382 w_k1_IBM_s110
                                 + 0.0553524459707643 w_k1_IBM_s111
                                 + 0.044994844999941 w_k1_IBM_s112
                                 + 0.0250761331010973 w_k1_IBM_s113
                                 + 0.037672080568446 w_k1_IBM_s114
                                 + 0.0387870329418337 w_k1_IBM_s115
                                 + 0.0531656922935855 w_k1_IBM_s116
                                 + 0.116488544630076 w_k1_IBM_s117
                                 + 0.0229234320910709 w_k1_INTC_s000
                                 + 0.0494360444280533 w_k1_INTC_s001
                                 + 0.0220803442961619 w_k1_INTC_s002
                                 + 0.0329988709173658 w_k1_INTC_s003
                                 + 0.021500560049233 w_k1_INTC_s004
                                 + 0.00958299243651545 w_k1_INTC_s005
                                 + 0.0201511270094198 w_k1_INTC_s006
                                 + 0.0205465538665048 w_k1_INTC_s007
                                 + 0.0306157613557022 w_k1_INTC_s010
                                 + 0.019908681981198 w_k1_INTC_s011
                                 + 0.0585504989444401 w_k1_INTC_s012
                                 + 0.0248785725647011 w_k1_INTC_s013
                                 + 0.0340101818977988 w_k1_INTC_s014
                                 + 0.0189712301434175 w_k1_INTC_s015
                                 + 0.0278267585850082 w_k1_INTC_s016
                                 + 0.0130598823284482 w_k1_INTC_s017
                                 + 0.010938582269146 w_k1_INTC_s100
                                 + 0.0222428631616649 w_k1_INTC_s101
                                 + 0.00728118046066134 w_k1_INTC_s102
                                 + 0.0191683696861907 w_k1_INTC_s103
                                 + 0.00898630621388065 w_k1_INTC_s104
                                 + 0.0339774555091102 w_k1_INTC_s105
                                 + 0.011315398899514 w_k1_INTC_s106
                                 + 0.0105001964136229 w_k1_INTC_s107
                                 + 0.0970299829686637 w_k1_INTC_s110
                                 + 0.0556510521011998 w_k1_INTC_s111
                                 + 0.0399666556758053 w_k1_INTC_s112
                                 + 0.0262090837175791 w_k1_INTC_s113
                                 + 0.0388306097886664 w_k1_INTC_s114
                                 + 0.0363929719987874 w_k1_INTC_s115
                                 + 0.0464830146432952 w_k1_INTC_s116
                                 + 0.12421640947576 w_k1_INTC_s117
                                 + 0.0238661381148984 w_k1_KO_s000
                                 + 0.0445233292801279 w_k1_KO_s001
                                 + 0.019249790825535 w_k1_KO_s002
                                 + 0.0359823548200198 w_k1_KO_s003
                                 + 0.0220728493337615 w_k1_KO_s004
                                 + 0.0093959963767603 w_k1_KO_s005
                                 + 0.0245974071538921 w_k1_KO_s006
                                 + 0.0219718412316222 w_k1_KO_s007
                                 + 0.0290597535238899 w_k1_KO_s010
                                 + 0.0240466389209543 w_k1_KO_s011
                                 + 0.062025614558316 w_k1_KO_s012
                                 + 0.0213835038563965 w_k1_KO_s013
                                 + 0.0322458658969462 w_k1_KO_s014
                                 + 0.0208298474763 w_k1_KO_s015
                                 + 0.0255538230073003 w_k1_KO_s016
                                 + 0.0141636668941221 w_k1_KO_s017
                                 + 0.0128385897004552 w_k1_KO_s100
                                 + 0.0228159770333176 w_k1_KO_s101
                                 + 0.00869252821418073 w_k1_KO_s102
                                 + 0.0205028296296756 w_k1_KO_s103
                                 + 0.00871199166052541 w_k1_KO_s104
                                 + 0.0314438625061243 w_k1_KO_s105
                                 + 0.0115026374912292 w_k1_KO_s106
                                 + 0.0128109275434697 w_k1_KO_s107
                                 + 0.0990553298652915 w_k1_KO_s110
                                 + 0.0534121943385619 w_k1_KO_s111
                                 + 0.0402044982534829 w_k1_KO_s112
                                 + 0.027881014434433 w_k1_KO_s113
                                 + 0.0354394580553698 w_k1_KO_s114
                                 + 0.0427225610088228 w_k1_KO_s115
                                 + 0.053062538057906 w_k1_KO_s116
                                 + 0.124735864201045 w_k1_KO_s117
                                 + 0.0290915332718091 w_k1_MAT_s000
                                 + 0.039452862858937 w_k1_MAT_s001
                                 + 0.0236381655667149 w_k1_MAT_s002
                                 + 0.0374445236723678 w_k1_MAT_s003
                                 + 0.0241647613575309 w_k1_MAT_s004
                                 + 0.00934795778125066 w_k1_MAT_s005
                                 + 0.0250888722339032 w_k1_MAT_s006
                                 + 0.0175552659022609 w_k1_MAT_s007
                                 + 0.0340765394072994 w_k1_MAT_s010
                                 + 0.0235179542821022 w_k1_MAT_s011
                                 + 0.0667753923300656 w_k1_MAT_s012
                                 + 0.0230467759274253 w_k1_MAT_s013
                                 + 0.0313416595374608 w_k1_MAT_s014
                                 + 0.0275174162056682 w_k1_MAT_s015
                                 + 0.025867294302894 w_k1_MAT_s016
                                 + 0.0155915667323846 w_k1_MAT_s017
                                 + 0.0110128173216788 w_k1_MAT_s100
                                 + 0.0261768612616811 w_k1_MAT_s101
                                 + 0.00956959361993999 w_k1_MAT_s102
                                 + 0.0183008094335244 w_k1_MAT_s103
                                 + 0.00947383004843438 w_k1_MAT_s104
                                 + 0.0348770792151666 w_k1_MAT_s105
                                 + 0.0124582466246254 w_k1_MAT_s106
                                 + 0.0103404533759495 w_k1_MAT_s107
                                 + 0.08938276413771 w_k1_MAT_s110
                                 + 0.058015189336377 w_k1_MAT_s111
                                 + 0.0431900081896343 w_k1_MAT_s112
                                 + 0.0263763712420627 w_k1_MAT_s113
                                 + 0.0341449688688545 w_k1_MAT_s114
                                 + 0.039702530604941 w_k1_MAT_s115
                                 + 0.0495311578846723 w_k1_MAT_s116
                                 + 0.0986928305847106 w_k1_MAT_s117
                                 + 0.0256263831648146 w_k1_MSFT_s000
                                 + 0.0439934815438623 w_k1_MSFT_s001
                                 + 0.0217482361854531 w_k1_MSFT_s002
                                 + 0.0352625352634976 w_k1_MSFT_s003
                                 + 0.0213541783945087 w_k1_MSFT_s004
                                 + 0.00975229549257304 w_k1_MSFT_s005
                                 + 0.021052473213394 w_k1_MSFT_s006
                                 + 0.0208083526773644 w_k1_MSFT_s007
                                 + 0.0316273722224125 w_k1_MSFT_s010
                                 + 0.022356200375317 w_k1_MSFT_s011
                                 + 0.0588536578931101 w_k1_MSFT_s012
                                 + 0.0229918932898998 w_k1_MSFT_s013
                                 + 0.0342347535098511 w_k1_MSFT_s014
                                 + 0.0205985064274711 w_k1_MSFT_s015
                                 + 0.0262347846264508 w_k1_MSFT_s016
                                 + 0.0138311234401147 w_k1_MSFT_s017
                                 + 0.0104045392874221 w_k1_MSFT_s100
                                 + 0.0226572897133688 w_k1_MSFT_s101
                                 + 0.00881557398538566 w_k1_MSFT_s102
                                 + 0.0190024755518864 w_k1_MSFT_s103
                                 + 0.00859625527695021 w_k1_MSFT_s104
                                 + 0.0334168078641902 w_k1_MSFT_s105
                                 + 0.011643338539617 w_k1_MSFT_s106
                                 + 0.0106066253556863 w_k1_MSFT_s107
                                 + 0.0919743575167818 w_k1_MSFT_s110
                                 + 0.05896411478544 w_k1_MSFT_s111
                                 + 0.0396705560437098 w_k1_MSFT_s112
                                 + 0.0265458506549063 w_k1_MSFT_s113
                                 + 0.0363218969651954 w_k1_MSFT_s114
                                 + 0.0352966596376739 w_k1_MSFT_s115
                                 + 0.0474465631080012 w_k1_MSFT_s116
                                 + 0.110898490120035 w_k1_MSFT_s117
                                 + 0.0245823233421756 w_k1_PG_s000
                                 + 0.0426877610335437 w_k1_PG_s001
                                 + 0.0224907411045128 w_k1_PG_s002
                                 + 0.0355973070336995 w_k1_PG_s003
                                 + 0.0231514119778946 w_k1_PG_s004
                                 + 0.00931659753011019 w_k1_PG_s005
                                 + 0.0222763184990363 w_k1_PG_s006
                                 + 0.0209191978524625 w_k1_PG_s007
                                 + 0.0299004365418727 w_k1_PG_s010
                                 + 0.021673620442051 w_k1_PG_s011
                                 + 0.0543591304311999 w_k1_PG_s012
                                 + 0.0207383091200432 w_k1_PG_s013
                                 + 0.0309654978248344 w_k1_PG_s014
                                 + 0.0201599829764773 w_k1_PG_s015
                                 + 0.025248316948717 w_k1_PG_s016
                                 + 0.0148173406965873 w_k1_PG_s017
                                 + 0.0119203269303962 w_k1_PG_s100
                                 + 0.0229226340533545 w_k1_PG_s101
                                 + 0.00847322523319753 w_k1_PG_s102
                                 + 0.0194303034325526 w_k1_PG_s103
                                 + 0.0095401996224558 w_k1_PG_s104
                                 + 0.0333220843237579 w_k1_PG_s105
                                 + 0.0124779845698847 w_k1_PG_s106
                                 + 0.0101905457702149 w_k1_PG_s107
                                 + 0.0898085996970574 w_k1_PG_s110
                                 + 0.0570478473934566 w_k1_PG_s111
                                 + 0.0449642351430741 w_k1_PG_s112
                                 + 0.0259695101046455 w_k1_PG_s113
                                 + 0.0358129079902543 w_k1_PG_s114
                                 + 0.0396916485689635 w_k1_PG_s115
                                 + 0.0489849386494969 w_k1_PG_s116
                                 + 0.120269549273568 w_k1_PG_s117 >= 1.041
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
