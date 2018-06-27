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
 w_k1_ADM_s0_return_constraint:  0.996623 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  1.09119 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.04918 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 1.06119 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.19784 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.857758 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.38652 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.23581 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.950127 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 0.652195 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 0.916055 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.828442 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.23379 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.06406 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.788518 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 0.981987 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.01222 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.0104 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.891954 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.84012 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.20705 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.680809 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.80627 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.650852 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.18769 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.860235 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 0.906524 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.28824 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.10323 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 0.802543 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.40732 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.03014 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.17612 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.02666 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.804872 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 0.776452 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.23247 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.67487 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  0.932795 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  1.29085 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 0.986508 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.10464 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.25503 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.799693 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 2.1336 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.20027 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.891887 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 0.639631 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 0.9542 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.870982 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.72133 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.4489 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.597172 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.970514 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.06279 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.33076 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.842344 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.838742 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.57899 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.528757 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 1.16434 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.738371 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.24867 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.888846 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 0.812804 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.49769 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.14608 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 0.823618 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.58856 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 1.07738 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.38576 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.958106 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.755907 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 0.731822 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.09744 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.79364 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  0.955499 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  1.30603 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 0.988811 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 1.13785 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.41621 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.825835 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 2.32607 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.30613 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.716077 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 0.492374 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.936043 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.816364 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.55675 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.22669 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.721578 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 0.883205 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.13086 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.36686 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.752161 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.56811 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.69435 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.413293 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 1.01734 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.689592 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.59885 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.799441 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 0.82453 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.38361 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.998349 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.882064 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.32016 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.32257 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.10712 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.889117 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.718828 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.576482 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.08608 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.96029 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  0.998669 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  1.12195 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.01695 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.976098 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.26013 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.893962 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.68976 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.19978 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.919378 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 0.623409 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 0.901907 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.888541 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.40761 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.12249 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.788133 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.04561 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.1015 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.19793 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.78523 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.766988 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.49999 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.551336 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.919106 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.734148 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.38181 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.800045 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 0.886467 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.21975 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.09125 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 0.812257 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.27154 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.14369 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.0142 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.915129 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.756212 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.73102 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.1408 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.49849 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  0.922906 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.38006 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.895499 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.35944 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 1.26774 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 0.932573 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 1.43139 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 0.993839 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.58934 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.760638 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.13449 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.850179 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 1.20982 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.31444 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.668607 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.789556 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.995142 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.54455 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 1.05827 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.73475 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 1.06171 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.806249 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.42615 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.962461 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 0.932175 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.20529 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.594822 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.6076 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.912451 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.60414 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.09122 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 1.12699 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.63072 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.903986 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.16013 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.66296 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.744621 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.27065 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 0.98715 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.12788 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 0.947523 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.956327 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.32855 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.931831 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.38431 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.18193 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.878249 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 0.734572 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 0.983709 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.831182 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.40752 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.03413 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.931483 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.03425 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.05072 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.1939 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.921871 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.930624 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.46342 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.572445 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.986 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.696055 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.09664 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.674545 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 0.750613 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.02068 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.902461 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 0.922833 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.09583 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.06803 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.02509 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.903324 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.828783 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 0.748911 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.04337 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.27137 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   0.968512 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   1.24111 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  0.919738 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.26255 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.24927 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.887533 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.58505 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.04992 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.690099 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 0.724708 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.07537 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.834181 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.30559 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.15425 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.737521 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.776971 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.00192 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.14376 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.865688 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.679824 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.15757 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.581467 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.24559 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.891861 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.30296 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.943367 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.778432 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.4294 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.92327 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.15803 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.20722 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 1.16355 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.3295 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.03826 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.935798 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 0.686087 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 0.970915 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.43743 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  0.957554 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  1.50828 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.894979 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.20702 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 1.13986 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 0.938797 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 1.9246 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 0.755143 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.600496 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 0.623536 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 1.2122 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.819571 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.5847 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.36264 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.558147 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.940991 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.17944 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.83487 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 1.23595 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.691971 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.29632 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.674533 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.73587 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 1.20394 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.12849 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.03134 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.849626 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.92557 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.16826 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.49024 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.23259 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.940873 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.4339 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.873882 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.954181 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.65192 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.544877 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.5851 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 0.974116 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 1.13643 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 0.997453 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 1.06284 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.25953 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.854437 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.57848 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.22517 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.835595 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 0.682868 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 0.994534 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.930008 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.23756 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.10196 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.821492 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 0.947988 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.01445 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.04673 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.825595 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.730718 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.50861 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.586363 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.948839 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.819535 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.28313 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.708591 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 0.906728 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.09672 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.943088 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 0.853395 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.18693 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.21272 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.11197 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.07162 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.837352 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.799645 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.08263 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.38112 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.00192 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   1.13948 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  0.963513 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.01792 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.16784 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.869026 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 1.46197 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.18838 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.849073 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 0.739326 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.05063 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.869295 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.26608 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.12006 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.883212 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.08518 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.10111 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.25907 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.909925 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.831102 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.32253 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.6811 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 1.05923 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.901759 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.20588 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.866121 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 0.869715 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.17063 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.0226 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.986045 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.34072 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 1.09217 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.15879 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.05293 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.909734 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 0.903866 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.16166 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.39206 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.319999981995779 z_k1_s0000
                                 - 0.511999504753267 z_k1_s0001
                                 - 0.641001255148052 z_k1_s0002
                                 - 0.570001316233049 z_k1_s0003
                                 - 0.389998241054307 z_k1_s0004
                                 - 0.411998660098373 z_k1_s0005
                                 - 0.470999430534749 z_k1_s0006
                                 - 0.571000275189643 z_k1_s0007
                                 - 0.841005472123098 z_k1_s0010
                                 - 0.912998541753056 z_k1_s0011
                                 - 1.04300472689897 z_k1_s0012
                                 - 0.804001691054574 z_k1_s0013
                                 - 1.43899944592742 z_k1_s0014
                                 - 1.91500744314096 z_k1_s0015
                                 - 0.545000671524257 z_k1_s0016
                                 - 0.709001212501938 z_k1_s0017
                                 - 0.936000455863433 z_k1_s0020
                                 - 0.358000272007645 z_k1_s0021
                                 - 0.35300023246562 z_k1_s0022
                                 - 0.289000091927894 z_k1_s0023
                                 - 0.436000157662853 z_k1_s0024
                                 - 1.38099986210151 z_k1_s0025
                                 - 0.300000087520301 z_k1_s0026
                                 - 0.516999792843137 z_k1_s0027
                                 - 1.12600229118211 z_k1_s0030
                                 - 1.32500213231955 z_k1_s0031
                                 - 0.994999205687199 z_k1_s0032
                                 - 0.450000624368827 z_k1_s0033
                                 - 0.518000025113367 z_k1_s0034
                                 - 1.44400527170067 z_k1_s0035
                                 - 1.74799802857868 z_k1_s0036
                                 - 1.21099823573527 z_k1_s0037
                                 - 1.13499589235988 z_k1_s0040
                                 - 1.27899455719281 z_k1_s0041
                                 - 0.679998827902418 z_k1_s0042
                                 - 1.71000064336325 z_k1_s0043
                                 - 2.22099132246565 z_k1_s0044
                                 - 1.52900140501229 z_k1_s0045
                                 - 1.77999816116014 z_k1_s0046
                                 - 0.664998417033249 z_k1_s0047
                                 - 1.37499971885789 z_k1_s0050
                                 - 0.750001696555405 z_k1_s0051
                                 - 0.578999193448713 z_k1_s0052
                                 - 0.554998949235294 z_k1_s0053
                                 - 1.48499942504703 z_k1_s0054
                                 - 0.771998924598284 z_k1_s0055
                                 - 0.674998813704916 z_k1_s0056
                                 - 0.591998788744482 z_k1_s0057
                                 - 0.163999854057083 z_k1_s0060
                                 - 0.692000866279978 z_k1_s0061
                                 - 0.584999148587743 z_k1_s0062
                                 - 0.41799863114299 z_k1_s0063
                                 - 0.419000540205652 z_k1_s0064
                                 - 0.192000137908311 z_k1_s0065
                                 - 0.796998765846887 z_k1_s0066
                                 - 0.350000111990142 z_k1_s0067
                                 - 0.36400032735795 z_k1_s0070
                                 - 0.634001088662392 z_k1_s0071
                                 - 0.768998805812204 z_k1_s0072
                                 - 0.374000256906173 z_k1_s0073
                                 - 0.404000045550843 z_k1_s0074
                                 - 0.507001488749507 z_k1_s0075
                                 - 0.2350000566348 z_k1_s0076
                                 - 0.518001753702863 z_k1_s0077
                                 - 0.216000650559718 z_k1_s0100
                                 - 0.131999974619866 z_k1_s0101
                                 - 0.118999950789881 z_k1_s0102
                                 - 0.37499979624963 z_k1_s0103
                                 - 0.174000064339814 z_k1_s0104
                                 - 0.171999169519897 z_k1_s0105
                                 - 0.347000508769595 z_k1_s0106
                                 - 0.12000006719987 z_k1_s0107
                                 - 0.897000413261548 z_k1_s0110
                                 - 0.190999902823805 z_k1_s0111
                                 - 0.657001009713318 z_k1_s0112
                                 - 0.413000106105929 z_k1_s0113
                                 - 0.335999919967533 z_k1_s0114
                                 - 0.306999992038788 z_k1_s0115
                                 - 0.195999890397727 z_k1_s0116
                                 - 0.778001086502223 z_k1_s0117
                                 - 0.355999948188079 z_k1_s0120
                                 - 0.33499981655792 z_k1_s0121
                                 - 0.610000218400838 z_k1_s0122
                                 - 0.62399778255497 z_k1_s0123
                                 - 0.345999765335656 z_k1_s0124
                                 - 0.273000028792525 z_k1_s0125
                                 - 0.925000089853795 z_k1_s0126
                                 - 0.737000353507217 z_k1_s0127
                                 - 1.21000253279315 z_k1_s0130
                                 - 0.27700021151071 z_k1_s0131
                                 - 0.554002520224923 z_k1_s0132
                                 - 0.369000859058553 z_k1_s0133
                                 - 0.59000101834506 z_k1_s0134
                                 - 0.588003432008944 z_k1_s0135
                                 - 1.36800060166154 z_k1_s0136
                                 - 0.287000726412305 z_k1_s0137
                                 - 0.68700048495471 z_k1_s0140
                                 - 0.762996256095329 z_k1_s0141
                                 - 0.94699701782674 z_k1_s0142
                                 - 0.715997954642453 z_k1_s0143
                                 - 0.637997922983776 z_k1_s0144
                                 - 0.591996392804587 z_k1_s0145
                                 - 0.763998386354358 z_k1_s0146
                                 - 0.251999319045561 z_k1_s0147
                                 - 0.181000231292539 z_k1_s0150
                                 - 0.292000267496737 z_k1_s0151
                                 - 0.515000373042127 z_k1_s0152
                                 - 0.53799961769206 z_k1_s0153
                                 - 0.50200145976215 z_k1_s0154
                                 - 0.531999418870086 z_k1_s0155
                                 - 0.181000231292539 z_k1_s0156
                                 - 0.294000333770728 z_k1_s0157
                                 - 0.995995223198024 z_k1_s0160
                                 - 0.759999111206787 z_k1_s0161
                                 - 0.24699935616831 z_k1_s0162
                                 - 0.224999493993554 z_k1_s0163
                                 - 0.849996826290686 z_k1_s0164
                                 - 0.474997669634762 z_k1_s0165
                                 - 0.454998440337523 z_k1_s0166
                                 - 0.724998685067138 z_k1_s0167
                                 - 0.109999481067818 z_k1_s0170
                                 - 0.162999617491231 z_k1_s0171
                                 - 0.187000316814758 z_k1_s0172
                                 - 0.0419999703167284 z_k1_s0173
                                 - 0.166000386156369 z_k1_s0174
                                 - 0.185999697259873 z_k1_s0175
                                 - 0.141000157277474 z_k1_s0176
                                 - 0.0949999977401901 z_k1_s0177
                                 - 0.0429999854649444 z_k1_s1000
                                 - 0.0900000043219575 z_k1_s1001
                                 - 0.070999820809645 z_k1_s1002
                                 - 0.024000001152522 z_k1_s1003
                                 - 0.0429999854649444 z_k1_s1004
                                 - 0.0300000014406525 z_k1_s1005
                                 - 0.0949999215621121 z_k1_s1006
                                 - 0.101999805698328 z_k1_s1007
                                 - 0.224999036615124 z_k1_s1010
                                 - 0.0659997260470716 z_k1_s1011
                                 - 0.253999523003294 z_k1_s1012
                                 - 0.150999002102985 z_k1_s1013
                                 - 0.171999309625616 z_k1_s1014
                                 - 0.132998923654342 z_k1_s1015
                                 - 0.100999633786992 z_k1_s1016
                                 - 0.193998829509389 z_k1_s1017
                                 - 0.11999989999745 z_k1_s1020
                                 - 0.223000169540853 z_k1_s1021
                                 - 0.0919999021147259 z_k1_s1022
                                 - 0.135000285309361 z_k1_s1023
                                 - 0.157999560688518 z_k1_s1024
                                 - 0.115999882114216 z_k1_s1025
                                 - 0.0959999199979599 z_k1_s1026
                                 - 0.333000024830218 z_k1_s1027
                                 - 0.0740000661186011 z_k1_s1030
                                 - 0.0820001290503344 z_k1_s1031
                                 - 0.0419999691916476 z_k1_s1032
                                 - 0.0559999245222014 z_k1_s1033
                                 - 0.0790001054509344 z_k1_s1034
                                 - 0.0419999691916476 z_k1_s1035
                                 - 0.0609999638545347 z_k1_s1036
                                 - 0.080000113317401 z_k1_s1037
                                 - 0.0670001228897051 z_k1_s1040
                                 - 0.293000856174619 z_k1_s1041
                                 - 0.0970000858283104 z_k1_s1042
                                 - 0.115000095231497 z_k1_s1043
                                 - 0.308999634087647 z_k1_s1044
                                 - 0.18800027398897 z_k1_s1045
                                 - 0.189999923477952 z_k1_s1046
                                 - 0.323000344512871 z_k1_s1047
                                 - 0.118000088417823 z_k1_s1050
                                 - 0.0490000145362164 z_k1_s1051
                                 - 0.176000144199465 z_k1_s1052
                                 - 0.108000204145118 z_k1_s1053
                                 - 0.079999801181594 z_k1_s1054
                                 - 0.079000176254297 z_k1_s1055
                                 - 0.04599997655441 z_k1_s1056
                                 - 0.0709999780361535 z_k1_s1057
                                 - 0.173999730638111 z_k1_s1060
                                 - 0.176999583552841 z_k1_s1061
                                 - 0.377999786419995 z_k1_s1062
                                 - 0.243998693310438 z_k1_s1063
                                 - 0.170999698123734 z_k1_s1064
                                 - 0.321000246244713 z_k1_s1065
                                 - 0.143999764693634 z_k1_s1066
                                 - 0.186999153135156 z_k1_s1067
                                 - 0.0719998317430789 z_k1_s1070
                                 - 0.0399999472350109 z_k1_s1071
                                 - 0.0580001570705768 z_k1_s1072
                                 - 0.0369999580623795 z_k1_s1073
                                 - 0.0389999508441338 z_k1_s1074
                                 - 0.0669998497886933 z_k1_s1075
                                 - 0.0539999883072166 z_k1_s1076
                                 - 0.0909999463695961 z_k1_s1077
                                 - 0.0740000538042322 z_k1_s1100
                                 - 0.216000378839483 z_k1_s1101
                                 - 0.0660000527064796 z_k1_s1102
                                 - 0.0800000109775266 z_k1_s1103
                                 - 0.0880001866753599 z_k1_s1104
                                 - 0.15199984625722 z_k1_s1105
                                 - 0.0960003623731933 z_k1_s1106
                                 - 0.101000035684137 z_k1_s1107
                                 - 0.131000262283509 z_k1_s1110
                                 - 0.113000285191102 z_k1_s1111
                                 - 0.143999816739252 z_k1_s1112
                                 - 0.0899998854620327 z_k1_s1113
                                 - 0.0970003055534072 z_k1_s1114
                                 - 0.159000226649474 z_k1_s1115
                                 - 0.0409999478215927 z_k1_s1116
                                 - 0.0829999801705179 z_k1_s1117
                                 - 0.206999733509918 z_k1_s1120
                                 - 0.126999774255439 z_k1_s1121
                                 - 0.33499882019816 z_k1_s1122
                                 - 0.229000264611712 z_k1_s1123
                                 - 0.205999931777736 z_k1_s1124
                                 - 0.32499917187841 z_k1_s1125
                                 - 0.0799999592544788 z_k1_s1126
                                 - 0.121999786995772 z_k1_s1127
                                 - 0.226000692026546 z_k1_s1130
                                 - 0.405000088352692 z_k1_s1131
                                 - 0.110000047721723 z_k1_s1132
                                 - 0.0849999495258623 z_k1_s1133
                                 - 0.366000053963169 z_k1_s1134
                                 - 0.18599962647302 z_k1_s1135
                                 - 0.214999831224233 z_k1_s1136
                                 - 0.1539999969304 z_k1_s1137
                                 - 0.182999965395319 z_k1_s1140
                                 - 0.102000065572792 z_k1_s1141
                                 - 0.413000760755466 z_k1_s1142
                                 - 0.138999898609021 z_k1_s1143
                                 - 0.307999932004145 z_k1_s1144
                                 - 0.174999953252356 z_k1_s1145
                                 - 0.324000670289995 z_k1_s1146
                                 - 0.140999990894752 z_k1_s1147
                                 - 0.312000941435637 z_k1_s1150
                                 - 0.168000404511279 z_k1_s1151
                                 - 0.273000348255168 z_k1_s1152
                                 - 0.194000609764519 z_k1_s1153
                                 - 0.227000380146432 z_k1_s1154
                                 - 0.298000289575046 z_k1_s1155
                                 - 0.177000307306272 z_k1_s1156
                                 - 0.253000775600078 z_k1_s1157
                                 - 0.543000096972381 z_k1_s1160
                                 - 0.225999593483042 z_k1_s1161
                                 - 0.319998291765158 z_k1_s1162
                                 - 0.300000847276467 z_k1_s1163
                                 - 0.162999807354445 z_k1_s1164
                                 - 0.253999672562179 z_k1_s1165
                                 - 0.417000002315906 z_k1_s1166
                                 - 0.388999400837487 z_k1_s1167
                                 - 0.038999976321955 z_k1_s1170
                                 - 0.0919998591441081 z_k1_s1171
                                 - 0.165999933216523 z_k1_s1172
                                 - 0.0599999686722379 z_k1_s1173
                                 - 0.0579999630864974 z_k1_s1174
                                 - 0.0510000098363983 z_k1_s1175
                                 - 0.107000232537127 z_k1_s1176
                                 - 0.0900002513583249 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.851483 w_k1_ADM_s000
                                 + 0.647214 w_k1_AXP_s000
                                 + 0.582154 w_k1_DIS_s000
                                 + 0.693822 w_k1_FDX_s000
                                 + 0.826079 w_k1_IBM_s000
                                 + 1.01334 w_k1_INTC_s000
                                 + 0.724353 w_k1_KO_s000
                                 + 0.875308 w_k1_MAT_s000
                                 + 0.747744 w_k1_MSFT_s000
                                 + 0.802679 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.38351 w_k1_ADM_s000
                                 + 1.43083 w_k1_AXP_s000
                                 + 1.23296 w_k1_DIS_s000
                                 + 1.13439 w_k1_FDX_s000
                                 + 1.34774 w_k1_IBM_s000 + 1.402 w_k1_INTC_s000
                                 + 1.48392 w_k1_KO_s000 + 1.69974 w_k1_MAT_s000
                                 + 1.20184 w_k1_MSFT_s000
                                 + 1.15353 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.11507 w_k1_ADM_s000
                                 + 0.885116 w_k1_AXP_s000
                                 + 1.0633 w_k1_DIS_s000 + 1.01118 w_k1_FDX_s000
                                 + 1.28913 w_k1_IBM_s000
                                 + 1.07842 w_k1_INTC_s000
                                 + 1.07779 w_k1_KO_s000
                                 + 0.940234 w_k1_MAT_s000
                                 + 0.991902 w_k1_MSFT_s000
                                 + 0.993674 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.02435 w_k1_ADM_s000
                                 + 0.695311 w_k1_AXP_s000
                                 + 1.04527 w_k1_DIS_s000
                                 + 0.959323 w_k1_FDX_s000
                                 + 0.818268 w_k1_IBM_s000
                                 + 1.1436 w_k1_INTC_s000
                                 + 0.977456 w_k1_KO_s000
                                 + 0.526205 w_k1_MAT_s000
                                 + 0.957051 w_k1_MSFT_s000
                                 + 0.858019 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.758744 w_k1_ADM_s000
                                 + 0.843747 w_k1_AXP_s000
                                 + 0.672329 w_k1_DIS_s000
                                 + 0.777962 w_k1_FDX_s000
                                 + 1.35914 w_k1_IBM_s000
                                 + 0.990642 w_k1_INTC_s000
                                 + 0.994026 w_k1_KO_s000
                                 + 1.13948 w_k1_MAT_s000
                                 + 0.856334 w_k1_MSFT_s000
                                 + 0.891896 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.82448 w_k1_ADM_s000
                                 + 1.28766 w_k1_AXP_s000
                                 + 1.64657 w_k1_DIS_s000 + 1.4144 w_k1_FDX_s000
                                 + 0.878762 w_k1_IBM_s000
                                 + 1.22117 w_k1_INTC_s000 + 1.2123 w_k1_KO_s000
                                 + 0.898889 w_k1_MAT_s000
                                 + 1.32257 w_k1_MSFT_s000
                                 + 1.20605 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.26792 w_k1_ADM_s000
                                 + 0.86583 w_k1_AXP_s000
                                 + 0.823916 w_k1_DIS_s000
                                 + 1.0497 w_k1_FDX_s000
                                 + 0.580986 w_k1_IBM_s000
                                 + 1.12364 w_k1_INTC_s000
                                 + 0.700145 w_k1_KO_s000
                                 + 0.538224 w_k1_MAT_s000
                                 + 1.08467 w_k1_MSFT_s000
                                 + 0.913462 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.958355 w_k1_ADM_s000
                                 + 1.0076 w_k1_AXP_s000
                                 + 0.830445 w_k1_DIS_s000
                                 + 0.943321 w_k1_FDX_s000
                                 + 0.864973 w_k1_IBM_s000
                                 + 0.994643 w_k1_INTC_s000
                                 + 0.936021 w_k1_KO_s000
                                 + 1.01008 w_k1_MAT_s000
                                 + 0.910829 w_k1_MSFT_s000
                                 + 0.918163 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.898475 w_k1_ADM_s001
                                 + 0.94347 w_k1_AXP_s001
                                 + 0.742609 w_k1_DIS_s001
                                 + 0.982933 w_k1_FDX_s001
                                 + 0.685535 w_k1_IBM_s001
                                 + 0.950932 w_k1_INTC_s001
                                 + 0.882322 w_k1_KO_s001
                                 + 1.02107 w_k1_MAT_s001
                                 + 0.910177 w_k1_MSFT_s001
                                 + 0.859289 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.552852 w_k1_ADM_s001
                                 + 0.605372 w_k1_AXP_s001
                                 + 0.620545 w_k1_DIS_s001
                                 + 0.753172 w_k1_FDX_s001
                                 + 0.735685 w_k1_IBM_s001
                                 + 0.816307 w_k1_INTC_s001
                                 + 0.835948 w_k1_KO_s001
                                 + 1.06538 w_k1_MAT_s001
                                 + 0.758654 w_k1_MSFT_s001
                                 + 0.756598 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.29594 w_k1_ADM_s001
                                 + 1.25285 w_k1_AXP_s001
                                 + 1.35718 w_k1_DIS_s001
                                 + 1.16446 w_k1_FDX_s001
                                 + 1.63237 w_k1_IBM_s001
                                 + 1.19128 w_k1_INTC_s001
                                 + 1.64777 w_k1_KO_s001 + 1.83279 w_k1_MAT_s001
                                 + 1.11955 w_k1_MSFT_s001
                                 + 1.10412 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.862912 w_k1_ADM_s001
                                 + 0.807508 w_k1_AXP_s001
                                 + 0.784777 w_k1_DIS_s001
                                 + 0.830476 w_k1_FDX_s001
                                 + 0.886503 w_k1_IBM_s001
                                 + 0.992644 w_k1_INTC_s001
                                 + 0.98243 w_k1_KO_s001 + 1.11144 w_k1_MAT_s001
                                 + 0.881377 w_k1_MSFT_s001
                                 + 0.856943 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.879743 w_k1_ADM_s001
                                 + 1.02465 w_k1_AXP_s001
                                 + 0.864412 w_k1_DIS_s001
                                 + 0.874767 w_k1_FDX_s001
                                 + 1.3137 w_k1_IBM_s001
                                 + 1.09565 w_k1_INTC_s001
                                 + 1.14398 w_k1_KO_s001 + 1.71107 w_k1_MAT_s001
                                 + 0.967582 w_k1_MSFT_s001
                                 + 0.985388 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.962185 w_k1_ADM_s001
                                 + 1.03101 w_k1_AXP_s001
                                 + 1.13678 w_k1_DIS_s001
                                 + 1.05382 w_k1_FDX_s001
                                 + 0.917204 w_k1_IBM_s001
                                 + 1.15605 w_k1_INTC_s001 + 1.0303 w_k1_KO_s001
                                 + 1.54519 w_k1_MAT_s001
                                 + 1.0581 w_k1_MSFT_s001
                                 + 0.904779 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 1.19906 w_k1_ADM_s001
                                 + 1.44202 w_k1_AXP_s001
                                 + 1.68944 w_k1_DIS_s001
                                 + 1.43184 w_k1_FDX_s001 + 1.0889 w_k1_IBM_s001
                                 + 1.30717 w_k1_INTC_s001
                                 + 1.49483 w_k1_KO_s001 + 1.71496 w_k1_MAT_s001
                                 + 1.42548 w_k1_MSFT_s001
                                 + 1.05348 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.17126 w_k1_ADM_s001
                                 + 1.21569 w_k1_AXP_s001
                                 + 1.19061 w_k1_DIS_s001
                                 + 1.30831 w_k1_FDX_s001
                                 + 0.793237 w_k1_IBM_s001
                                 + 1.12936 w_k1_INTC_s001
                                 + 1.09202 w_k1_KO_s001 + 1.06896 w_k1_MAT_s001
                                 + 1.10409 w_k1_MSFT_s001
                                 + 1.01326 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.08852 w_k1_ADM_s002
                                 + 1.50664 w_k1_AXP_s002
                                 + 1.21971 w_k1_DIS_s002
                                 + 1.23132 w_k1_FDX_s002
                                 + 1.22008 w_k1_IBM_s002
                                 + 1.27188 w_k1_INTC_s002
                                 + 1.20915 w_k1_KO_s002 + 1.43582 w_k1_MAT_s002
                                 + 1.29226 w_k1_MSFT_s002
                                 + 1.21944 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.19545 w_k1_ADM_s002
                                 + 1.61562 w_k1_AXP_s002
                                 + 1.69746 w_k1_DIS_s002
                                 + 1.26027 w_k1_FDX_s002
                                 + 1.52222 w_k1_IBM_s002
                                 + 1.24067 w_k1_INTC_s002
                                 + 1.52304 w_k1_KO_s002 + 1.72289 w_k1_MAT_s002
                                 + 1.28271 w_k1_MSFT_s002
                                 + 1.36733 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.768605 w_k1_ADM_s002
                                 + 0.822347 w_k1_AXP_s002
                                 + 0.642053 w_k1_DIS_s002
                                 + 0.880583 w_k1_FDX_s002
                                 + 0.669013 w_k1_IBM_s002
                                 + 0.915962 w_k1_INTC_s002
                                 + 0.752378 w_k1_KO_s002
                                 + 0.900957 w_k1_MAT_s002
                                 + 0.831897 w_k1_MSFT_s002
                                 + 0.867959 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.866329 w_k1_ADM_s002
                                 + 1.08369 w_k1_AXP_s002
                                 + 0.721212 w_k1_DIS_s002
                                 + 0.894313 w_k1_FDX_s002
                                 + 1.20052 w_k1_IBM_s002
                                 + 0.943199 w_k1_INTC_s002
                                 + 1.04307 w_k1_KO_s002 + 1.48866 w_k1_MAT_s002
                                 + 0.931072 w_k1_MSFT_s002
                                 + 1.03051 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.857398 w_k1_ADM_s002
                                 + 0.709752 w_k1_AXP_s002
                                 + 0.793367 w_k1_DIS_s002
                                 + 0.813227 w_k1_FDX_s002
                                 + 1.00216 w_k1_IBM_s002
                                 + 1.00342 w_k1_INTC_s002
                                 + 0.923391 w_k1_KO_s002
                                 + 0.613524 w_k1_MAT_s002
                                 + 0.895507 w_k1_MSFT_s002
                                 + 0.899806 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.858455 w_k1_ADM_s002
                                 + 1.00562 w_k1_AXP_s002
                                 + 1.12244 w_k1_DIS_s002
                                 + 1.07537 w_k1_FDX_s002
                                 + 0.909799 w_k1_IBM_s002
                                 + 1.15719 w_k1_INTC_s002
                                 + 1.00365 w_k1_KO_s002 + 1.01046 w_k1_MAT_s002
                                 + 1.16718 w_k1_MSFT_s002
                                 + 0.999506 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.767805 w_k1_ADM_s002
                                 + 0.855328 w_k1_AXP_s002
                                 + 0.744119 w_k1_DIS_s002
                                 + 0.692248 w_k1_FDX_s002
                                 + 1.19988 w_k1_IBM_s002
                                 + 0.941021 w_k1_INTC_s002
                                 + 0.943305 w_k1_KO_s002
                                 + 1.34024 w_k1_MAT_s002
                                 + 0.784905 w_k1_MSFT_s002
                                 + 0.893169 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 0.848745 w_k1_ADM_s002
                                 + 0.728751 w_k1_AXP_s002
                                 + 0.942399 w_k1_DIS_s002
                                 + 0.904969 w_k1_FDX_s002
                                 + 0.758688 w_k1_IBM_s002
                                 + 0.909852 w_k1_INTC_s002
                                 + 0.894152 w_k1_KO_s002
                                 + 0.990282 w_k1_MAT_s002
                                 + 1.12918 w_k1_MSFT_s002
                                 + 0.950343 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.856628 w_k1_ADM_s003
                                 + 0.813436 w_k1_AXP_s003
                                 + 0.737755 w_k1_DIS_s003
                                 + 0.887776 w_k1_FDX_s003
                                 + 0.597198 w_k1_IBM_s003
                                 + 0.90492 w_k1_INTC_s003
                                 + 0.724523 w_k1_KO_s003
                                 + 0.668212 w_k1_MAT_s003
                                 + 0.908555 w_k1_MSFT_s003
                                 + 0.928767 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.11791 w_k1_ADM_s003
                                 + 0.794013 w_k1_AXP_s003
                                 + 1.0949 w_k1_DIS_s003 + 1.02846 w_k1_FDX_s003
                                 + 0.873867 w_k1_IBM_s003
                                 + 1.14029 w_k1_INTC_s003
                                 + 0.901353 w_k1_KO_s003
                                 + 0.730625 w_k1_MAT_s003
                                 + 1.06781 w_k1_MSFT_s003
                                 + 1.03681 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.22808 w_k1_ADM_s003
                                 + 1.32393 w_k1_AXP_s003
                                 + 1.47174 w_k1_DIS_s003
                                 + 1.36263 w_k1_FDX_s003
                                 + 1.08116 w_k1_IBM_s003
                                 + 1.27409 w_k1_INTC_s003
                                 + 1.02054 w_k1_KO_s003 + 1.32314 w_k1_MAT_s003
                                 + 1.40527 w_k1_MSFT_s003
                                 + 1.18927 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.08095 w_k1_ADM_s003
                                 + 1.23327 w_k1_AXP_s003
                                 + 1.20547 w_k1_DIS_s003
                                 + 1.11948 w_k1_FDX_s003
                                 + 1.00142 w_k1_IBM_s003
                                 + 1.14067 w_k1_INTC_s003
                                 + 1.11516 w_k1_KO_s003 + 1.43147 w_k1_MAT_s003
                                 + 1.17443 w_k1_MSFT_s003
                                 + 1.14862 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.30723 w_k1_ADM_s003
                                 + 1.18487 w_k1_AXP_s003
                                 + 1.24224 w_k1_DIS_s003
                                 + 1.25151 w_k1_FDX_s003
                                 + 0.710654 w_k1_IBM_s003
                                 + 1.15178 w_k1_INTC_s003
                                 + 0.957093 w_k1_KO_s003
                                 + 0.93936 w_k1_MAT_s003
                                 + 1.25848 w_k1_MSFT_s003
                                 + 1.01999 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.931604 w_k1_ADM_s003
                                 + 1.06047 w_k1_AXP_s003
                                 + 0.818752 w_k1_DIS_s003
                                 + 0.83645 w_k1_FDX_s003
                                 + 1.22017 w_k1_IBM_s003
                                 + 0.942661 w_k1_INTC_s003
                                 + 1.00549 w_k1_KO_s003 + 1.40522 w_k1_MAT_s003
                                 + 0.963761 w_k1_MSFT_s003
                                 + 0.993249 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.13658 w_k1_ADM_s003
                                 + 1.21008 w_k1_AXP_s003 + 1.3971 w_k1_DIS_s003
                                 + 1.10534 w_k1_FDX_s003
                                 + 1.51943 w_k1_IBM_s003
                                 + 1.09766 w_k1_INTC_s003
                                 + 1.28621 w_k1_KO_s003 + 1.23244 w_k1_MAT_s003
                                 + 1.0422 w_k1_MSFT_s003 + 1.17917 w_k1_PG_s003
                                 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.802442 w_k1_ADM_s003
                                 + 0.746916 w_k1_AXP_s003
                                 + 0.731544 w_k1_DIS_s003
                                 + 0.794643 w_k1_FDX_s003
                                 + 0.77114 w_k1_IBM_s003
                                 + 0.94672 w_k1_INTC_s003
                                 + 0.780092 w_k1_KO_s003
                                 + 0.815847 w_k1_MAT_s003
                                 + 0.789266 w_k1_MSFT_s003
                                 + 0.793865 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.725144 w_k1_ADM_s004
                                 + 0.594196 w_k1_AXP_s004
                                 + 0.629054 w_k1_DIS_s004
                                 + 0.721665 w_k1_FDX_s004
                                 + 0.766644 w_k1_IBM_s004
                                 + 0.804014 w_k1_INTC_s004
                                 + 0.650049 w_k1_KO_s004
                                 + 0.496235 w_k1_MAT_s004
                                 + 0.723168 w_k1_MSFT_s004
                                 + 0.752091 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.863228 w_k1_ADM_s004
                                 + 0.953203 w_k1_AXP_s004
                                 + 0.852231 w_k1_DIS_s004
                                 + 0.828929 w_k1_FDX_s004
                                 + 1.41091 w_k1_IBM_s004
                                 + 0.95222 w_k1_INTC_s004
                                 + 0.954594 w_k1_KO_s004
                                 + 1.04698 w_k1_MAT_s004
                                 + 0.794588 w_k1_MSFT_s004
                                 + 0.8897 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.03006 w_k1_ADM_s004
                                 + 1.22929 w_k1_AXP_s004 + 1.4852 w_k1_DIS_s004
                                 + 1.17833 w_k1_FDX_s004
                                 + 1.74117 w_k1_IBM_s004
                                 + 1.15297 w_k1_INTC_s004
                                 + 1.48913 w_k1_KO_s004 + 1.40569 w_k1_MAT_s004
                                 + 1.1215 w_k1_MSFT_s004 + 1.12228 w_k1_PG_s004
                                 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.03559 w_k1_ADM_s004
                                 + 1.07196 w_k1_AXP_s004
                                 + 1.29206 w_k1_DIS_s004
                                 + 1.06779 w_k1_FDX_s004
                                 + 1.23822 w_k1_IBM_s004
                                 + 0.987707 w_k1_INTC_s004
                                 + 1.14713 w_k1_KO_s004
                                 + 0.898331 w_k1_MAT_s004
                                 + 0.987441 w_k1_MSFT_s004
                                 + 1.00178 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.09035 w_k1_ADM_s004
                                 + 1.05015 w_k1_AXP_s004
                                 + 1.03533 w_k1_DIS_s004
                                 + 1.05005 w_k1_FDX_s004
                                 + 0.806701 w_k1_IBM_s004
                                 + 1.05044 w_k1_INTC_s004
                                 + 0.855089 w_k1_KO_s004
                                 + 0.752823 w_k1_MAT_s004
                                 + 1.06356 w_k1_MSFT_s004
                                 + 0.946085 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.834111 w_k1_ADM_s004
                                 + 0.80813 w_k1_AXP_s004
                                 + 0.804994 w_k1_DIS_s004
                                 + 0.877117 w_k1_FDX_s004
                                 + 0.900511 w_k1_IBM_s004
                                 + 1.07272 w_k1_INTC_s004
                                 + 0.821431 w_k1_KO_s004
                                 + 0.71854 w_k1_MAT_s004
                                 + 0.969291 w_k1_MSFT_s004
                                 + 0.867437 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.22031 w_k1_ADM_s004
                                 + 1.34407 w_k1_AXP_s004 + 1.3657 w_k1_DIS_s004
                                 + 1.30913 w_k1_FDX_s004
                                 + 0.89287 w_k1_IBM_s004
                                 + 1.26888 w_k1_INTC_s004
                                 + 0.995394 w_k1_KO_s004
                                 + 0.68753 w_k1_MAT_s004
                                 + 1.14333 w_k1_MSFT_s004
                                 + 1.02402 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.991734 w_k1_ADM_s004
                                 + 0.779277 w_k1_AXP_s004
                                 + 0.957016 w_k1_DIS_s004
                                 + 0.82523 w_k1_FDX_s004
                                 + 0.810134 w_k1_IBM_s004
                                 + 0.917828 w_k1_INTC_s004
                                 + 0.840277 w_k1_KO_s004
                                 + 0.471419 w_k1_MAT_s004
                                 + 0.876286 w_k1_MSFT_s004
                                 + 0.794459 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.962061 w_k1_ADM_s005
                                 + 1.02672 w_k1_AXP_s005
                                 + 1.10231 w_k1_DIS_s005
                                 + 1.11756 w_k1_FDX_s005
                                 + 0.89425 w_k1_IBM_s005
                                 + 1.3865 w_k1_INTC_s005 + 1.07991 w_k1_KO_s005
                                 + 0.718307 w_k1_MAT_s005
                                 + 1.11168 w_k1_MSFT_s005
                                 + 1.10433 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.674206 w_k1_ADM_s005
                                 + 0.624516 w_k1_AXP_s005
                                 + 0.673846 w_k1_DIS_s005
                                 + 0.759052 w_k1_FDX_s005
                                 + 0.664572 w_k1_IBM_s005
                                 + 1.06166 w_k1_INTC_s005
                                 + 0.771789 w_k1_KO_s005
                                 + 0.469701 w_k1_MAT_s005
                                 + 0.825274 w_k1_MSFT_s005
                                 + 0.808808 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.844538 w_k1_ADM_s005
                                 + 0.793598 w_k1_AXP_s005
                                 + 0.646347 w_k1_DIS_s005
                                 + 0.707864 w_k1_FDX_s005
                                 + 1.17982 w_k1_IBM_s005
                                 + 0.826444 w_k1_INTC_s005
                                 + 0.879972 w_k1_KO_s005
                                 + 0.978639 w_k1_MAT_s005
                                 + 0.744656 w_k1_MSFT_s005
                                 + 0.889125 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.1276 w_k1_ADM_s005
                                 + 1.27287 w_k1_AXP_s005
                                 + 0.976559 w_k1_DIS_s005
                                 + 0.897185 w_k1_FDX_s005
                                 + 1.44475 w_k1_IBM_s005
                                 + 1.29744 w_k1_INTC_s005
                                 + 1.26048 w_k1_KO_s005 + 1.50498 w_k1_MAT_s005
                                 + 0.98117 w_k1_MSFT_s005
                                 + 1.02341 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.02325 w_k1_ADM_s005
                                 + 1.01023 w_k1_AXP_s005
                                 + 0.772851 w_k1_DIS_s005
                                 + 0.924859 w_k1_FDX_s005
                                 + 0.88336 w_k1_IBM_s005
                                 + 1.05411 w_k1_INTC_s005
                                 + 0.873025 w_k1_KO_s005
                                 + 1.20603 w_k1_MAT_s005
                                 + 0.923941 w_k1_MSFT_s005
                                 + 0.98613 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.993927 w_k1_ADM_s005
                                 + 1.19176 w_k1_AXP_s005
                                 + 0.972025 w_k1_DIS_s005
                                 + 1.03475 w_k1_FDX_s005
                                 + 1.09781 w_k1_IBM_s005
                                 + 1.20098 w_k1_INTC_s005
                                 + 1.12368 w_k1_KO_s005 + 1.32628 w_k1_MAT_s005
                                 + 0.981868 w_k1_MSFT_s005
                                 + 1.02293 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 1.27786 w_k1_ADM_s005
                                 + 1.26755 w_k1_AXP_s005
                                 + 1.53252 w_k1_DIS_s005
                                 + 1.35686 w_k1_FDX_s005
                                 + 0.953379 w_k1_IBM_s005
                                 + 1.41745 w_k1_INTC_s005
                                 + 1.05073 w_k1_KO_s005
                                 + 0.867584 w_k1_MAT_s005
                                 + 1.26202 w_k1_MSFT_s005
                                 + 1.09885 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.14614 w_k1_ADM_s005
                                 + 1.48704 w_k1_AXP_s005
                                 + 1.35815 w_k1_DIS_s005
                                 + 1.16893 w_k1_FDX_s005
                                 + 1.57301 w_k1_IBM_s005
                                 + 1.19472 w_k1_INTC_s005
                                 + 1.49421 w_k1_KO_s005 + 1.73214 w_k1_MAT_s005
                                 + 1.12691 w_k1_MSFT_s005
                                 + 1.22738 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.763181 w_k1_ADM_s006
                                 + 1.02133 w_k1_AXP_s006
                                 + 0.819816 w_k1_DIS_s006
                                 + 0.94319 w_k1_FDX_s006
                                 + 0.836722 w_k1_IBM_s006
                                 + 0.959924 w_k1_INTC_s006
                                 + 0.800264 w_k1_KO_s006
                                 + 0.838407 w_k1_MAT_s006
                                 + 0.97847 w_k1_MSFT_s006 + 0.8968 w_k1_PG_s006
                                 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.997222 w_k1_ADM_s006
                                 + 1.24828 w_k1_AXP_s006 + 0.9283 w_k1_DIS_s006
                                 + 1.09388 w_k1_FDX_s006
                                 + 0.852752 w_k1_IBM_s006
                                 + 1.12317 w_k1_INTC_s006
                                 + 0.808293 w_k1_KO_s006
                                 + 0.964167 w_k1_MAT_s006
                                 + 1.11031 w_k1_MSFT_s006
                                 + 0.975073 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.775618 w_k1_ADM_s006
                                 + 0.749459 w_k1_AXP_s006
                                 + 0.694432 w_k1_DIS_s006
                                 + 0.749692 w_k1_FDX_s006
                                 + 0.750462 w_k1_IBM_s006
                                 + 0.704567 w_k1_INTC_s006
                                 + 0.681328 w_k1_KO_s006
                                 + 0.74735 w_k1_MAT_s006
                                 + 0.861135 w_k1_MSFT_s006
                                 + 0.692668 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.04931 w_k1_ADM_s006
                                 + 0.893178 w_k1_AXP_s006
                                 + 0.837041 w_k1_DIS_s006
                                 + 0.82516 w_k1_FDX_s006
                                 + 1.31519 w_k1_IBM_s006
                                 + 0.831939 w_k1_INTC_s006
                                 + 1.11425 w_k1_KO_s006 + 1.33845 w_k1_MAT_s006
                                 + 0.923633 w_k1_MSFT_s006
                                 + 0.974318 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.15821 w_k1_ADM_s006
                                 + 0.953702 w_k1_AXP_s006
                                 + 1.19827 w_k1_DIS_s006
                                 + 1.02546 w_k1_FDX_s006
                                 + 1.38074 w_k1_IBM_s006
                                 + 1.07465 w_k1_INTC_s006
                                 + 1.36821 w_k1_KO_s006
                                 + 0.913716 w_k1_MAT_s006
                                 + 1.2391 w_k1_MSFT_s006 + 1.02229 w_k1_PG_s006
                                 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.15652 w_k1_ADM_s006
                                 + 1.41582 w_k1_AXP_s006
                                 + 1.67213 w_k1_DIS_s006
                                 + 1.36419 w_k1_FDX_s006
                                 + 1.49255 w_k1_IBM_s006
                                 + 1.12937 w_k1_INTC_s006
                                 + 1.42354 w_k1_KO_s006 + 1.81498 w_k1_MAT_s006
                                 + 1.27551 w_k1_MSFT_s006
                                 + 1.24866 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.950191 w_k1_ADM_s006
                                 + 0.780108 w_k1_AXP_s006
                                 + 1.01671 w_k1_DIS_s006
                                 + 1.03671 w_k1_FDX_s006
                                 + 0.763469 w_k1_IBM_s006
                                 + 0.938468 w_k1_INTC_s006
                                 + 1.01564 w_k1_KO_s006
                                 + 0.643899 w_k1_MAT_s006
                                 + 1.27177 w_k1_MSFT_s006
                                 + 0.928972 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.12414 w_k1_ADM_s006
                                 + 1.08565 w_k1_AXP_s006
                                 + 1.46761 w_k1_DIS_s006
                                 + 1.18466 w_k1_FDX_s006
                                 + 0.821324 w_k1_IBM_s006
                                 + 1.12723 w_k1_INTC_s006
                                 + 1.11501 w_k1_KO_s006
                                 + 0.682683 w_k1_MAT_s006
                                 + 1.31444 w_k1_MSFT_s006
                                 + 1.06209 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.14616 w_k1_ADM_s007
                                 + 1.06614 w_k1_AXP_s007
                                 + 1.32077 w_k1_DIS_s007
                                 + 1.01167 w_k1_FDX_s007
                                 + 1.05538 w_k1_IBM_s007
                                 + 1.07428 w_k1_INTC_s007
                                 + 1.10509 w_k1_KO_s007
                                 + 0.840335 w_k1_MAT_s007
                                 + 0.967307 w_k1_MSFT_s007
                                 + 1.0003 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.943013 w_k1_ADM_s007
                                 + 0.724524 w_k1_AXP_s007
                                 + 0.880043 w_k1_DIS_s007
                                 + 0.89713 w_k1_FDX_s007
                                 + 0.62788 w_k1_IBM_s007
                                 + 0.960775 w_k1_INTC_s007
                                 + 0.773044 w_k1_KO_s007
                                 + 0.455735 w_k1_MAT_s007
                                 + 0.825901 w_k1_MSFT_s007
                                 + 0.771782 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.90531 w_k1_ADM_s007
                                 + 0.753055 w_k1_AXP_s007
                                 + 0.756223 w_k1_DIS_s007
                                 + 0.822948 w_k1_FDX_s007
                                 + 0.958817 w_k1_IBM_s007
                                 + 0.905738 w_k1_INTC_s007
                                 + 0.789736 w_k1_KO_s007
                                 + 0.870893 w_k1_MAT_s007
                                 + 0.801773 w_k1_MSFT_s007
                                 + 0.837019 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.02522 w_k1_ADM_s007
                                 + 1.35101 w_k1_AXP_s007 + 1.0451 w_k1_DIS_s007
                                 + 1.10723 w_k1_FDX_s007
                                 + 0.826548 w_k1_IBM_s007
                                 + 1.0887 w_k1_INTC_s007
                                 + 0.940851 w_k1_KO_s007
                                 + 1.18025 w_k1_MAT_s007
                                 + 0.910814 w_k1_MSFT_s007
                                 + 0.878396 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.932025 w_k1_ADM_s007
                                 + 1.0856 w_k1_AXP_s007 + 1.43343 w_k1_DIS_s007
                                 + 1.07371 w_k1_FDX_s007
                                 + 1.55457 w_k1_IBM_s007
                                 + 1.44222 w_k1_INTC_s007 + 1.5895 w_k1_KO_s007
                                 + 1.71864 w_k1_MAT_s007
                                 + 1.2179 w_k1_MSFT_s007 + 1.15971 w_k1_PG_s007
                                 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 1.03953 w_k1_ADM_s007
                                 + 1.03944 w_k1_AXP_s007
                                 + 0.933462 w_k1_DIS_s007
                                 + 0.912958 w_k1_FDX_s007
                                 + 1.24033 w_k1_IBM_s007
                                 + 1.07515 w_k1_INTC_s007 + 1.0524 w_k1_KO_s007
                                 + 1.28017 w_k1_MAT_s007
                                 + 0.883757 w_k1_MSFT_s007
                                 + 0.939742 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.40583 w_k1_ADM_s007
                                 + 1.35508 w_k1_AXP_s007 + 1.8334 w_k1_DIS_s007
                                 + 1.29387 w_k1_FDX_s007
                                 + 1.44816 w_k1_IBM_s007
                                 + 1.29295 w_k1_INTC_s007
                                 + 1.73827 w_k1_KO_s007 + 1.68508 w_k1_MAT_s007
                                 + 1.44698 w_k1_MSFT_s007
                                 + 1.19544 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 1.07035 w_k1_ADM_s007
                                 + 0.960727 w_k1_AXP_s007
                                 + 1.4806 w_k1_DIS_s007 + 1.1806 w_k1_FDX_s007
                                 + 0.718421 w_k1_IBM_s007
                                 + 1.36348 w_k1_INTC_s007
                                 + 1.13209 w_k1_KO_s007
                                 + 0.992272 w_k1_MAT_s007
                                 + 1.25789 w_k1_MSFT_s007
                                 + 1.09239 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.02451 w_k1_ADM_s010
                                 + 1.3068 w_k1_AXP_s010
                                 + 0.949518 w_k1_DIS_s010
                                 + 1.25089 w_k1_FDX_s010
                                 + 0.925057 w_k1_IBM_s010
                                 + 1.15046 w_k1_INTC_s010
                                 + 0.998789 w_k1_KO_s010 + 1.283 w_k1_MAT_s010
                                 + 1.23501 w_k1_MSFT_s010
                                 + 1.00322 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.10448 w_k1_ADM_s010
                                 + 1.01733 w_k1_AXP_s010
                                 + 1.13047 w_k1_DIS_s010
                                 + 1.14432 w_k1_FDX_s010
                                 + 0.867329 w_k1_IBM_s010
                                 + 1.01023 w_k1_INTC_s010 + 1.1076 w_k1_KO_s010
                                 + 1.1546 w_k1_MAT_s010
                                 + 1.09335 w_k1_MSFT_s010
                                 + 1.11654 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.07644 w_k1_ADM_s010
                                 + 0.766833 w_k1_AXP_s010
                                 + 0.849227 w_k1_DIS_s010
                                 + 1.03034 w_k1_FDX_s010
                                 + 0.631703 w_k1_IBM_s010
                                 + 0.976537 w_k1_INTC_s010
                                 + 0.806765 w_k1_KO_s010
                                 + 0.721753 w_k1_MAT_s010
                                 + 1.11505 w_k1_MSFT_s010
                                 + 0.888344 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.97977 w_k1_ADM_s010
                                 + 1.04905 w_k1_AXP_s010
                                 + 0.931963 w_k1_DIS_s010 + 1.004 w_k1_FDX_s010
                                 + 1.24742 w_k1_IBM_s010
                                 + 1.22851 w_k1_INTC_s010
                                 + 1.09589 w_k1_KO_s010 + 1.51154 w_k1_MAT_s010
                                 + 1.13631 w_k1_MSFT_s010
                                 + 1.02208 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 1.04582 w_k1_ADM_s010
                                 + 1.08293 w_k1_AXP_s010
                                 + 0.713015 w_k1_DIS_s010
                                 + 0.96783 w_k1_FDX_s010
                                 + 0.822052 w_k1_IBM_s010
                                 + 0.920097 w_k1_INTC_s010
                                 + 0.688075 w_k1_KO_s010
                                 + 0.764901 w_k1_MAT_s010
                                 + 0.816762 w_k1_MSFT_s010
                                 + 0.844853 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.11633 w_k1_ADM_s010
                                 + 1.30126 w_k1_AXP_s010
                                 + 1.48788 w_k1_DIS_s010
                                 + 1.20491 w_k1_FDX_s010
                                 + 1.20021 w_k1_IBM_s010
                                 + 1.33824 w_k1_INTC_s010
                                 + 1.11996 w_k1_KO_s010 + 1.03142 w_k1_MAT_s010
                                 + 1.29581 w_k1_MSFT_s010
                                 + 1.18204 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.783965 w_k1_ADM_s010
                                 + 0.744947 w_k1_AXP_s010
                                 + 0.666249 w_k1_DIS_s010
                                 + 0.760941 w_k1_FDX_s010
                                 + 0.880557 w_k1_IBM_s010
                                 + 0.785894 w_k1_INTC_s010
                                 + 0.750599 w_k1_KO_s010
                                 + 0.94838 w_k1_MAT_s010
                                 + 0.819625 w_k1_MSFT_s010
                                 + 0.891629 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.4396 w_k1_ADM_s010
                                 + 1.59401 w_k1_AXP_s010
                                 + 1.36723 w_k1_DIS_s010
                                 + 1.39854 w_k1_FDX_s010
                                 + 1.41108 w_k1_IBM_s010
                                 + 1.33531 w_k1_INTC_s010
                                 + 1.17107 w_k1_KO_s010 + 1.69864 w_k1_MAT_s010
                                 + 1.3247 w_k1_MSFT_s010 + 1.26993 w_k1_PG_s010
                                 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 0.868967 w_k1_ADM_s011
                                 + 0.805051 w_k1_AXP_s011
                                 + 0.834609 w_k1_DIS_s011
                                 + 0.8654 w_k1_FDX_s011
                                 + 0.831367 w_k1_IBM_s011
                                 + 1.07162 w_k1_INTC_s011
                                 + 0.877177 w_k1_KO_s011
                                 + 0.642763 w_k1_MAT_s011
                                 + 0.913224 w_k1_MSFT_s011
                                 + 0.82716 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.740981 w_k1_ADM_s011
                                 + 0.636851 w_k1_AXP_s011
                                 + 1.0803 w_k1_DIS_s011
                                 + 0.858916 w_k1_FDX_s011
                                 + 0.809842 w_k1_IBM_s011
                                 + 0.978598 w_k1_INTC_s011
                                 + 1.02371 w_k1_KO_s011
                                 + 0.718195 w_k1_MAT_s011
                                 + 1.14275 w_k1_MSFT_s011
                                 + 0.815232 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 0.966191 w_k1_ADM_s011
                                 + 1.24202 w_k1_AXP_s011
                                 + 1.10173 w_k1_DIS_s011
                                 + 1.04657 w_k1_FDX_s011
                                 + 1.40969 w_k1_IBM_s011
                                 + 1.1145 w_k1_INTC_s011 + 1.27721 w_k1_KO_s011
                                 + 1.14367 w_k1_MAT_s011
                                 + 1.07852 w_k1_MSFT_s011
                                 + 0.968748 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.36866 w_k1_ADM_s011
                                 + 1.57149 w_k1_AXP_s011
                                 + 1.50834 w_k1_DIS_s011
                                 + 1.44413 w_k1_FDX_s011
                                 + 1.04678 w_k1_IBM_s011
                                 + 1.30146 w_k1_INTC_s011
                                 + 1.28374 w_k1_KO_s011 + 0.97081 w_k1_MAT_s011
                                 + 1.25205 w_k1_MSFT_s011
                                 + 0.95098 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.889064 w_k1_ADM_s011
                                 + 0.704488 w_k1_AXP_s011
                                 + 0.790937 w_k1_DIS_s011
                                 + 0.808726 w_k1_FDX_s011
                                 + 1.03954 w_k1_IBM_s011
                                 + 0.916737 w_k1_INTC_s011
                                 + 0.978779 w_k1_KO_s011
                                 + 0.738586 w_k1_MAT_s011
                                 + 0.926343 w_k1_MSFT_s011
                                 + 0.859185 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 1.01453 w_k1_ADM_s011
                                 + 1.05052 w_k1_AXP_s011
                                 + 1.45197 w_k1_DIS_s011
                                 + 1.14153 w_k1_FDX_s011
                                 + 1.29255 w_k1_IBM_s011
                                 + 1.15398 w_k1_INTC_s011
                                 + 1.27824 w_k1_KO_s011 + 1.3795 w_k1_MAT_s011
                                 + 1.38141 w_k1_MSFT_s011
                                 + 0.908444 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.616027 w_k1_ADM_s011
                                 + 0.804457 w_k1_AXP_s011
                                 + 0.869762 w_k1_DIS_s011
                                 + 0.762628 w_k1_FDX_s011
                                 + 1.12649 w_k1_IBM_s011
                                 + 0.994232 w_k1_INTC_s011
                                 + 1.04495 w_k1_KO_s011
                                 + 0.917213 w_k1_MAT_s011
                                 + 0.93479 w_k1_MSFT_s011
                                 + 0.839304 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.961757 w_k1_ADM_s011
                                 + 1.15442 w_k1_AXP_s011
                                 + 1.03211 w_k1_DIS_s011
                                 + 1.09335 w_k1_FDX_s011
                                 + 0.755564 w_k1_IBM_s011
                                 + 1.0973 w_k1_INTC_s011
                                 + 0.939823 w_k1_KO_s011
                                 + 0.787354 w_k1_MAT_s011
                                 + 1.11727 w_k1_MSFT_s011
                                 + 0.784511 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.08731 w_k1_ADM_s012
                                 + 0.883345 w_k1_AXP_s012
                                 + 1.22376 w_k1_DIS_s012
                                 + 0.983195 w_k1_FDX_s012
                                 + 1.1098 w_k1_IBM_s012
                                 + 1.01308 w_k1_INTC_s012
                                 + 1.11878 w_k1_KO_s012
                                 + 0.950729 w_k1_MAT_s012
                                 + 1.13624 w_k1_MSFT_s012
                                 + 1.06737 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.880659 w_k1_ADM_s012
                                 + 1.22321 w_k1_AXP_s012
                                 + 1.50224 w_k1_DIS_s012
                                 + 1.23398 w_k1_FDX_s012
                                 + 0.868779 w_k1_IBM_s012
                                 + 1.13561 w_k1_INTC_s012
                                 + 1.01823 w_k1_KO_s012
                                 + 0.973957 w_k1_MAT_s012
                                 + 1.22464 w_k1_MSFT_s012
                                 + 1.12103 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.07679 w_k1_ADM_s012
                                 + 1.0473 w_k1_AXP_s012 + 1.20053 w_k1_DIS_s012
                                 + 1.02983 w_k1_FDX_s012
                                 + 1.46416 w_k1_IBM_s012
                                 + 1.04779 w_k1_INTC_s012 + 1.1754 w_k1_KO_s012
                                 + 1.18129 w_k1_MAT_s012
                                 + 1.09062 w_k1_MSFT_s012
                                 + 1.12826 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.950901 w_k1_ADM_s012
                                 + 0.671232 w_k1_AXP_s012
                                 + 0.643362 w_k1_DIS_s012
                                 + 0.719241 w_k1_FDX_s012
                                 + 0.867814 w_k1_IBM_s012
                                 + 0.864114 w_k1_INTC_s012
                                 + 0.802519 w_k1_KO_s012
                                 + 0.735305 w_k1_MAT_s012
                                 + 0.811848 w_k1_MSFT_s012
                                 + 0.917461 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.31357 w_k1_ADM_s012
                                 + 1.4602 w_k1_AXP_s012 + 1.64084 w_k1_DIS_s012
                                 + 1.39602 w_k1_FDX_s012
                                 + 1.28464 w_k1_IBM_s012
                                 + 1.60705 w_k1_INTC_s012 + 1.2816 w_k1_KO_s012
                                 + 1.2346 w_k1_MAT_s012
                                 + 1.37769 w_k1_MSFT_s012
                                 + 1.12784 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.4758 w_k1_ADM_s012
                                 + 1.29417 w_k1_AXP_s012
                                 + 1.06797 w_k1_DIS_s012
                                 + 1.15011 w_k1_FDX_s012
                                 + 0.938345 w_k1_IBM_s012
                                 + 1.10945 w_k1_INTC_s012
                                 + 0.895664 w_k1_KO_s012
                                 + 0.992366 w_k1_MAT_s012
                                 + 1.05449 w_k1_MSFT_s012
                                 + 1.02532 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.915351 w_k1_ADM_s012
                                 + 0.927407 w_k1_AXP_s012
                                 + 0.857312 w_k1_DIS_s012
                                 + 0.870585 w_k1_FDX_s012
                                 + 1.10229 w_k1_IBM_s012
                                 + 0.986683 w_k1_INTC_s012
                                 + 1.0457 w_k1_KO_s012 + 0.824603 w_k1_MAT_s012
                                 + 0.94309 w_k1_MSFT_s012
                                 + 0.890098 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.820493 w_k1_ADM_s012
                                 + 0.758192 w_k1_AXP_s012
                                 + 0.946855 w_k1_DIS_s012
                                 + 0.946839 w_k1_FDX_s012
                                 + 0.742768 w_k1_IBM_s012
                                 + 0.994485 w_k1_INTC_s012
                                 + 0.891534 w_k1_KO_s012
                                 + 0.821091 w_k1_MAT_s012
                                 + 1.11907 w_k1_MSFT_s012
                                 + 0.85555 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.15532 w_k1_ADM_s013
                                 + 1.02357 w_k1_AXP_s013
                                 + 1.01838 w_k1_DIS_s013
                                 + 1.03678 w_k1_FDX_s013
                                 + 0.842109 w_k1_IBM_s013
                                 + 0.865634 w_k1_INTC_s013
                                 + 0.945434 w_k1_KO_s013
                                 + 0.972039 w_k1_MAT_s013
                                 + 1.06578 w_k1_MSFT_s013
                                 + 0.903385 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.19227 w_k1_ADM_s013
                                 + 1.79501 w_k1_AXP_s013
                                 + 1.55645 w_k1_DIS_s013
                                 + 1.25376 w_k1_FDX_s013
                                 + 1.78805 w_k1_IBM_s013
                                 + 1.45611 w_k1_INTC_s013
                                 + 1.64271 w_k1_KO_s013 + 1.74258 w_k1_MAT_s013
                                 + 1.45103 w_k1_MSFT_s013
                                 + 1.30121 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.896663 w_k1_ADM_s013
                                 + 0.735321 w_k1_AXP_s013
                                 + 0.616416 w_k1_DIS_s013
                                 + 0.697817 w_k1_FDX_s013
                                 + 1.02582 w_k1_IBM_s013
                                 + 0.807111 w_k1_INTC_s013
                                 + 0.902523 w_k1_KO_s013
                                 + 0.88713 w_k1_MAT_s013
                                 + 0.716168 w_k1_MSFT_s013
                                 + 0.787223 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.21904 w_k1_ADM_s013
                                 + 0.880775 w_k1_AXP_s013
                                 + 0.868105 w_k1_DIS_s013
                                 + 0.929877 w_k1_FDX_s013
                                 + 0.650053 w_k1_IBM_s013
                                 + 0.919286 w_k1_INTC_s013
                                 + 0.849542 w_k1_KO_s013
                                 + 0.540636 w_k1_MAT_s013
                                 + 0.984107 w_k1_MSFT_s013
                                 + 0.84892 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.99191 w_k1_ADM_s013
                                 + 1.22466 w_k1_AXP_s013
                                 + 1.16092 w_k1_DIS_s013
                                 + 0.95095 w_k1_FDX_s013
                                 + 1.39714 w_k1_IBM_s013
                                 + 0.891713 w_k1_INTC_s013
                                 + 1.32508 w_k1_KO_s013 + 1.26995 w_k1_MAT_s013
                                 + 1.05945 w_k1_MSFT_s013
                                 + 0.922011 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.854384 w_k1_ADM_s013
                                 + 0.745114 w_k1_AXP_s013
                                 + 0.572732 w_k1_DIS_s013
                                 + 0.680896 w_k1_FDX_s013
                                 + 0.666605 w_k1_IBM_s013
                                 + 0.880212 w_k1_INTC_s013
                                 + 0.805149 w_k1_KO_s013
                                 + 0.738198 w_k1_MAT_s013
                                 + 0.861595 w_k1_MSFT_s013
                                 + 0.783246 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.48756 w_k1_ADM_s013
                                 + 1.44919 w_k1_AXP_s013
                                 + 1.37278 w_k1_DIS_s013
                                 + 1.29222 w_k1_FDX_s013
                                 + 1.08187 w_k1_IBM_s013
                                 + 1.08147 w_k1_INTC_s013
                                 + 1.29732 w_k1_KO_s013 + 1.38083 w_k1_MAT_s013
                                 + 1.24815 w_k1_MSFT_s013
                                 + 1.08966 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.971686 w_k1_ADM_s013
                                 + 0.746744 w_k1_AXP_s013
                                 + 0.846606 w_k1_DIS_s013
                                 + 0.851751 w_k1_FDX_s013
                                 + 1.01297 w_k1_IBM_s013
                                 + 0.929168 w_k1_INTC_s013
                                 + 1.03166 w_k1_KO_s013
                                 + 0.670625 w_k1_MAT_s013
                                 + 0.948399 w_k1_MSFT_s013
                                 + 0.851834 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.27046 w_k1_ADM_s014
                                 + 1.50715 w_k1_AXP_s014
                                 + 1.57079 w_k1_DIS_s014
                                 + 1.33847 w_k1_FDX_s014
                                 + 1.12783 w_k1_IBM_s014
                                 + 1.30018 w_k1_INTC_s014
                                 + 1.25594 w_k1_KO_s014 + 1.26235 w_k1_MAT_s014
                                 + 1.22367 w_k1_MSFT_s014
                                 + 1.17786 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.860576 w_k1_ADM_s014
                                 + 1.1763 w_k1_AXP_s014 + 1.03143 w_k1_DIS_s014
                                 + 1.13453 w_k1_FDX_s014
                                 + 1.00321 w_k1_IBM_s014
                                 + 1.17938 w_k1_INTC_s014
                                 + 0.954769 w_k1_KO_s014
                                 + 0.763035 w_k1_MAT_s014
                                 + 1.10874 w_k1_MSFT_s014
                                 + 1.02311 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.844657 w_k1_ADM_s014
                                 + 0.826513 w_k1_AXP_s014
                                 + 0.652263 w_k1_DIS_s014
                                 + 0.819028 w_k1_FDX_s014
                                 + 0.754621 w_k1_IBM_s014
                                 + 0.806888 w_k1_INTC_s014
                                 + 0.773778 w_k1_KO_s014
                                 + 0.585394 w_k1_MAT_s014
                                 + 0.824065 w_k1_MSFT_s014
                                 + 0.893182 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.779393 w_k1_ADM_s014
                                 + 0.787569 w_k1_AXP_s014
                                 + 0.650613 w_k1_DIS_s014
                                 + 0.810043 w_k1_FDX_s014
                                 + 1.01253 w_k1_IBM_s014
                                 + 0.802983 w_k1_INTC_s014
                                 + 0.786316 w_k1_KO_s014
                                 + 0.849697 w_k1_MAT_s014
                                 + 0.776017 w_k1_MSFT_s014
                                 + 0.915008 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.906895 w_k1_ADM_s014
                                 + 1.2137 w_k1_AXP_s014 + 1.21999 w_k1_DIS_s014
                                 + 1.07429 w_k1_FDX_s014
                                 + 1.51799 w_k1_IBM_s014
                                 + 1.03979 w_k1_INTC_s014 + 1.2245 w_k1_KO_s014
                                 + 1.29817 w_k1_MAT_s014
                                 + 1.12251 w_k1_MSFT_s014
                                 + 1.05237 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.597234 w_k1_ADM_s014
                                 + 0.835182 w_k1_AXP_s014
                                 + 0.883653 w_k1_DIS_s014
                                 + 0.842131 w_k1_FDX_s014
                                 + 1.19532 w_k1_IBM_s014
                                 + 0.971357 w_k1_INTC_s014
                                 + 1.00038 w_k1_KO_s014
                                 + 0.864946 w_k1_MAT_s014
                                 + 0.952741 w_k1_MSFT_s014
                                 + 1.00574 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.815996 w_k1_ADM_s014
                                 + 0.895975 w_k1_AXP_s014
                                 + 1.04945 w_k1_DIS_s014
                                 + 1.08508 w_k1_FDX_s014
                                 + 0.794079 w_k1_IBM_s014
                                 + 1.00975 w_k1_INTC_s014
                                 + 0.961632 w_k1_KO_s014
                                 + 0.702973 w_k1_MAT_s014
                                 + 1.09623 w_k1_MSFT_s014
                                 + 1.05308 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.758968 w_k1_ADM_s014
                                 + 1.30051 w_k1_AXP_s014
                                 + 1.26632 w_k1_DIS_s014
                                 + 1.01817 w_k1_FDX_s014
                                 + 1.84775 w_k1_IBM_s014
                                 + 1.3363 w_k1_INTC_s014 + 1.39597 w_k1_KO_s014
                                 + 1.47586 w_k1_MAT_s014
                                 + 0.960824 w_k1_MSFT_s014
                                 + 1.18519 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.943054 w_k1_ADM_s015
                                 + 1.37974 w_k1_AXP_s015
                                 + 1.13096 w_k1_DIS_s015
                                 + 1.10041 w_k1_FDX_s015
                                 + 1.21483 w_k1_IBM_s015
                                 + 0.939041 w_k1_INTC_s015
                                 + 1.09206 w_k1_KO_s015 + 1.63605 w_k1_MAT_s015
                                 + 1.04552 w_k1_MSFT_s015
                                 + 1.00918 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.84349 w_k1_ADM_s015
                                 + 1.27844 w_k1_AXP_s015
                                 + 1.43788 w_k1_DIS_s015 + 1.1262 w_k1_FDX_s015
                                 + 1.53689 w_k1_IBM_s015
                                 + 1.03617 w_k1_INTC_s015
                                 + 1.21868 w_k1_KO_s015 + 2.26789 w_k1_MAT_s015
                                 + 1.16976 w_k1_MSFT_s015
                                 + 1.12899 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 1.42576 w_k1_ADM_s015
                                 + 1.29286 w_k1_AXP_s015
                                 + 1.55744 w_k1_DIS_s015
                                 + 1.36557 w_k1_FDX_s015
                                 + 1.13159 w_k1_IBM_s015
                                 + 0.956472 w_k1_INTC_s015
                                 + 1.07303 w_k1_KO_s015 + 1.10673 w_k1_MAT_s015
                                 + 1.18396 w_k1_MSFT_s015
                                 + 1.21002 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.616202 w_k1_ADM_s015
                                 + 0.567998 w_k1_AXP_s015
                                 + 0.787747 w_k1_DIS_s015
                                 + 0.790831 w_k1_FDX_s015
                                 + 0.71745 w_k1_IBM_s015
                                 + 0.781202 w_k1_INTC_s015
                                 + 0.710453 w_k1_KO_s015
                                 + 0.688051 w_k1_MAT_s015
                                 + 0.780857 w_k1_MSFT_s015
                                 + 0.833485 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.949452 w_k1_ADM_s015
                                 + 0.848806 w_k1_AXP_s015
                                 + 0.882535 w_k1_DIS_s015
                                 + 0.888371 w_k1_FDX_s015
                                 + 0.959072 w_k1_IBM_s015
                                 + 0.742357 w_k1_INTC_s015
                                 + 0.869399 w_k1_KO_s015
                                 + 1.09343 w_k1_MAT_s015
                                 + 0.82977 w_k1_MSFT_s015
                                 + 0.938169 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.882052 w_k1_ADM_s015
                                 + 0.880327 w_k1_AXP_s015
                                 + 1.05095 w_k1_DIS_s015
                                 + 0.946519 w_k1_FDX_s015
                                 + 1.07673 w_k1_IBM_s015
                                 + 0.969599 w_k1_INTC_s015
                                 + 1.16478 w_k1_KO_s015 + 1.35261 w_k1_MAT_s015
                                 + 1.04893 w_k1_MSFT_s015
                                 + 0.961556 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.714276 w_k1_ADM_s015
                                 + 0.966853 w_k1_AXP_s015
                                 + 1.14598 w_k1_DIS_s015
                                 + 0.972829 w_k1_FDX_s015
                                 + 1.38288 w_k1_IBM_s015
                                 + 0.958958 w_k1_INTC_s015
                                 + 1.09429 w_k1_KO_s015 + 1.27334 w_k1_MAT_s015
                                 + 1.00707 w_k1_MSFT_s015
                                 + 1.12732 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.819104 w_k1_ADM_s015
                                 + 0.875507 w_k1_AXP_s015
                                 + 0.933522 w_k1_DIS_s015
                                 + 1.05564 w_k1_FDX_s015
                                 + 0.733497 w_k1_IBM_s015
                                 + 0.929676 w_k1_INTC_s015
                                 + 0.757356 w_k1_KO_s015
                                 + 0.800957 w_k1_MAT_s015
                                 + 1.06585 w_k1_MSFT_s015
                                 + 1.00724 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 0.866697 w_k1_ADM_s016
                                 + 0.888415 w_k1_AXP_s016
                                 + 0.915867 w_k1_DIS_s016
                                 + 0.880805 w_k1_FDX_s016
                                 + 0.939493 w_k1_IBM_s016
                                 + 0.845977 w_k1_INTC_s016
                                 + 0.954509 w_k1_KO_s016
                                 + 0.933514 w_k1_MAT_s016
                                 + 0.835528 w_k1_MSFT_s016
                                 + 0.835476 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.05139 w_k1_ADM_s016
                                 + 1.21348 w_k1_AXP_s016
                                 + 1.43282 w_k1_DIS_s016
                                 + 1.11351 w_k1_FDX_s016
                                 + 1.47269 w_k1_IBM_s016
                                 + 0.997909 w_k1_INTC_s016
                                 + 1.43038 w_k1_KO_s016 + 1.72126 w_k1_MAT_s016
                                 + 1.03121 w_k1_MSFT_s016
                                 + 1.09477 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.863162 w_k1_ADM_s016
                                 + 0.783679 w_k1_AXP_s016
                                 + 0.770511 w_k1_DIS_s016
                                 + 0.772828 w_k1_FDX_s016
                                 + 1.29344 w_k1_IBM_s016
                                 + 0.825562 w_k1_INTC_s016
                                 + 0.985899 w_k1_KO_s016
                                 + 1.16165 w_k1_MAT_s016
                                 + 0.731486 w_k1_MSFT_s016
                                 + 0.845855 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.14289 w_k1_ADM_s016
                                 + 0.981267 w_k1_AXP_s016
                                 + 1.21831 w_k1_DIS_s016
                                 + 0.945652 w_k1_FDX_s016
                                 + 1.69077 w_k1_IBM_s016
                                 + 0.953263 w_k1_INTC_s016
                                 + 1.25496 w_k1_KO_s016 + 1.37217 w_k1_MAT_s016
                                 + 0.805251 w_k1_MSFT_s016
                                 + 1.06295 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.962792 w_k1_ADM_s016
                                 + 0.685297 w_k1_AXP_s016
                                 + 0.566488 w_k1_DIS_s016
                                 + 0.753036 w_k1_FDX_s016
                                 + 0.747197 w_k1_IBM_s016
                                 + 0.712248 w_k1_INTC_s016
                                 + 0.72233 w_k1_KO_s016 + 0.54333 w_k1_MAT_s016
                                 + 0.668186 w_k1_MSFT_s016
                                 + 0.840094 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.07368 w_k1_ADM_s016
                                 + 1.07619 w_k1_AXP_s016
                                 + 0.937081 w_k1_DIS_s016
                                 + 1.07938 w_k1_FDX_s016
                                 + 0.79423 w_k1_IBM_s016
                                 + 1.05645 w_k1_INTC_s016
                                 + 0.96702 w_k1_KO_s016 + 1.10979 w_k1_MAT_s016
                                 + 0.941035 w_k1_MSFT_s016
                                 + 0.961653 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 1.39702 w_k1_ADM_s016
                                 + 1.70128 w_k1_AXP_s016
                                 + 1.44889 w_k1_DIS_s016
                                 + 1.46348 w_k1_FDX_s016
                                 + 1.04812 w_k1_IBM_s016
                                 + 1.1262 w_k1_INTC_s016 + 1.22265 w_k1_KO_s016
                                 + 1.22935 w_k1_MAT_s016
                                 + 1.16587 w_k1_MSFT_s016
                                 + 1.15723 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.17887 w_k1_ADM_s016
                                 + 0.91855 w_k1_AXP_s016
                                 + 1.34061 w_k1_DIS_s016
                                 + 1.15547 w_k1_FDX_s016
                                 + 0.964402 w_k1_IBM_s016
                                 + 1.00373 w_k1_INTC_s016
                                 + 1.10293 w_k1_KO_s016
                                 + 0.742626 w_k1_MAT_s016
                                 + 1.10491 w_k1_MSFT_s016
                                 + 1.03249 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.606919 w_k1_ADM_s017
                                 + 1.03074 w_k1_AXP_s017
                                 + 1.02501 w_k1_DIS_s017
                                 + 0.907892 w_k1_FDX_s017
                                 + 1.33933 w_k1_IBM_s017
                                 + 0.938562 w_k1_INTC_s017
                                 + 1.05996 w_k1_KO_s017 + 1.26471 w_k1_MAT_s017
                                 + 0.91042 w_k1_MSFT_s017
                                 + 0.99405 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.871246 w_k1_ADM_s017
                                 + 0.968348 w_k1_AXP_s017
                                 + 1.15613 w_k1_DIS_s017
                                 + 1.05578 w_k1_FDX_s017
                                 + 0.706646 w_k1_IBM_s017
                                 + 1.12928 w_k1_INTC_s017
                                 + 0.90324 w_k1_KO_s017
                                 + 0.616082 w_k1_MAT_s017
                                 + 1.07069 w_k1_MSFT_s017
                                 + 0.952606 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.675737 w_k1_ADM_s017
                                 + 0.903515 w_k1_AXP_s017
                                 + 0.845198 w_k1_DIS_s017
                                 + 0.930714 w_k1_FDX_s017
                                 + 0.901151 w_k1_IBM_s017
                                 + 1.26335 w_k1_INTC_s017
                                 + 0.985459 w_k1_KO_s017
                                 + 0.994642 w_k1_MAT_s017
                                 + 0.872634 w_k1_MSFT_s017
                                 + 1.06336 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.52787 w_k1_ADM_s017
                                 + 1.37164 w_k1_AXP_s017
                                 + 1.67549 w_k1_DIS_s017
                                 + 1.38815 w_k1_FDX_s017
                                 + 0.910051 w_k1_IBM_s017
                                 + 1.54519 w_k1_INTC_s017
                                 + 1.26886 w_k1_KO_s017 + 1.00738 w_k1_MAT_s017
                                 + 1.37369 w_k1_MSFT_s017
                                 + 1.28498 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.681547 w_k1_ADM_s017
                                 + 0.634316 w_k1_AXP_s017
                                 + 0.585715 w_k1_DIS_s017
                                 + 0.732702 w_k1_FDX_s017
                                 + 0.756089 w_k1_IBM_s017
                                 + 0.834359 w_k1_INTC_s017
                                 + 0.682405 w_k1_KO_s017
                                 + 0.444445 w_k1_MAT_s017
                                 + 0.675488 w_k1_MSFT_s017
                                 + 0.758192 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.914091 w_k1_ADM_s017
                                 + 1.2451 w_k1_AXP_s017 + 1.37005 w_k1_DIS_s017
                                 + 1.18398 w_k1_FDX_s017
                                 + 1.09225 w_k1_IBM_s017
                                 + 1.42978 w_k1_INTC_s017
                                 + 1.15845 w_k1_KO_s017
                                 + 0.961491 w_k1_MAT_s017
                                 + 1.14078 w_k1_MSFT_s017
                                 + 1.19317 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.857634 w_k1_ADM_s017
                                 + 1.08326 w_k1_AXP_s017
                                 + 0.956105 w_k1_DIS_s017
                                 + 1.00309 w_k1_FDX_s017
                                 + 1.06797 w_k1_IBM_s017
                                 + 1.20259 w_k1_INTC_s017 + 1.1274 w_k1_KO_s017
                                 + 1.27374 w_k1_MAT_s017
                                 + 0.939232 w_k1_MSFT_s017
                                 + 1.0773 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.09115 w_k1_ADM_s017
                                 + 1.52527 w_k1_AXP_s017
                                 + 1.63113 w_k1_DIS_s017
                                 + 1.26517 w_k1_FDX_s017
                                 + 1.63194 w_k1_IBM_s017
                                 + 1.81011 w_k1_INTC_s017
                                 + 1.47702 w_k1_KO_s017 + 1.55124 w_k1_MAT_s017
                                 + 1.41384 w_k1_MSFT_s017
                                 + 1.36285 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.20922 w_k1_ADM_s100
                                 + 1.15871 w_k1_AXP_s100
                                 + 1.39397 w_k1_DIS_s100
                                 + 1.17311 w_k1_FDX_s100
                                 + 0.987203 w_k1_IBM_s100
                                 + 1.10399 w_k1_INTC_s100
                                 + 1.15664 w_k1_KO_s100
                                 + 0.879601 w_k1_MAT_s100
                                 + 1.0626 w_k1_MSFT_s100 + 1.06449 w_k1_PG_s100
                                 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.10942 w_k1_ADM_s100
                                 + 0.874576 w_k1_AXP_s100
                                 + 0.869964 w_k1_DIS_s100
                                 + 0.909647 w_k1_FDX_s100
                                 + 1.00725 w_k1_IBM_s100
                                 + 0.584875 w_k1_INTC_s100
                                 + 0.887799 w_k1_KO_s100
                                 + 0.99636 w_k1_MAT_s100
                                 + 0.711038 w_k1_MSFT_s100
                                 + 0.972546 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 0.968529 w_k1_ADM_s100
                                 + 1.13918 w_k1_AXP_s100
                                 + 1.14568 w_k1_DIS_s100
                                 + 1.04276 w_k1_FDX_s100
                                 + 1.41045 w_k1_IBM_s100
                                 + 0.900059 w_k1_INTC_s100
                                 + 1.1612 w_k1_KO_s100 + 1.29418 w_k1_MAT_s100
                                 + 1.10941 w_k1_MSFT_s100
                                 + 1.09566 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 1.12829 w_k1_ADM_s100
                                 + 1.074 w_k1_AXP_s100 + 1.44563 w_k1_DIS_s100
                                 + 1.01847 w_k1_FDX_s100
                                 + 1.87148 w_k1_IBM_s100
                                 + 0.933832 w_k1_INTC_s100
                                 + 1.48819 w_k1_KO_s100 + 1.94261 w_k1_MAT_s100
                                 + 1.16747 w_k1_MSFT_s100
                                 + 1.22198 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.633545 w_k1_ADM_s100
                                 + 0.68545 w_k1_AXP_s100
                                 + 0.616362 w_k1_DIS_s100
                                 + 0.805277 w_k1_FDX_s100
                                 + 0.771229 w_k1_IBM_s100
                                 + 0.545582 w_k1_INTC_s100
                                 + 0.686404 w_k1_KO_s100
                                 + 1.06498 w_k1_MAT_s100
                                 + 0.616596 w_k1_MSFT_s100
                                 + 0.750948 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.80024 w_k1_ADM_s100
                                 + 1.45293 w_k1_AXP_s100
                                 + 1.60302 w_k1_DIS_s100
                                 + 1.47839 w_k1_FDX_s100
                                 + 1.08159 w_k1_IBM_s100
                                 + 1.2427 w_k1_INTC_s100 + 1.33472 w_k1_KO_s100
                                 + 1.09845 w_k1_MAT_s100
                                 + 1.27848 w_k1_MSFT_s100
                                 + 1.13715 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.14683 w_k1_ADM_s100
                                 + 0.695163 w_k1_AXP_s100
                                 + 0.951506 w_k1_DIS_s100
                                 + 0.988995 w_k1_FDX_s100
                                 + 0.679326 w_k1_IBM_s100
                                 + 0.935252 w_k1_INTC_s100
                                 + 0.87113 w_k1_KO_s100
                                 + 0.666807 w_k1_MAT_s100
                                 + 1.08943 w_k1_MSFT_s100
                                 + 0.88696 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.21011 w_k1_ADM_s100
                                 + 1.1191 w_k1_AXP_s100
                                 + 0.987284 w_k1_DIS_s100
                                 + 1.1168 w_k1_FDX_s100 + 1.04915 w_k1_IBM_s100
                                 + 1.07839 w_k1_INTC_s100
                                 + 0.991354 w_k1_KO_s100
                                 + 1.14469 w_k1_MAT_s100
                                 + 1.07468 w_k1_MSFT_s100
                                 + 1.08798 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 1.56385 w_k1_ADM_s101
                                 + 1.1236 w_k1_AXP_s101 + 1.06002 w_k1_DIS_s101
                                 + 0.930661 w_k1_FDX_s101
                                 + 1.30278 w_k1_IBM_s101
                                 + 0.94538 w_k1_INTC_s101
                                 + 1.05716 w_k1_KO_s101
                                 + 0.982823 w_k1_MAT_s101
                                 + 0.800863 w_k1_MSFT_s101
                                 + 0.98003 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.02978 w_k1_ADM_s101
                                 + 1.00947 w_k1_AXP_s101
                                 + 0.82961 w_k1_DIS_s101
                                 + 1.06395 w_k1_FDX_s101
                                 + 0.932167 w_k1_IBM_s101
                                 + 1.11656 w_k1_INTC_s101
                                 + 0.764203 w_k1_KO_s101
                                 + 1.10796 w_k1_MAT_s101
                                 + 0.961563 w_k1_MSFT_s101
                                 + 0.949071 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.08299 w_k1_ADM_s101
                                 + 0.890037 w_k1_AXP_s101
                                 + 0.851229 w_k1_DIS_s101
                                 + 0.845766 w_k1_FDX_s101
                                 + 0.92574 w_k1_IBM_s101
                                 + 0.913809 w_k1_INTC_s101
                                 + 0.943481 w_k1_KO_s101
                                 + 0.848814 w_k1_MAT_s101
                                 + 0.854147 w_k1_MSFT_s101
                                 + 0.90694 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 1.27462 w_k1_ADM_s101
                                 + 0.857564 w_k1_AXP_s101
                                 + 0.870293 w_k1_DIS_s101
                                 + 1.00113 w_k1_FDX_s101
                                 + 0.715469 w_k1_IBM_s101
                                 + 0.943497 w_k1_INTC_s101
                                 + 0.701008 w_k1_KO_s101
                                 + 0.49872 w_k1_MAT_s101
                                 + 0.743129 w_k1_MSFT_s101
                                 + 0.786259 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.65019 w_k1_ADM_s101
                                 + 1.08959 w_k1_AXP_s101
                                 + 1.18692 w_k1_DIS_s101
                                 + 1.17936 w_k1_FDX_s101
                                 + 0.883751 w_k1_IBM_s101
                                 + 1.21701 w_k1_INTC_s101
                                 + 1.12102 w_k1_KO_s101
                                 + 0.850603 w_k1_MAT_s101
                                 + 1.18396 w_k1_MSFT_s101
                                 + 1.11304 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.92913 w_k1_ADM_s101
                                 + 0.587866 w_k1_AXP_s101
                                 + 0.507668 w_k1_DIS_s101
                                 + 0.63519 w_k1_FDX_s101
                                 + 0.737521 w_k1_IBM_s101
                                 + 0.684902 w_k1_INTC_s101
                                 + 0.609675 w_k1_KO_s101
                                 + 0.508815 w_k1_MAT_s101
                                 + 0.625437 w_k1_MSFT_s101
                                 + 0.765116 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 2.08232 w_k1_ADM_s101
                                 + 1.49033 w_k1_AXP_s101
                                 + 1.75677 w_k1_DIS_s101
                                 + 1.31053 w_k1_FDX_s101
                                 + 1.46637 w_k1_IBM_s101
                                 + 1.05289 w_k1_INTC_s101
                                 + 1.47667 w_k1_KO_s101 + 1.13004 w_k1_MAT_s101
                                 + 1.09867 w_k1_MSFT_s101
                                 + 1.26059 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 1.72405 w_k1_ADM_s101
                                 + 1.25863 w_k1_AXP_s101
                                 + 1.35083 w_k1_DIS_s101
                                 + 1.16436 w_k1_FDX_s101
                                 + 1.15131 w_k1_IBM_s101
                                 + 1.08183 w_k1_INTC_s101
                                 + 1.17088 w_k1_KO_s101 + 1.04705 w_k1_MAT_s101
                                 + 1.07073 w_k1_MSFT_s101
                                 + 1.18319 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.19239 w_k1_ADM_s102
                                 + 1.57519 w_k1_AXP_s102 + 1.7455 w_k1_DIS_s102
                                 + 1.31128 w_k1_FDX_s102
                                 + 1.44387 w_k1_IBM_s102
                                 + 1.35929 w_k1_INTC_s102
                                 + 1.42256 w_k1_KO_s102 + 1.6863 w_k1_MAT_s102
                                 + 1.43407 w_k1_MSFT_s102
                                 + 1.18828 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.848436 w_k1_ADM_s102
                                 + 0.70848 w_k1_AXP_s102
                                 + 0.700625 w_k1_DIS_s102
                                 + 0.820687 w_k1_FDX_s102
                                 + 0.713039 w_k1_IBM_s102
                                 + 0.99514 w_k1_INTC_s102
                                 + 0.746004 w_k1_KO_s102
                                 + 0.558039 w_k1_MAT_s102
                                 + 0.770136 w_k1_MSFT_s102
                                 + 0.805533 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.07005 w_k1_ADM_s102
                                 + 0.976483 w_k1_AXP_s102
                                 + 0.826287 w_k1_DIS_s102
                                 + 1.0277 w_k1_FDX_s102
                                 + 0.734249 w_k1_IBM_s102
                                 + 1.02058 w_k1_INTC_s102
                                 + 0.73596 w_k1_KO_s102
                                 + 0.867062 w_k1_MAT_s102
                                 + 0.890489 w_k1_MSFT_s102
                                 + 0.825273 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.48582 w_k1_ADM_s102
                                 + 1.24708 w_k1_AXP_s102
                                 + 1.23135 w_k1_DIS_s102
                                 + 1.01316 w_k1_FDX_s102
                                 + 1.62413 w_k1_IBM_s102
                                 + 0.942977 w_k1_INTC_s102
                                 + 1.33261 w_k1_KO_s102 + 1.53079 w_k1_MAT_s102
                                 + 1.03098 w_k1_MSFT_s102
                                 + 1.09453 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.02786 w_k1_ADM_s102
                                 + 1.27166 w_k1_AXP_s102
                                 + 1.13989 w_k1_DIS_s102
                                 + 1.08453 w_k1_FDX_s102
                                 + 1.13091 w_k1_IBM_s102
                                 + 0.98683 w_k1_INTC_s102
                                 + 0.993343 w_k1_KO_s102
                                 + 1.35329 w_k1_MAT_s102
                                 + 0.998938 w_k1_MSFT_s102
                                 + 1.05571 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.3518 w_k1_ADM_s102
                                 + 1.48667 w_k1_AXP_s102 + 1.2635 w_k1_DIS_s102
                                 + 1.27207 w_k1_FDX_s102
                                 + 0.805437 w_k1_IBM_s102
                                 + 1.2026 w_k1_INTC_s102
                                 + 0.919049 w_k1_KO_s102
                                 + 1.16506 w_k1_MAT_s102
                                 + 1.27754 w_k1_MSFT_s102
                                 + 1.12036 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.843987 w_k1_ADM_s102
                                 + 0.833793 w_k1_AXP_s102
                                 + 1.20564 w_k1_DIS_s102
                                 + 0.972498 w_k1_FDX_s102
                                 + 0.98081 w_k1_IBM_s102
                                 + 1.09604 w_k1_INTC_s102
                                 + 0.960889 w_k1_KO_s102
                                 + 0.813724 w_k1_MAT_s102
                                 + 1.21871 w_k1_MSFT_s102
                                 + 0.936569 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.794725 w_k1_ADM_s102
                                 + 0.894202 w_k1_AXP_s102
                                 + 0.827697 w_k1_DIS_s102
                                 + 0.840473 w_k1_FDX_s102
                                 + 1.08229 w_k1_IBM_s102
                                 + 0.821609 w_k1_INTC_s102
                                 + 0.931382 w_k1_KO_s102
                                 + 1.27713 w_k1_MAT_s102
                                 + 0.887125 w_k1_MSFT_s102
                                 + 1.02249 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.11568 w_k1_ADM_s103
                                 + 0.831489 w_k1_AXP_s103
                                 + 0.846666 w_k1_DIS_s103
                                 + 0.984221 w_k1_FDX_s103
                                 + 0.75792 w_k1_IBM_s103
                                 + 1.16092 w_k1_INTC_s103
                                 + 0.926252 w_k1_KO_s103
                                 + 0.729283 w_k1_MAT_s103
                                 + 1.23303 w_k1_MSFT_s103
                                 + 1.08414 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.38165 w_k1_ADM_s103
                                 + 1.02191 w_k1_AXP_s103 + 1.0488 w_k1_DIS_s103
                                 + 1.10965 w_k1_FDX_s103
                                 + 0.914889 w_k1_IBM_s103
                                 + 1.28959 w_k1_INTC_s103 + 1.1049 w_k1_KO_s103
                                 + 1.19854 w_k1_MAT_s103
                                 + 1.29609 w_k1_MSFT_s103
                                 + 1.13627 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.865668 w_k1_ADM_s103
                                 + 0.772265 w_k1_AXP_s103
                                 + 0.6886 w_k1_DIS_s103
                                 + 0.708973 w_k1_FDX_s103
                                 + 1.17324 w_k1_IBM_s103
                                 + 0.83592 w_k1_INTC_s103
                                 + 0.986303 w_k1_KO_s103
                                 + 1.22925 w_k1_MAT_s103
                                 + 0.846212 w_k1_MSFT_s103
                                 + 0.96499 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.973925 w_k1_ADM_s103
                                 + 1.0489 w_k1_AXP_s103 + 1.04444 w_k1_DIS_s103
                                 + 0.971855 w_k1_FDX_s103
                                 + 1.61341 w_k1_IBM_s103
                                 + 1.16418 w_k1_INTC_s103
                                 + 1.04646 w_k1_KO_s103 + 1.25564 w_k1_MAT_s103
                                 + 0.94251 w_k1_MSFT_s103
                                 + 0.875376 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.989683 w_k1_ADM_s103
                                 + 0.910526 w_k1_AXP_s103
                                 + 1.16647 w_k1_DIS_s103
                                 + 1.09127 w_k1_FDX_s103
                                 + 1.01987 w_k1_IBM_s103
                                 + 1.31166 w_k1_INTC_s103
                                 + 1.19441 w_k1_KO_s103
                                 + 0.803955 w_k1_MAT_s103
                                 + 1.26104 w_k1_MSFT_s103
                                 + 1.03885 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.754694 w_k1_ADM_s103
                                 + 0.682317 w_k1_AXP_s103
                                 + 0.54154 w_k1_DIS_s103
                                 + 0.704158 w_k1_FDX_s103
                                 + 0.789352 w_k1_IBM_s103
                                 + 0.90975 w_k1_INTC_s103
                                 + 0.768316 w_k1_KO_s103
                                 + 0.956934 w_k1_MAT_s103
                                 + 0.820197 w_k1_MSFT_s103
                                 + 0.823873 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.872853 w_k1_ADM_s103
                                 + 0.817518 w_k1_AXP_s103
                                 + 0.619616 w_k1_DIS_s103
                                 + 0.794007 w_k1_FDX_s103
                                 + 0.976532 w_k1_IBM_s103
                                 + 1.08852 w_k1_INTC_s103
                                 + 0.823882 w_k1_KO_s103
                                 + 1.07689 w_k1_MAT_s103
                                 + 0.89596 w_k1_MSFT_s103
                                 + 0.805314 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.08936 w_k1_ADM_s103
                                 + 1.65535 w_k1_AXP_s103
                                 + 1.14966 w_k1_DIS_s103
                                 + 1.26054 w_k1_FDX_s103
                                 + 1.17279 w_k1_IBM_s103
                                 + 1.58277 w_k1_INTC_s103 + 1.1906 w_k1_KO_s103
                                 + 1.64746 w_k1_MAT_s103
                                 + 1.24197 w_k1_MSFT_s103
                                 + 1.19174 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.14558 w_k1_ADM_s104
                                 + 1.5123 w_k1_AXP_s104
                                 + 0.937913 w_k1_DIS_s104
                                 + 1.14113 w_k1_FDX_s104
                                 + 1.54494 w_k1_IBM_s104
                                 + 1.12284 w_k1_INTC_s104
                                 + 1.15558 w_k1_KO_s104 + 1.35936 w_k1_MAT_s104
                                 + 0.960337 w_k1_MSFT_s104
                                 + 1.06137 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.990452 w_k1_ADM_s104
                                 + 0.764363 w_k1_AXP_s104
                                 + 0.689462 w_k1_DIS_s104
                                 + 0.812866 w_k1_FDX_s104
                                 + 0.788121 w_k1_IBM_s104
                                 + 0.9104 w_k1_INTC_s104
                                 + 0.696131 w_k1_KO_s104
                                 + 0.439493 w_k1_MAT_s104
                                 + 0.712329 w_k1_MSFT_s104
                                 + 0.768599 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 1.29052 w_k1_ADM_s104
                                 + 1.0547 w_k1_AXP_s104 + 1.20483 w_k1_DIS_s104
                                 + 1.07803 w_k1_FDX_s104
                                 + 1.73345 w_k1_IBM_s104
                                 + 1.05051 w_k1_INTC_s104
                                 + 1.42596 w_k1_KO_s104 + 1.15434 w_k1_MAT_s104
                                 + 1.02817 w_k1_MSFT_s104
                                 + 1.20983 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.973797 w_k1_ADM_s104
                                 + 1.18825 w_k1_AXP_s104
                                 + 1.13282 w_k1_DIS_s104
                                 + 1.09072 w_k1_FDX_s104
                                 + 1.15321 w_k1_IBM_s104
                                 + 1.36517 w_k1_INTC_s104
                                 + 1.17789 w_k1_KO_s104
                                 + 0.753307 w_k1_MAT_s104
                                 + 1.0585 w_k1_MSFT_s104
                                 + 0.949047 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.929156 w_k1_ADM_s104
                                 + 0.764732 w_k1_AXP_s104
                                 + 0.680811 w_k1_DIS_s104
                                 + 0.829532 w_k1_FDX_s104
                                 + 1.13953 w_k1_IBM_s104
                                 + 1.0209 w_k1_INTC_s104
                                 + 0.878337 w_k1_KO_s104
                                 + 0.554226 w_k1_MAT_s104
                                 + 0.821186 w_k1_MSFT_s104
                                 + 0.813366 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.53481 w_k1_ADM_s104
                                 + 1.41804 w_k1_AXP_s104
                                 + 1.55721 w_k1_DIS_s104
                                 + 1.48234 w_k1_FDX_s104
                                 + 1.21552 w_k1_IBM_s104
                                 + 1.48787 w_k1_INTC_s104
                                 + 1.25379 w_k1_KO_s104
                                 + 0.667741 w_k1_MAT_s104
                                 + 1.26426 w_k1_MSFT_s104
                                 + 1.07911 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.802545 w_k1_ADM_s104
                                 + 0.94293 w_k1_AXP_s104
                                 + 0.957412 w_k1_DIS_s104
                                 + 0.847855 w_k1_FDX_s104
                                 + 1.22181 w_k1_IBM_s104
                                 + 1.01592 w_k1_INTC_s104
                                 + 1.03592 w_k1_KO_s104
                                 + 0.750044 w_k1_MAT_s104
                                 + 0.893622 w_k1_MSFT_s104
                                 + 0.931995 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.14562 w_k1_ADM_s104
                                 + 1.17543 w_k1_AXP_s104 + 1.1843 w_k1_DIS_s104
                                 + 1.20938 w_k1_FDX_s104
                                 + 0.914845 w_k1_IBM_s104
                                 + 1.04981 w_k1_INTC_s104
                                 + 0.88671 w_k1_KO_s104
                                 + 0.873559 w_k1_MAT_s104
                                 + 1.05841 w_k1_MSFT_s104
                                 + 1.18199 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.0548 w_k1_ADM_s105
                                 + 1.39795 w_k1_AXP_s105
                                 + 1.53428 w_k1_DIS_s105
                                 + 1.15548 w_k1_FDX_s105
                                 + 1.41175 w_k1_IBM_s105
                                 + 1.18126 w_k1_INTC_s105
                                 + 1.41047 w_k1_KO_s105 + 1.68502 w_k1_MAT_s105
                                 + 1.19759 w_k1_MSFT_s105 + 1.2844 w_k1_PG_s105
                                 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 0.758734 w_k1_ADM_s105
                                 + 0.738521 w_k1_AXP_s105
                                 + 0.683728 w_k1_DIS_s105
                                 + 0.699035 w_k1_FDX_s105
                                 + 0.793835 w_k1_IBM_s105
                                 + 1.03347 w_k1_INTC_s105
                                 + 0.793935 w_k1_KO_s105
                                 + 0.624248 w_k1_MAT_s105
                                 + 0.896673 w_k1_MSFT_s105
                                 + 0.813538 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.939527 w_k1_ADM_s105
                                 + 1.16293 w_k1_AXP_s105
                                 + 1.17089 w_k1_DIS_s105
                                 + 1.07076 w_k1_FDX_s105
                                 + 1.13806 w_k1_IBM_s105
                                 + 1.40878 w_k1_INTC_s105
                                 + 1.24384 w_k1_KO_s105 + 1.18671 w_k1_MAT_s105
                                 + 1.08188 w_k1_MSFT_s105 + 1.075 w_k1_PG_s105
                                 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.63563 w_k1_ADM_s105
                                 + 0.835633 w_k1_AXP_s105
                                 + 0.83694 w_k1_DIS_s105
                                 + 0.881343 w_k1_FDX_s105
                                 + 0.83034 w_k1_IBM_s105
                                 + 0.95372 w_k1_INTC_s105
                                 + 0.800713 w_k1_KO_s105
                                 + 0.759737 w_k1_MAT_s105
                                 + 0.845192 w_k1_MSFT_s105
                                 + 0.802074 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.986825 w_k1_ADM_s105
                                 + 0.908938 w_k1_AXP_s105
                                 + 1.10597 w_k1_DIS_s105
                                 + 1.08098 w_k1_FDX_s105
                                 + 0.688641 w_k1_IBM_s105
                                 + 1.22479 w_k1_INTC_s105
                                 + 1.01239 w_k1_KO_s105
                                 + 0.556987 w_k1_MAT_s105
                                 + 1.23006 w_k1_MSFT_s105
                                 + 1.02815 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.12573 w_k1_ADM_s105
                                 + 1.21749 w_k1_AXP_s105
                                 + 1.07993 w_k1_DIS_s105
                                 + 1.07467 w_k1_FDX_s105
                                 + 0.868955 w_k1_IBM_s105
                                 + 1.09507 w_k1_INTC_s105
                                 + 0.981899 w_k1_KO_s105
                                 + 0.940339 w_k1_MAT_s105
                                 + 1.08494 w_k1_MSFT_s105
                                 + 0.994237 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.929136 w_k1_ADM_s105
                                 + 0.792484 w_k1_AXP_s105
                                 + 0.646122 w_k1_DIS_s105
                                 + 0.986518 w_k1_FDX_s105
                                 + 0.563353 w_k1_IBM_s105
                                 + 1.02804 w_k1_INTC_s105
                                 + 0.634743 w_k1_KO_s105
                                 + 0.436347 w_k1_MAT_s105
                                 + 0.960285 w_k1_MSFT_s105
                                 + 0.832839 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.970516 w_k1_ADM_s105
                                 + 0.730341 w_k1_AXP_s105
                                 + 0.769136 w_k1_DIS_s105
                                 + 0.704751 w_k1_FDX_s105
                                 + 1.06447 w_k1_IBM_s105
                                 + 0.904604 w_k1_INTC_s105
                                 + 1.01976 w_k1_KO_s105 + 1.0638 w_k1_MAT_s105
                                 + 0.960774 w_k1_MSFT_s105
                                 + 0.962392 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.781441 w_k1_ADM_s106
                                 + 0.786204 w_k1_AXP_s106
                                 + 1.08555 w_k1_DIS_s106
                                 + 0.917651 w_k1_FDX_s106
                                 + 0.868276 w_k1_IBM_s106
                                 + 0.900384 w_k1_INTC_s106
                                 + 0.96935 w_k1_KO_s106
                                 + 0.852522 w_k1_MAT_s106
                                 + 1.10468 w_k1_MSFT_s106
                                 + 1.01628 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.00981 w_k1_ADM_s106
                                 + 0.89147 w_k1_AXP_s106
                                 + 0.735659 w_k1_DIS_s106
                                 + 0.784237 w_k1_FDX_s106
                                 + 0.946404 w_k1_IBM_s106
                                 + 0.865133 w_k1_INTC_s106
                                 + 0.865153 w_k1_KO_s106 + 0.9112 w_k1_MAT_s106
                                 + 0.816139 w_k1_MSFT_s106
                                 + 0.875506 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.06785 w_k1_ADM_s106
                                 + 1.17629 w_k1_AXP_s106
                                 + 1.26454 w_k1_DIS_s106
                                 + 1.05962 w_k1_FDX_s106
                                 + 1.19149 w_k1_IBM_s106
                                 + 0.952966 w_k1_INTC_s106
                                 + 1.25579 w_k1_KO_s106 + 0.9923 w_k1_MAT_s106
                                 + 1.1729 w_k1_MSFT_s106 + 1.07377 w_k1_PG_s106
                                 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.16305 w_k1_ADM_s106
                                 + 0.850545 w_k1_AXP_s106
                                 + 0.880831 w_k1_DIS_s106
                                 + 0.805497 w_k1_FDX_s106
                                 + 1.15192 w_k1_IBM_s106
                                 + 0.976854 w_k1_INTC_s106
                                 + 1.17559 w_k1_KO_s106
                                 + 0.955249 w_k1_MAT_s106
                                 + 0.909734 w_k1_MSFT_s106
                                 + 0.951429 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.993742 w_k1_ADM_s106
                                 + 1.27088 w_k1_AXP_s106
                                 + 1.50242 w_k1_DIS_s106 + 1.1242 w_k1_FDX_s106
                                 + 1.59895 w_k1_IBM_s106
                                 + 1.18265 w_k1_INTC_s106 + 1.509 w_k1_KO_s106
                                 + 1.56671 w_k1_MAT_s106
                                 + 1.12718 w_k1_MSFT_s106
                                 + 1.09927 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.915875 w_k1_ADM_s106
                                 + 0.894185 w_k1_AXP_s106
                                 + 0.926591 w_k1_DIS_s106
                                 + 1.03315 w_k1_FDX_s106
                                 + 0.703617 w_k1_IBM_s106
                                 + 0.889984 w_k1_INTC_s106
                                 + 0.851227 w_k1_KO_s106
                                 + 0.710673 w_k1_MAT_s106
                                 + 0.928774 w_k1_MSFT_s106
                                 + 1.0322 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.55471 w_k1_ADM_s106
                                 + 1.62827 w_k1_AXP_s106
                                 + 1.21298 w_k1_DIS_s106
                                 + 1.29504 w_k1_FDX_s106
                                 + 0.916833 w_k1_IBM_s106
                                 + 1.00259 w_k1_INTC_s106
                                 + 1.17228 w_k1_KO_s106 + 1.35164 w_k1_MAT_s106
                                 + 1.22725 w_k1_MSFT_s106
                                 + 1.25263 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.778095 w_k1_ADM_s106
                                 + 0.681775 w_k1_AXP_s106
                                 + 0.581944 w_k1_DIS_s106
                                 + 0.773755 w_k1_FDX_s106
                                 + 0.654277 w_k1_IBM_s106
                                 + 0.770846 w_k1_INTC_s106
                                 + 0.654801 w_k1_KO_s106
                                 + 0.582302 w_k1_MAT_s106
                                 + 0.783282 w_k1_MSFT_s106
                                 + 0.747048 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.977286 w_k1_ADM_s107
                                 + 1.27685 w_k1_AXP_s107
                                 + 1.18413 w_k1_DIS_s107
                                 + 0.993549 w_k1_FDX_s107
                                 + 1.61938 w_k1_IBM_s107
                                 + 0.938836 w_k1_INTC_s107
                                 + 1.47272 w_k1_KO_s107 + 2.31133 w_k1_MAT_s107
                                 + 1.05 w_k1_MSFT_s107 + 1.0641 w_k1_PG_s107
                                 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.0311 w_k1_ADM_s107
                                 + 1.10583 w_k1_AXP_s107
                                 + 0.821484 w_k1_DIS_s107
                                 + 0.890853 w_k1_FDX_s107
                                 + 0.939086 w_k1_IBM_s107
                                 + 0.795336 w_k1_INTC_s107
                                 + 0.857198 w_k1_KO_s107
                                 + 1.34769 w_k1_MAT_s107
                                 + 0.945394 w_k1_MSFT_s107
                                 + 0.830992 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.936324 w_k1_ADM_s107
                                 + 0.767167 w_k1_AXP_s107
                                 + 0.585813 w_k1_DIS_s107
                                 + 0.785033 w_k1_FDX_s107
                                 + 0.673917 w_k1_IBM_s107
                                 + 0.770185 w_k1_INTC_s107
                                 + 0.702932 w_k1_KO_s107
                                 + 0.50933 w_k1_MAT_s107
                                 + 0.707364 w_k1_MSFT_s107
                                 + 0.75518 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.918328 w_k1_ADM_s107
                                 + 0.96941 w_k1_AXP_s107
                                 + 0.729289 w_k1_DIS_s107
                                 + 0.734465 w_k1_FDX_s107
                                 + 1.17208 w_k1_IBM_s107
                                 + 0.612881 w_k1_INTC_s107
                                 + 0.978172 w_k1_KO_s107
                                 + 1.67862 w_k1_MAT_s107
                                 + 0.833394 w_k1_MSFT_s107
                                 + 0.931404 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.00042 w_k1_ADM_s107
                                 + 0.590571 w_k1_AXP_s107
                                 + 0.60836 w_k1_DIS_s107
                                 + 0.716125 w_k1_FDX_s107
                                 + 0.905191 w_k1_IBM_s107
                                 + 0.58356 w_k1_INTC_s107
                                 + 0.884116 w_k1_KO_s107
                                 + 0.920872 w_k1_MAT_s107
                                 + 0.858107 w_k1_MSFT_s107
                                 + 0.83952 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.25656 w_k1_ADM_s107
                                 + 1.27286 w_k1_AXP_s107
                                 + 0.961845 w_k1_DIS_s107
                                 + 1.03531 w_k1_FDX_s107
                                 + 0.794335 w_k1_IBM_s107
                                 + 0.923104 w_k1_INTC_s107
                                 + 0.920502 w_k1_KO_s107
                                 + 0.709489 w_k1_MAT_s107
                                 + 0.961579 w_k1_MSFT_s107
                                 + 1.09342 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.16001 w_k1_ADM_s107
                                 + 0.979434 w_k1_AXP_s107
                                 + 1.01577 w_k1_DIS_s107
                                 + 0.909079 w_k1_FDX_s107
                                 + 1.11893 w_k1_IBM_s107
                                 + 0.708337 w_k1_INTC_s107
                                 + 1.1574 w_k1_KO_s107 + 1.10703 w_k1_MAT_s107
                                 + 0.80201 w_k1_MSFT_s107
                                 + 0.936122 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.10109 w_k1_ADM_s107
                                 + 1.42077 w_k1_AXP_s107
                                 + 1.30145 w_k1_DIS_s107
                                 + 1.13573 w_k1_FDX_s107
                                 + 1.04895 w_k1_IBM_s107
                                 + 0.853847 w_k1_INTC_s107
                                 + 1.00948 w_k1_KO_s107 + 1.19933 w_k1_MAT_s107
                                 + 0.89772 w_k1_MSFT_s107
                                 + 0.982468 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.67437 w_k1_ADM_s110
                                 + 0.693813 w_k1_AXP_s110
                                 + 0.573317 w_k1_DIS_s110
                                 + 0.702474 w_k1_FDX_s110
                                 + 0.891299 w_k1_IBM_s110
                                 + 0.897004 w_k1_INTC_s110
                                 + 0.830029 w_k1_KO_s110
                                 + 0.971219 w_k1_MAT_s110
                                 + 0.752346 w_k1_MSFT_s110
                                 + 0.723101 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 0.929854 w_k1_ADM_s110
                                 + 1.03906 w_k1_AXP_s110
                                 + 0.893663 w_k1_DIS_s110
                                 + 0.915368 w_k1_FDX_s110
                                 + 1.07338 w_k1_IBM_s110
                                 + 0.913348 w_k1_INTC_s110
                                 + 0.993184 w_k1_KO_s110
                                 + 1.41804 w_k1_MAT_s110
                                 + 0.909121 w_k1_MSFT_s110
                                 + 0.99887 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.945997 w_k1_ADM_s110
                                 + 1.16346 w_k1_AXP_s110
                                 + 1.37482 w_k1_DIS_s110
                                 + 1.20699 w_k1_FDX_s110 + 1.3589 w_k1_IBM_s110
                                 + 1.28549 w_k1_INTC_s110
                                 + 1.29482 w_k1_KO_s110 + 1.47228 w_k1_MAT_s110
                                 + 1.28001 w_k1_MSFT_s110
                                 + 1.10479 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.958 w_k1_ADM_s110
                                 + 0.957741 w_k1_AXP_s110
                                 + 1.20708 w_k1_DIS_s110
                                 + 0.920207 w_k1_FDX_s110
                                 + 1.18605 w_k1_IBM_s110
                                 + 1.0633 w_k1_INTC_s110 + 1.06337 w_k1_KO_s110
                                 + 0.788394 w_k1_MAT_s110
                                 + 0.997837 w_k1_MSFT_s110
                                 + 0.91421 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.809215 w_k1_ADM_s110
                                 + 0.843496 w_k1_AXP_s110
                                 + 1.02566 w_k1_DIS_s110
                                 + 0.912433 w_k1_FDX_s110
                                 + 1.49827 w_k1_IBM_s110
                                 + 0.918522 w_k1_INTC_s110
                                 + 1.33518 w_k1_KO_s110 + 1.32139 w_k1_MAT_s110
                                 + 1.00455 w_k1_MSFT_s110
                                 + 0.92071 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.928909 w_k1_ADM_s110
                                 + 0.672784 w_k1_AXP_s110
                                 + 0.773662 w_k1_DIS_s110
                                 + 0.860158 w_k1_FDX_s110
                                 + 0.742145 w_k1_IBM_s110
                                 + 0.888412 w_k1_INTC_s110
                                 + 0.796239 w_k1_KO_s110
                                 + 0.698865 w_k1_MAT_s110
                                 + 0.984339 w_k1_MSFT_s110
                                 + 0.853442 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.24093 w_k1_ADM_s110
                                 + 0.849787 w_k1_AXP_s110
                                 + 1.13506 w_k1_DIS_s110
                                 + 1.18036 w_k1_FDX_s110
                                 + 0.859159 w_k1_IBM_s110
                                 + 1.13912 w_k1_INTC_s110
                                 + 0.994277 w_k1_KO_s110
                                 + 0.662084 w_k1_MAT_s110
                                 + 1.16923 w_k1_MSFT_s110
                                 + 0.991494 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.14668 w_k1_ADM_s110
                                 + 1.25398 w_k1_AXP_s110
                                 + 1.48619 w_k1_DIS_s110
                                 + 1.27321 w_k1_FDX_s110
                                 + 1.06334 w_k1_IBM_s110
                                 + 0.946034 w_k1_INTC_s110
                                 + 1.12556 w_k1_KO_s110 + 1.34882 w_k1_MAT_s110
                                 + 1.19133 w_k1_MSFT_s110
                                 + 1.07086 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.977115 w_k1_ADM_s111
                                 + 0.754501 w_k1_AXP_s111
                                 + 0.833892 w_k1_DIS_s111
                                 + 0.819472 w_k1_FDX_s111
                                 + 1.03698 w_k1_IBM_s111
                                 + 0.840646 w_k1_INTC_s111
                                 + 0.953498 w_k1_KO_s111
                                 + 0.971084 w_k1_MAT_s111
                                 + 0.921023 w_k1_MSFT_s111
                                 + 0.919616 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.29265 w_k1_ADM_s111
                                 + 0.839348 w_k1_AXP_s111
                                 + 1.28795 w_k1_DIS_s111
                                 + 1.05438 w_k1_FDX_s111
                                 + 1.10651 w_k1_IBM_s111
                                 + 1.22685 w_k1_INTC_s111
                                 + 1.25282 w_k1_KO_s111
                                 + 0.864282 w_k1_MAT_s111
                                 + 1.26373 w_k1_MSFT_s111
                                 + 1.15796 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 1.22668 w_k1_ADM_s111
                                 + 1.07765 w_k1_AXP_s111
                                 + 1.21266 w_k1_DIS_s111
                                 + 1.10936 w_k1_FDX_s111
                                 + 0.951141 w_k1_IBM_s111
                                 + 1.28275 w_k1_INTC_s111
                                 + 1.17575 w_k1_KO_s111 + 0.88419 w_k1_MAT_s111
                                 + 1.12092 w_k1_MSFT_s111
                                 + 1.09374 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.964352 w_k1_ADM_s111
                                 + 1.09935 w_k1_AXP_s111
                                 + 0.904307 w_k1_DIS_s111
                                 + 0.887487 w_k1_FDX_s111
                                 + 1.18764 w_k1_IBM_s111
                                 + 0.92008 w_k1_INTC_s111
                                 + 1.03382 w_k1_KO_s111 + 1.21615 w_k1_MAT_s111
                                 + 0.902218 w_k1_MSFT_s111
                                 + 0.943949 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.76922 w_k1_ADM_s111
                                 + 1.3663 w_k1_AXP_s111 + 1.38269 w_k1_DIS_s111
                                 + 1.13079 w_k1_FDX_s111
                                 + 1.46381 w_k1_IBM_s111
                                 + 1.12823 w_k1_INTC_s111
                                 + 1.47919 w_k1_KO_s111 + 1.42167 w_k1_MAT_s111
                                 + 1.14095 w_k1_MSFT_s111
                                 + 1.22164 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.986299 w_k1_ADM_s111
                                 + 0.872849 w_k1_AXP_s111
                                 + 0.756901 w_k1_DIS_s111
                                 + 0.820805 w_k1_FDX_s111
                                 + 0.725357 w_k1_IBM_s111
                                 + 0.911576 w_k1_INTC_s111
                                 + 0.774838 w_k1_KO_s111
                                 + 0.61726 w_k1_MAT_s111
                                 + 0.822903 w_k1_MSFT_s111
                                 + 0.852242 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.92938 w_k1_ADM_s111
                                 + 0.721359 w_k1_AXP_s111
                                 + 0.731903 w_k1_DIS_s111
                                 + 0.748434 w_k1_FDX_s111
                                 + 1.4267 w_k1_IBM_s111
                                 + 0.732021 w_k1_INTC_s111
                                 + 1.19819 w_k1_KO_s111 + 1.43366 w_k1_MAT_s111
                                 + 0.839309 w_k1_MSFT_s111
                                 + 0.955961 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.2595 w_k1_ADM_s111
                                 + 1.05342 w_k1_AXP_s111
                                 + 0.876226 w_k1_DIS_s111
                                 + 0.972592 w_k1_FDX_s111
                                 + 0.848566 w_k1_IBM_s111
                                 + 0.759741 w_k1_INTC_s111
                                 + 0.870566 w_k1_KO_s111
                                 + 0.754122 w_k1_MAT_s111
                                 + 0.803621 w_k1_MSFT_s111
                                 + 1.00282 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.06882 w_k1_ADM_s112
                                 + 1.11674 w_k1_AXP_s112
                                 + 1.12095 w_k1_DIS_s112
                                 + 1.10269 w_k1_FDX_s112
                                 + 0.993501 w_k1_IBM_s112
                                 + 0.907378 w_k1_INTC_s112
                                 + 1.01343 w_k1_KO_s112 + 1.10912 w_k1_MAT_s112
                                 + 0.899603 w_k1_MSFT_s112
                                 + 1.07043 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.19747 w_k1_ADM_s112
                                 + 1.05944 w_k1_AXP_s112
                                 + 1.74016 w_k1_DIS_s112
                                 + 1.23752 w_k1_FDX_s112
                                 + 1.21107 w_k1_IBM_s112
                                 + 1.4958 w_k1_INTC_s112 + 1.53927 w_k1_KO_s112
                                 + 1.00567 w_k1_MAT_s112
                                 + 1.53839 w_k1_MSFT_s112
                                 + 1.15413 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 0.673318 w_k1_ADM_s112
                                 + 0.738555 w_k1_AXP_s112
                                 + 0.714934 w_k1_DIS_s112
                                 + 0.804357 w_k1_FDX_s112
                                 + 0.86019 w_k1_IBM_s112
                                 + 0.923349 w_k1_INTC_s112
                                 + 0.889183 w_k1_KO_s112
                                 + 0.902831 w_k1_MAT_s112
                                 + 0.795837 w_k1_MSFT_s112
                                 + 0.916007 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.951341 w_k1_ADM_s112
                                 + 1.09377 w_k1_AXP_s112
                                 + 1.15523 w_k1_DIS_s112
                                 + 0.933141 w_k1_FDX_s112
                                 + 1.35175 w_k1_IBM_s112
                                 + 1.1228 w_k1_INTC_s112 + 1.25372 w_k1_KO_s112
                                 + 1.66092 w_k1_MAT_s112
                                 + 0.96168 w_k1_MSFT_s112
                                 + 1.01916 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.864592 w_k1_ADM_s112
                                 + 0.778231 w_k1_AXP_s112
                                 + 1.09312 w_k1_DIS_s112
                                 + 1.03328 w_k1_FDX_s112
                                 + 0.656772 w_k1_IBM_s112
                                 + 0.991858 w_k1_INTC_s112
                                 + 0.770758 w_k1_KO_s112
                                 + 0.702124 w_k1_MAT_s112
                                 + 0.910647 w_k1_MSFT_s112
                                 + 0.824223 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.17851 w_k1_ADM_s112
                                 + 1.13072 w_k1_AXP_s112
                                 + 1.25708 w_k1_DIS_s112
                                 + 1.21125 w_k1_FDX_s112
                                 + 0.849932 w_k1_IBM_s112
                                 + 1.0014 w_k1_INTC_s112
                                 + 0.946584 w_k1_KO_s112
                                 + 0.975757 w_k1_MAT_s112
                                 + 1.13541 w_k1_MSFT_s112 + 1.0863 w_k1_PG_s112
                                 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.812774 w_k1_ADM_s112
                                 + 0.557447 w_k1_AXP_s112
                                 + 0.836546 w_k1_DIS_s112
                                 + 0.727284 w_k1_FDX_s112
                                 + 1.13933 w_k1_IBM_s112
                                 + 0.910223 w_k1_INTC_s112
                                 + 0.95269 w_k1_KO_s112
                                 + 0.924026 w_k1_MAT_s112
                                 + 0.898352 w_k1_MSFT_s112
                                 + 0.865191 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.851357 w_k1_ADM_s112
                                 + 1.33205 w_k1_AXP_s112
                                 + 1.65789 w_k1_DIS_s112
                                 + 1.11394 w_k1_FDX_s112
                                 + 1.74415 w_k1_IBM_s112
                                 + 1.09974 w_k1_INTC_s112
                                 + 1.48215 w_k1_KO_s112 + 1.79605 w_k1_MAT_s112
                                 + 1.18837 w_k1_MSFT_s112
                                 + 1.37082 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.820753 w_k1_ADM_s113
                                 + 0.913127 w_k1_AXP_s113
                                 + 0.924767 w_k1_DIS_s113
                                 + 0.952187 w_k1_FDX_s113
                                 + 0.933321 w_k1_IBM_s113
                                 + 1.0376 w_k1_INTC_s113
                                 + 0.813993 w_k1_KO_s113
                                 + 0.906721 w_k1_MAT_s113
                                 + 0.852936 w_k1_MSFT_s113
                                 + 0.859132 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 1.09834 w_k1_ADM_s113
                                 + 1.38932 w_k1_AXP_s113
                                 + 1.29995 w_k1_DIS_s113
                                 + 1.20782 w_k1_FDX_s113
                                 + 1.45525 w_k1_IBM_s113
                                 + 1.23452 w_k1_INTC_s113
                                 + 1.09046 w_k1_KO_s113 + 1.67341 w_k1_MAT_s113
                                 + 0.994578 w_k1_MSFT_s113
                                 + 1.05182 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 0.9776 w_k1_ADM_s113
                                 + 0.982635 w_k1_AXP_s113
                                 + 0.912181 w_k1_DIS_s113
                                 + 1.04269 w_k1_FDX_s113
                                 + 0.756733 w_k1_IBM_s113
                                 + 1.1192 w_k1_INTC_s113
                                 + 0.789576 w_k1_KO_s113
                                 + 0.737744 w_k1_MAT_s113
                                 + 0.889307 w_k1_MSFT_s113
                                 + 0.903873 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.747665 w_k1_ADM_s113
                                 + 0.60603 w_k1_AXP_s113
                                 + 0.612424 w_k1_DIS_s113
                                 + 0.730453 w_k1_FDX_s113
                                 + 0.837188 w_k1_IBM_s113
                                 + 0.859194 w_k1_INTC_s113
                                 + 0.60875 w_k1_KO_s113
                                 + 0.558429 w_k1_MAT_s113
                                 + 0.692084 w_k1_MSFT_s113
                                 + 0.706651 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.05306 w_k1_ADM_s113
                                 + 1.10864 w_k1_AXP_s113
                                 + 1.25566 w_k1_DIS_s113
                                 + 1.19525 w_k1_FDX_s113
                                 + 0.954574 w_k1_IBM_s113
                                 + 1.01369 w_k1_INTC_s113
                                 + 1.01729 w_k1_KO_s113 + 1.19316 w_k1_MAT_s113
                                 + 0.987331 w_k1_MSFT_s113
                                 + 1.07256 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.866712 w_k1_ADM_s113
                                 + 0.915976 w_k1_AXP_s113
                                 + 0.693812 w_k1_DIS_s113
                                 + 0.913139 w_k1_FDX_s113
                                 + 0.787406 w_k1_IBM_s113
                                 + 0.865335 w_k1_INTC_s113
                                 + 0.710694 w_k1_KO_s113
                                 + 0.776853 w_k1_MAT_s113
                                 + 0.735341 w_k1_MSFT_s113
                                 + 0.857025 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.04965 w_k1_ADM_s113
                                 + 0.947837 w_k1_AXP_s113
                                 + 0.789619 w_k1_DIS_s113
                                 + 0.853603 w_k1_FDX_s113
                                 + 1.13825 w_k1_IBM_s113
                                 + 0.99911 w_k1_INTC_s113
                                 + 0.926149 w_k1_KO_s113
                                 + 1.01104 w_k1_MAT_s113
                                 + 0.775138 w_k1_MSFT_s113
                                 + 0.938925 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.03554 w_k1_ADM_s113
                                 + 0.874327 w_k1_AXP_s113
                                 + 1.43829 w_k1_DIS_s113 + 1.0493 w_k1_FDX_s113
                                 + 1.27791 w_k1_IBM_s113
                                 + 1.21549 w_k1_INTC_s113
                                 + 1.21054 w_k1_KO_s113
                                 + 0.789358 w_k1_MAT_s113
                                 + 1.09759 w_k1_MSFT_s113
                                 + 0.924963 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.00814 w_k1_ADM_s114
                                 + 0.859453 w_k1_AXP_s114
                                 + 0.718074 w_k1_DIS_s114
                                 + 0.837141 w_k1_FDX_s114
                                 + 0.674991 w_k1_IBM_s114
                                 + 0.913681 w_k1_INTC_s114
                                 + 0.867475 w_k1_KO_s114
                                 + 0.63894 w_k1_MAT_s114
                                 + 0.856319 w_k1_MSFT_s114
                                 + 0.884345 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.900865 w_k1_ADM_s114
                                 + 1.30626 w_k1_AXP_s114
                                 + 1.48925 w_k1_DIS_s114
                                 + 1.12505 w_k1_FDX_s114
                                 + 1.80169 w_k1_IBM_s114
                                 + 0.990394 w_k1_INTC_s114
                                 + 1.51339 w_k1_KO_s114 + 2.03468 w_k1_MAT_s114
                                 + 1.29304 w_k1_MSFT_s114
                                 + 1.26625 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.00837 w_k1_ADM_s114
                                 + 1.1489 w_k1_AXP_s114 + 0.92967 w_k1_DIS_s114
                                 + 1.14204 w_k1_FDX_s114
                                 + 0.79245 w_k1_IBM_s114
                                 + 1.02171 w_k1_INTC_s114
                                 + 0.859673 w_k1_KO_s114
                                 + 1.02272 w_k1_MAT_s114
                                 + 1.10407 w_k1_MSFT_s114
                                 + 1.07695 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.99711 w_k1_ADM_s114
                                 + 1.02453 w_k1_AXP_s114
                                 + 1.00797 w_k1_DIS_s114
                                 + 0.85875 w_k1_FDX_s114
                                 + 1.37623 w_k1_IBM_s114
                                 + 0.809464 w_k1_INTC_s114
                                 + 1.19997 w_k1_KO_s114 + 1.21916 w_k1_MAT_s114
                                 + 0.943427 w_k1_MSFT_s114
                                 + 0.993256 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.27506 w_k1_ADM_s114
                                 + 1.46108 w_k1_AXP_s114
                                 + 1.18394 w_k1_DIS_s114
                                 + 1.05733 w_k1_FDX_s114
                                 + 1.23347 w_k1_IBM_s114
                                 + 1.22341 w_k1_INTC_s114
                                 + 1.28906 w_k1_KO_s114 + 1.454 w_k1_MAT_s114
                                 + 1.0805 w_k1_MSFT_s114 + 1.24464 w_k1_PG_s114
                                 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.17473 w_k1_ADM_s114
                                 + 0.847752 w_k1_AXP_s114
                                 + 0.856073 w_k1_DIS_s114
                                 + 0.892665 w_k1_FDX_s114
                                 + 0.86297 w_k1_IBM_s114
                                 + 0.834195 w_k1_INTC_s114
                                 + 0.918772 w_k1_KO_s114
                                 + 0.946441 w_k1_MAT_s114
                                 + 0.950516 w_k1_MSFT_s114
                                 + 1.06995 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.880637 w_k1_ADM_s114
                                 + 1.13409 w_k1_AXP_s114
                                 + 1.13039 w_k1_DIS_s114 + 1.0066 w_k1_FDX_s114
                                 + 1.01678 w_k1_IBM_s114
                                 + 1.01603 w_k1_INTC_s114
                                 + 0.947075 w_k1_KO_s114
                                 + 0.865689 w_k1_MAT_s114
                                 + 1.10918 w_k1_MSFT_s114
                                 + 0.979866 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.22281 w_k1_ADM_s114
                                 + 1.50752 w_k1_AXP_s114
                                 + 1.59662 w_k1_DIS_s114
                                 + 1.38054 w_k1_FDX_s114
                                 + 1.16433 w_k1_IBM_s114
                                 + 1.36183 w_k1_INTC_s114
                                 + 1.39829 w_k1_KO_s114 + 1.24819 w_k1_MAT_s114
                                 + 1.30601 w_k1_MSFT_s114 + 1.0956 w_k1_PG_s114
                                 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.806876 w_k1_ADM_s115
                                 + 0.675188 w_k1_AXP_s115
                                 + 0.594768 w_k1_DIS_s115
                                 + 0.728412 w_k1_FDX_s115
                                 + 0.968389 w_k1_IBM_s115
                                 + 0.863679 w_k1_INTC_s115
                                 + 0.772512 w_k1_KO_s115
                                 + 0.731844 w_k1_MAT_s115
                                 + 0.764208 w_k1_MSFT_s115
                                 + 0.704326 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.946758 w_k1_ADM_s115
                                 + 0.998809 w_k1_AXP_s115
                                 + 1.49147 w_k1_DIS_s115
                                 + 1.14962 w_k1_FDX_s115
                                 + 1.09264 w_k1_IBM_s115
                                 + 1.28406 w_k1_INTC_s115
                                 + 1.37448 w_k1_KO_s115
                                 + 0.885622 w_k1_MAT_s115
                                 + 1.30207 w_k1_MSFT_s115
                                 + 1.14616 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.24356 w_k1_ADM_s115
                                 + 1.04351 w_k1_AXP_s115
                                 + 1.21369 w_k1_DIS_s115
                                 + 1.21826 w_k1_FDX_s115
                                 + 0.782752 w_k1_IBM_s115
                                 + 1.1639 w_k1_INTC_s115
                                 + 0.876343 w_k1_KO_s115
                                 + 0.609738 w_k1_MAT_s115
                                 + 1.06374 w_k1_MSFT_s115
                                 + 0.929835 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.806744 w_k1_ADM_s115
                                 + 0.749816 w_k1_AXP_s115
                                 + 0.882201 w_k1_DIS_s115
                                 + 0.988892 w_k1_FDX_s115
                                 + 0.606529 w_k1_IBM_s115
                                 + 1.0555 w_k1_INTC_s115
                                 + 0.736389 w_k1_KO_s115
                                 + 0.400644 w_k1_MAT_s115
                                 + 0.982465 w_k1_MSFT_s115
                                 + 0.841476 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.920162 w_k1_ADM_s115
                                 + 0.919945 w_k1_AXP_s115
                                 + 0.836552 w_k1_DIS_s115
                                 + 0.85716 w_k1_FDX_s115 + 1.2702 w_k1_IBM_s115
                                 + 1.22706 w_k1_INTC_s115
                                 + 1.07457 w_k1_KO_s115
                                 + 0.839339 w_k1_MAT_s115
                                 + 0.997527 w_k1_MSFT_s115
                                 + 0.898221 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.32814 w_k1_ADM_s115
                                 + 1.33941 w_k1_AXP_s115
                                 + 1.33617 w_k1_DIS_s115
                                 + 1.22209 w_k1_FDX_s115
                                 + 1.21019 w_k1_IBM_s115
                                 + 1.04965 w_k1_INTC_s115
                                 + 0.911945 w_k1_KO_s115
                                 + 0.999327 w_k1_MAT_s115
                                 + 1.05453 w_k1_MSFT_s115
                                 + 0.979195 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.21751 w_k1_ADM_s115
                                 + 0.937679 w_k1_AXP_s115
                                 + 0.773705 w_k1_DIS_s115
                                 + 1.02163 w_k1_FDX_s115
                                 + 0.79929 w_k1_IBM_s115
                                 + 0.893274 w_k1_INTC_s115
                                 + 0.804698 w_k1_KO_s115
                                 + 0.682298 w_k1_MAT_s115
                                 + 0.87306 w_k1_MSFT_s115
                                 + 0.914221 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.5214 w_k1_ADM_s115
                                 + 1.10534 w_k1_AXP_s115
                                 + 1.47496 w_k1_DIS_s115
                                 + 1.06418 w_k1_FDX_s115
                                 + 1.73072 w_k1_IBM_s115
                                 + 0.993217 w_k1_INTC_s115
                                 + 1.40132 w_k1_KO_s115 + 1.16689 w_k1_MAT_s115
                                 + 1.13715 w_k1_MSFT_s115
                                 + 1.12254 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.95354 w_k1_ADM_s116
                                 + 0.910231 w_k1_AXP_s116
                                 + 1.02121 w_k1_DIS_s116
                                 + 1.04513 w_k1_FDX_s116
                                 + 0.905473 w_k1_IBM_s116
                                 + 1.01584 w_k1_INTC_s116
                                 + 0.84894 w_k1_KO_s116 + 1.05417 w_k1_MAT_s116
                                 + 1.0249 w_k1_MSFT_s116
                                 + 0.989159 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.16588 w_k1_ADM_s116
                                 + 1.39494 w_k1_AXP_s116
                                 + 1.41523 w_k1_DIS_s116
                                 + 1.36598 w_k1_FDX_s116
                                 + 0.868108 w_k1_IBM_s116
                                 + 0.995531 w_k1_INTC_s116
                                 + 1.05025 w_k1_KO_s116 + 1.42162 w_k1_MAT_s116
                                 + 1.23237 w_k1_MSFT_s116
                                 + 1.03882 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.01174 w_k1_ADM_s116
                                 + 1.10808 w_k1_AXP_s116
                                 + 0.841083 w_k1_DIS_s116
                                 + 0.913423 w_k1_FDX_s116
                                 + 0.91027 w_k1_IBM_s116
                                 + 0.807846 w_k1_INTC_s116
                                 + 0.886293 w_k1_KO_s116
                                 + 1.40082 w_k1_MAT_s116
                                 + 0.913183 w_k1_MSFT_s116
                                 + 0.879552 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.06048 w_k1_ADM_s116
                                 + 1.40663 w_k1_AXP_s116 + 1.3707 w_k1_DIS_s116
                                 + 1.23439 w_k1_FDX_s116
                                 + 1.60811 w_k1_IBM_s116
                                 + 1.33634 w_k1_INTC_s116
                                 + 1.53899 w_k1_KO_s116 + 2.00232 w_k1_MAT_s116
                                 + 1.3654 w_k1_MSFT_s116 + 1.03439 w_k1_PG_s116
                                 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.0068 w_k1_ADM_s116
                                 + 1.193 w_k1_AXP_s116 + 0.914847 w_k1_DIS_s116
                                 + 1.04047 w_k1_FDX_s116
                                 + 1.22107 w_k1_IBM_s116
                                 + 1.18069 w_k1_INTC_s116
                                 + 0.956076 w_k1_KO_s116
                                 + 1.61627 w_k1_MAT_s116
                                 + 1.0821 w_k1_MSFT_s116
                                 + 0.862713 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 1.43765 w_k1_ADM_s116
                                 + 1.43551 w_k1_AXP_s116
                                 + 1.45279 w_k1_DIS_s116
                                 + 1.29927 w_k1_FDX_s116
                                 + 1.15563 w_k1_IBM_s116
                                 + 0.997671 w_k1_INTC_s116
                                 + 1.35625 w_k1_KO_s116 + 1.58817 w_k1_MAT_s116
                                 + 1.29866 w_k1_MSFT_s116
                                 + 1.14753 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.820469 w_k1_ADM_s116
                                 + 0.997441 w_k1_AXP_s116
                                 + 0.903482 w_k1_DIS_s116
                                 + 0.840566 w_k1_FDX_s116
                                 + 1.27288 w_k1_IBM_s116
                                 + 0.925742 w_k1_INTC_s116
                                 + 0.971742 w_k1_KO_s116
                                 + 1.37137 w_k1_MAT_s116
                                 + 1.03089 w_k1_MSFT_s116
                                 + 0.869359 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.793353 w_k1_ADM_s116
                                 + 0.731534 w_k1_AXP_s116
                                 + 0.701624 w_k1_DIS_s116
                                 + 0.816811 w_k1_FDX_s116
                                 + 0.71046 w_k1_IBM_s116
                                 + 0.856942 w_k1_INTC_s116
                                 + 0.623541 w_k1_KO_s116
                                 + 0.751591 w_k1_MAT_s116
                                 + 0.772113 w_k1_MSFT_s116
                                 + 0.720645 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 1.08407 w_k1_ADM_s117
                                 + 1.41324 w_k1_AXP_s117
                                 + 1.74981 w_k1_DIS_s117
                                 + 1.50913 w_k1_FDX_s117
                                 + 1.61696 w_k1_IBM_s117
                                 + 1.45791 w_k1_INTC_s117
                                 + 1.70716 w_k1_KO_s117 + 1.1678 w_k1_MAT_s117
                                 + 1.58392 w_k1_MSFT_s117
                                 + 1.24906 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.05602 w_k1_ADM_s117
                                 + 1.19183 w_k1_AXP_s117
                                 + 1.28894 w_k1_DIS_s117
                                 + 1.30643 w_k1_FDX_s117
                                 + 1.13207 w_k1_IBM_s117
                                 + 1.27951 w_k1_INTC_s117
                                 + 1.10249 w_k1_KO_s117 + 1.01856 w_k1_MAT_s117
                                 + 1.30123 w_k1_MSFT_s117
                                 + 1.19342 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 0.825119 w_k1_ADM_s117
                                 + 0.83089 w_k1_AXP_s117
                                 + 0.77499 w_k1_DIS_s117
                                 + 0.926041 w_k1_FDX_s117
                                 + 0.807831 w_k1_IBM_s117
                                 + 0.904447 w_k1_INTC_s117
                                 + 0.855743 w_k1_KO_s117
                                 + 0.771875 w_k1_MAT_s117
                                 + 0.968519 w_k1_MSFT_s117
                                 + 0.808521 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.842352 w_k1_ADM_s117
                                 + 1.04558 w_k1_AXP_s117
                                 + 0.81948 w_k1_DIS_s117
                                 + 0.957462 w_k1_FDX_s117
                                 + 1.49661 w_k1_IBM_s117
                                 + 0.849784 w_k1_INTC_s117
                                 + 1.09365 w_k1_KO_s117 + 1.23346 w_k1_MAT_s117
                                 + 0.937541 w_k1_MSFT_s117
                                 + 0.956671 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.62942 w_k1_ADM_s117
                                 + 0.601813 w_k1_AXP_s117
                                 + 0.773967 w_k1_DIS_s117
                                 + 0.778859 w_k1_FDX_s117 + 1.034 w_k1_IBM_s117
                                 + 0.957731 w_k1_INTC_s117
                                 + 1.04405 w_k1_KO_s117 + 0.85686 w_k1_MAT_s117
                                 + 0.930849 w_k1_MSFT_s117
                                 + 0.93463 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.06006 w_k1_ADM_s117
                                 + 1.2426 w_k1_AXP_s117 + 0.73655 w_k1_DIS_s117
                                 + 0.931412 w_k1_FDX_s117
                                 + 0.953093 w_k1_IBM_s117
                                 + 0.693984 w_k1_INTC_s117
                                 + 0.864744 w_k1_KO_s117
                                 + 1.05739 w_k1_MAT_s117
                                 + 0.801168 w_k1_MSFT_s117
                                 + 0.87704 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.21481 w_k1_ADM_s117
                                 + 1.16686 w_k1_AXP_s117
                                 + 1.13762 w_k1_DIS_s117
                                 + 1.00731 w_k1_FDX_s117
                                 + 1.35757 w_k1_IBM_s117
                                 + 1.16575 w_k1_INTC_s117
                                 + 1.24266 w_k1_KO_s117 + 1.03245 w_k1_MAT_s117
                                 + 1.03649 w_k1_MSFT_s117
                                 + 0.914657 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 1.15707 w_k1_ADM_s117
                                 + 1.06085 w_k1_AXP_s117
                                 + 1.05234 w_k1_DIS_s117
                                 + 1.20231 w_k1_FDX_s117
                                 + 0.653312 w_k1_IBM_s117
                                 + 1.01436 w_k1_INTC_s117
                                 + 0.831748 w_k1_KO_s117
                                 + 0.515532 w_k1_MAT_s117
                                 + 1.08314 w_k1_MSFT_s117
                                 + 0.817774 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0447147580483195 w_k1_ADM_s000
                                 + 0.0789828410596981 w_k1_ADM_s001
                                 + 0.0419700800187739 w_k1_ADM_s002
                                 + 0.0913504672943214 w_k1_ADM_s003
                                 + 0.109270644094656 w_k1_ADM_s004
                                 + 0.0677119659696762 w_k1_ADM_s005
                                 + 0.035656788691433 w_k1_ADM_s006
                                 + 0.0388308344285163 w_k1_ADM_s007
                                 + 0.0168136494666934 w_k1_ADM_s010
                                 + 0.0360021159510608 w_k1_ADM_s011
                                 + 0.0424109690025753 w_k1_ADM_s012
                                 + 0.0607624573459613 w_k1_ADM_s013
                                 + 0.0463419090262379 w_k1_ADM_s014
                                 + 0.0279875408202874 w_k1_ADM_s015
                                 + 0.0495131843074795 w_k1_ADM_s016
                                 + 0.00907050558394646 w_k1_ADM_s017
                                 + 0.005613179672717 w_k1_ADM_s100
                                 + 0.0183955730290621 w_k1_ADM_s101
                                 + 0.0129619487008255 w_k1_ADM_s102
                                 + 0.00537028738895961 w_k1_ADM_s103
                                 + 0.0170229729107895 w_k1_ADM_s104
                                 + 0.00675175048647664 w_k1_ADM_s105
                                 + 0.0183544288792816 w_k1_ADM_s106
                                 + 0.00485938729173668 w_k1_ADM_s107
                                 + 0.00837177641771514 w_k1_ADM_s110
                                 + 0.0100858505634277 w_k1_ADM_s111
                                 + 0.0154674987780835 w_k1_ADM_s112
                                 + 0.0173318193169113 w_k1_ADM_s113
                                 + 0.0188747290574443 w_k1_ADM_s114
                                 + 0.021118811921351 w_k1_ADM_s115
                                 + 0.0260318094060351 w_k1_ADM_s116
                                 + 0.00651634138612064 w_k1_ADM_s117
                                 + 0.0374610072665458 w_k1_AXP_s000
                                 + 0.0839882361423777 w_k1_AXP_s001
                                 + 0.0492365897419266 w_k1_AXP_s002
                                 + 0.0900509965912168 w_k1_AXP_s003
                                 + 0.110411855595892 w_k1_AXP_s004
                                 + 0.0720221091797818 w_k1_AXP_s005
                                 + 0.0351625089383048 w_k1_AXP_s006
                                 + 0.0371348302105532 w_k1_AXP_s007
                                 + 0.0176322763397982 w_k1_AXP_s010
                                 + 0.0392383453100267 w_k1_AXP_s011
                                 + 0.0405712304311808 w_k1_AXP_s012
                                 + 0.0582560595969683 w_k1_AXP_s013
                                 + 0.0556054231536578 w_k1_AXP_s014
                                 + 0.0292127944055821 w_k1_AXP_s015
                                 + 0.0475516891049649 w_k1_AXP_s016
                                 + 0.0113231305017537 w_k1_AXP_s017
                                 + 0.00488444589205702 w_k1_AXP_s100
                                 + 0.0133528374672335 w_k1_AXP_s101
                                 + 0.0135639491439957 w_k1_AXP_s102
                                 + 0.00519386118918988 w_k1_AXP_s103
                                 + 0.0162595289245451 w_k1_AXP_s104
                                 + 0.00753275369876165 w_k1_AXP_s105
                                 + 0.0181307388637241 w_k1_AXP_s106
                                 + 0.00507023149779886 w_k1_AXP_s107
                                 + 0.00813910019513998 w_k1_AXP_s110
                                 + 0.00836133395639743 w_k1_AXP_s111
                                 + 0.0159850634640078 w_k1_AXP_s112
                                 + 0.0184320911699071 w_k1_AXP_s113
                                 + 0.0208579994253292 w_k1_AXP_s114
                                 + 0.0186239674332444 w_k1_AXP_s115
                                 + 0.0284566232027995 w_k1_AXP_s116
                                 + 0.00684035772414991 w_k1_AXP_s117
                                 + 0.0389778614082352 w_k1_DIS_s000
                                 + 0.0842332252147717 w_k1_DIS_s001
                                 + 0.0479086892550086 w_k1_DIS_s002
                                 + 0.09442090133146 w_k1_DIS_s003
                                 + 0.116209837236949 w_k1_DIS_s004
                                 + 0.0667384307610763 w_k1_DIS_s005
                                 + 0.036800634769789 w_k1_DIS_s006
                                 + 0.0426128609420638 w_k1_DIS_s007
                                 + 0.0158009763741374 w_k1_DIS_s010
                                 + 0.0398672962060947 w_k1_DIS_s011
                                 + 0.0442282088705323 w_k1_DIS_s012
                                 + 0.0546785963358417 w_k1_DIS_s013
                                 + 0.0537199977209422 w_k1_DIS_s014
                                 + 0.0333447220423127 w_k1_DIS_s015
                                 + 0.0502338340885116 w_k1_DIS_s016
                                 + 0.0117144824299284 w_k1_DIS_s017
                                 + 0.00519965580755317 w_k1_DIS_s100
                                 + 0.0135204392969119 w_k1_DIS_s101
                                 + 0.0132598267904725 w_k1_DIS_s102
                                 + 0.0048073021548327 w_k1_DIS_s103
                                 + 0.0157956003513306 w_k1_DIS_s104
                                 + 0.00769136633708657 w_k1_DIS_s105
                                 + 0.0184985538255133 w_k1_DIS_s106
                                 + 0.00440530314786754 w_k1_DIS_s107
                                 + 0.00889687831983614 w_k1_DIS_s110
                                 + 0.0086799264068975 w_k1_DIS_s111
                                 + 0.0186000477783994 w_k1_DIS_s112
                                 + 0.0186775881735504 w_k1_DIS_s113
                                 + 0.0191320995748711 w_k1_DIS_s114
                                 + 0.0203680977953815 w_k1_DIS_s115
                                 + 0.0272252619907472 w_k1_DIS_s116
                                 + 0.00663532493312598 w_k1_DIS_s117
                                 + 0.0391699253685331 w_k1_FDX_s000
                                 + 0.0838134787197473 w_k1_FDX_s001
                                 + 0.0468819179507343 w_k1_FDX_s002
                                 + 0.0897249580256802 w_k1_FDX_s003
                                 + 0.110587577957414 w_k1_FDX_s004
                                 + 0.0679384939589003 w_k1_FDX_s005
                                 + 0.0362761454743523 w_k1_FDX_s006
                                 + 0.0379624228932351 w_k1_FDX_s007
                                 + 0.017278712792087 w_k1_FDX_s010
                                 + 0.0384662448466592 w_k1_FDX_s011
                                 + 0.0414051629319152 w_k1_FDX_s012
                                 + 0.0530515754604542 w_k1_FDX_s013
                                 + 0.0541028870490337 w_k1_FDX_s014
                                 + 0.030927128839553 w_k1_FDX_s015
                                 + 0.0478357996564433 w_k1_FDX_s016
                                 + 0.0110778118808332 w_k1_FDX_s017
                                 + 0.00517637433055826 w_k1_FDX_s100
                                 + 0.012911882478013 w_k1_FDX_s101
                                 + 0.0126384449113782 w_k1_FDX_s102
                                 + 0.00513087313603111 w_k1_FDX_s103
                                 + 0.0163135614296094 w_k1_FDX_s104
                                 + 0.00721033020539186 w_k1_FDX_s105
                                 + 0.0175061324841972 w_k1_FDX_s106
                                 + 0.00429612724070193 w_k1_FDX_s107
                                 + 0.00855928323541305 w_k1_FDX_s110
                                 + 0.00817723680894355 w_k1_FDX_s111
                                 + 0.0166916346825634 w_k1_FDX_s112
                                 + 0.018335683206718 w_k1_FDX_s113
                                 + 0.0186165060306705 w_k1_FDX_s114
                                 + 0.0195365891910835 w_k1_FDX_s115
                                 + 0.027066940392554 w_k1_FDX_s116
                                 + 0.00698884265267936 w_k1_FDX_s117
                                 + 0.0390679013481584 w_k1_IBM_s000
                                 + 0.0846626084071221 w_k1_IBM_s001
                                 + 0.0471564260051907 w_k1_IBM_s002
                                 + 0.090765754474154 w_k1_IBM_s003
                                 + 0.112726378481886 w_k1_IBM_s004
                                 + 0.0694702012661117 w_k1_IBM_s005
                                 + 0.0347714261380989 w_k1_IBM_s006
                                 + 0.0379805008863997 w_k1_IBM_s007
                                 + 0.016816108375086 w_k1_IBM_s010
                                 + 0.0381362401629587 w_k1_IBM_s011
                                 + 0.0438847345774934 w_k1_IBM_s012
                                 + 0.0530942321912874 w_k1_IBM_s013
                                 + 0.0572826488838906 w_k1_IBM_s014
                                 + 0.0315763914090473 w_k1_IBM_s015
                                 + 0.0494333049794355 w_k1_IBM_s016
                                 + 0.0110353373974534 w_k1_IBM_s017
                                 + 0.00515318988633971 w_k1_IBM_s100
                                 + 0.0131936932146437 w_k1_IBM_s101
                                 + 0.0134575508164126 w_k1_IBM_s102
                                 + 0.00537848375085037 w_k1_IBM_s103
                                 + 0.0174346685897886 w_k1_IBM_s104
                                 + 0.00720690477635267 w_k1_IBM_s105
                                 + 0.0180369702306808 w_k1_IBM_s106
                                 + 0.00481012011522966 w_k1_IBM_s107
                                 + 0.00916909064108331 w_k1_IBM_s110
                                 + 0.00890979722617235 w_k1_IBM_s111
                                 + 0.0167262826709708 w_k1_IBM_s112
                                 + 0.0189206222567719 w_k1_IBM_s113
                                 + 0.0185041041783064 w_k1_IBM_s114
                                 + 0.0204538173737463 w_k1_IBM_s115
                                 + 0.0276130648310028 w_k1_IBM_s116
                                 + 0.00703746391526892 w_k1_IBM_s117
                                 + 0.0437185843434893 w_k1_INTC_s000
                                 + 0.088892344766362 w_k1_INTC_s001
                                 + 0.0501883004041495 w_k1_INTC_s002
                                 + 0.0933386271310864 w_k1_INTC_s003
                                 + 0.114456013845083 w_k1_INTC_s004
                                 + 0.0805783003126306 w_k1_INTC_s005
                                 + 0.0350418905751002 w_k1_INTC_s006
                                 + 0.0424174413839431 w_k1_INTC_s007
                                 + 0.017819653108271 w_k1_INTC_s010
                                 + 0.0412875181387403 w_k1_INTC_s011
                                 + 0.0442397938294601 w_k1_INTC_s012
                                 + 0.0502692371888145 w_k1_INTC_s013
                                 + 0.0547875929150165 w_k1_INTC_s014
                                 + 0.0272078544581003 w_k1_INTC_s015
                                 + 0.0436673798040854 w_k1_INTC_s016
                                 + 0.0133442671376318 w_k1_INTC_s017
                                 + 0.00446011760243682 w_k1_INTC_s100
                                 + 0.0127760776571591 w_k1_INTC_s101
                                 + 0.0128046221134493 w_k1_INTC_s102
                                 + 0.00626809519855561 w_k1_INTC_s103
                                 + 0.0172816404565036 w_k1_INTC_s104
                                 + 0.00836986626838195 w_k1_INTC_s105
                                 + 0.0168480494763098 w_k1_INTC_s106
                                 + 0.00367313715918436 w_k1_INTC_s107
                                 + 0.00854342273124406 w_k1_INTC_s110
                                 + 0.00863732861994251 w_k1_INTC_s111
                                 + 0.0168099828897669 w_k1_INTC_s112
                                 + 0.0186457904628232 w_k1_INTC_s113
                                 + 0.0184671296697587 w_k1_INTC_s114
                                 + 0.020084375437184 w_k1_INTC_s115
                                 + 0.0260124798263345 w_k1_INTC_s116
                                 + 0.00682668079750156 w_k1_INTC_s117
                                 + 0.0399094571347251 w_k1_KO_s000
                                 + 0.092219098280522 w_k1_KO_s001
                                 + 0.047779577347863 w_k1_KO_s002
                                 + 0.0866805802935744 w_k1_KO_s003
                                 + 0.104186199343329 w_k1_KO_s004
                                 + 0.0703052054786252 w_k1_KO_s005
                                 + 0.0360123196692288 w_k1_KO_s006
                                 + 0.0402219300892036 w_k1_KO_s007
                                 + 0.0158224976541568 w_k1_KO_s010
                                 + 0.040089523939321 w_k1_KO_s011
                                 + 0.0426943988258063 w_k1_KO_s012
                                 + 0.0573854093103211 w_k1_KO_s013
                                 + 0.0534700126147471 w_k1_KO_s014
                                 + 0.0296518380136346 w_k1_KO_s015
                                 + 0.0499289281324115 w_k1_KO_s016
                                 + 0.0112942736878204 w_k1_KO_s017
                                 + 0.00501231131659145 w_k1_KO_s100
                                 + 0.0128398566619694 w_k1_KO_s101
                                 + 0.0125062988390853 w_k1_KO_s102
                                 + 0.00531303699876534 w_k1_KO_s103
                                 + 0.0154551771692111 w_k1_KO_s104
                                 + 0.00770893891576586 w_k1_KO_s105
                                 + 0.019058670329667 w_k1_KO_s106
                                 + 0.00467802144961009 w_k1_KO_s107
                                 + 0.00894134958128142 w_k1_KO_s110
                                 + 0.00916892959305104 w_k1_KO_s111
                                 + 0.0171369640714681 w_k1_KO_s112
                                 + 0.0165425854300907 w_k1_KO_s113
                                 + 0.0189678234703527 w_k1_KO_s114
                                 + 0.0186669403920096 w_k1_KO_s115
                                 + 0.0259174187670946 w_k1_KO_s116
                                 + 0.00688159622872271 w_k1_KO_s117
                                 + 0.0369798871394879 w_k1_MAT_s000
                                 + 0.117504515375158 w_k1_MAT_s001
                                 + 0.0528597355801153 w_k1_MAT_s002
                                 + 0.093391991281607 w_k1_MAT_s003
                                 + 0.0870227759289416 w_k1_MAT_s004
                                 + 0.0716772552325593 w_k1_MAT_s005
                                 + 0.0328482214910858 w_k1_MAT_s006
                                 + 0.0395931944715799 w_k1_MAT_s007
                                 + 0.0192567323216493 w_k1_MAT_s010
                                 + 0.0333007807046519 w_k1_MAT_s011
                                 + 0.0391014020178394 w_k1_MAT_s012
                                 + 0.0561461296844518 w_k1_MAT_s013
                                 + 0.0486143622507965 w_k1_MAT_s014
                                 + 0.0363293510282453 w_k1_MAT_s015
                                 + 0.0492032755824841 w_k1_MAT_s016
                                 + 0.0104742728662791 w_k1_MAT_s017
                                 + 0.00524856821240332 w_k1_MAT_s100
                                 + 0.0111639893478874 w_k1_MAT_s101
                                 + 0.0146559313760421 w_k1_MAT_s102
                                 + 0.00575382598304491 w_k1_MAT_s103
                                 + 0.0113990985086065 w_k1_MAT_s104
                                 + 0.0073478107668352 w_k1_MAT_s105
                                 + 0.0171734828261339 w_k1_MAT_s106
                                 + 0.00564341181490967 w_k1_MAT_s107
                                 + 0.00960710632997023 w_k1_MAT_s110
                                 + 0.00819071930390009 w_k1_MAT_s111
                                 + 0.0179490331958111 w_k1_MAT_s112
                                 + 0.0193139463449455 w_k1_MAT_s113
                                 + 0.0198624837252507 w_k1_MAT_s114
                                 + 0.0152562577603733 w_k1_MAT_s115
                                 + 0.0347373375835848 w_k1_MAT_s116
                                 + 0.00601885457125986 w_k1_MAT_s117
                                 + 0.0394577869162628 w_k1_MSFT_s000
                                 + 0.0831274641718452 w_k1_MSFT_s001
                                 + 0.0505308021299352 w_k1_MSFT_s002
                                 + 0.0918575422080535 w_k1_MSFT_s003
                                 + 0.107502687997276 w_k1_MSFT_s004
                                 + 0.0677226438177172 w_k1_MSFT_s005
                                 + 0.0405637992345383 w_k1_MSFT_s006
                                 + 0.0376465672539309 w_k1_MSFT_s007
                                 + 0.017782613147289 w_k1_MSFT_s010
                                 + 0.0405091300052422 w_k1_MSFT_s011
                                 + 0.0444829488384897 w_k1_MSFT_s012
                                 + 0.0556278587039863 w_k1_MSFT_s013
                                 + 0.0538246326327226 w_k1_MSFT_s014
                                 + 0.0303086626424663 w_k1_MSFT_s015
                                 + 0.0432423095907015 w_k1_MSFT_s016
                                 + 0.010866082217023 w_k1_MSFT_s017
                                 + 0.00494453336792969 w_k1_MSFT_s100
                                 + 0.0117832935085747 w_k1_MSFT_s101
                                 + 0.0128337117422083 w_k1_MSFT_s102
                                 + 0.00573927025638939 w_k1_MSFT_s103
                                 + 0.0149759818910745 w_k1_MSFT_s104
                                 + 0.00763447687520899 w_k1_MSFT_s105
                                 + 0.018160810466377 w_k1_MSFT_s106
                                 + 0.00408171081458193 w_k1_MSFT_s107
                                 + 0.00886942663707843 w_k1_MSFT_s110
                                 + 0.0084869431061734 w_k1_MSFT_s111
                                 + 0.016418717344078 w_k1_MSFT_s112
                                 + 0.0158603896805105 w_k1_MSFT_s113
                                 + 0.0191837116310471 w_k1_MSFT_s114
                                 + 0.0192110307552224 w_k1_MSFT_s115
                                 + 0.0277334800842209 w_k1_MSFT_s116
                                 + 0.00701748665956204 w_k1_MSFT_s117
                                 + 0.0377272255752647 w_k1_PG_s000
                                 + 0.0769720680346779 w_k1_PG_s001
                                 + 0.0476701923966152 w_k1_PG_s002
                                 + 0.0910493924079262 w_k1_PG_s003
                                 + 0.102463618437656 w_k1_PG_s004
                                 + 0.069302882069557 w_k1_PG_s005
                                 + 0.0341450432125391 w_k1_PG_s006
                                 + 0.0361736898607086 w_k1_PG_s007
                                 + 0.0166517388502173 w_k1_PG_s010
                                 + 0.0326932764406296 w_k1_PG_s011
                                 + 0.0414028713232653 w_k1_PG_s012
                                 + 0.0494258059598005 w_k1_PG_s013
                                 + 0.0546082685759183 w_k1_PG_s014
                                 + 0.0306658607580002 w_k1_PG_s015
                                 + 0.0455819508986453 w_k1_PG_s016
                                 + 0.0114659681643652 w_k1_PG_s017
                                 + 0.00502061517563101 w_k1_PG_s100
                                 + 0.0128228920313941 w_k1_PG_s101
                                 + 0.0127307746283848 w_k1_PG_s102
                                 + 0.00524086618618603 w_k1_PG_s103
                                 + 0.0153587112263941 w_k1_PG_s104
                                 + 0.007346840773027 w_k1_PG_s105
                                 + 0.0180921596802254 w_k1_PG_s106
                                 + 0.0043407245151881 w_k1_PG_s107
                                 + 0.00829405205749532 w_k1_PG_s110
                                 + 0.00870207924552932 w_k1_PG_s111
                                 + 0.0166769362311543 w_k1_PG_s112
                                 + 0.0167591935538978 w_k1_PG_s113
                                 + 0.0191638340618814 w_k1_PG_s114
                                 + 0.017709156452282 w_k1_PG_s115
                                 + 0.0243860681984343 w_k1_PG_s116
                                 + 0.00620528375730719 w_k1_PG_s117 >= 1.05
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
