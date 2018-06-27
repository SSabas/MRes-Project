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
 w_k1_ADM_s0_return_constraint:  1.1626 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.927965 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.0517 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.975493 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.921515 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.13196 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.06144 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.13942 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.720761 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.02089 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.61561 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 1.3853 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.942919 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 1.71109 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 1.35307 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.05348 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.4679 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.18162 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 0.803306 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.972619 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 0.854491 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.954831 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.887604 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.715224 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.05746 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.938411 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.41019 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.04376 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 1.38757 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.0911 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.32582 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.05158 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 0.744934 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 0.994933 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 0.880707 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.35013 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 0.987267 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.21658 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.25817 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.925336 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.09165 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 1.03328 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.812979 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.16167 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.86712 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 0.84266 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.543602 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.32913 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.57148 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 1.33322 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.755511 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 1.62649 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 1.33466 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 0.884394 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.87989 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.26696 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.624975 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.9527 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 0.694758 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.896384 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.712127 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.593673 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.17611 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.892652 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.44693 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.16268 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 1.66527 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.01668 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 1.48336 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.943012 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.823963 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.23667 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 0.772854 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.42013 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 0.95248 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 1.4971 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.36867 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.876891 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.20372 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.929494 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.866234 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.19914 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.820362 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.10183 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.513434 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.13525 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.71606 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.94791 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.732299 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 1.78024 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 1.12121 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.18293 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 2.10844 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.41817 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.722829 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.935897 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 0.699665 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.715725 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.685732 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.494923 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.12876 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.781112 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.39894 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.12939 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 1.95364 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 0.991248 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.82228 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 1.11375 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.652674 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.03976 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.701148 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.40539 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.01489 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 1.61143 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.19731 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.918957 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.20955 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.955871 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.95871 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.10574 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 0.902497 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 0.991303 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.654008 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.07853 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.26724 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 1.12875 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.79757 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 1.45558 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 1.29515 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.07175 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.56296 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.22478 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 0.859793 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.852015 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 0.770306 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.884424 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.733674 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.614783 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 0.957659 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.892917 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.39971 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.04316 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.53772 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.08721 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.33324 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.18222 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.644023 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.10702 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 0.789253 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.13716 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 0.910964 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 1.42094 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.15282 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.974314 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 0.83638 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.10641 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.787122 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.33479 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.743781 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.07761 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.593928 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.22867 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 2.01188 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.848756 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.84958 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 1.30092 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.837125 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.903482 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.73058 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.42412 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.552585 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.50482 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.875449 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.64081 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.01841 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.628738 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.56977 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.690497 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.877156 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.10514 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 1.251 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.695646 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 1.4142 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.636507 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.13461 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.925519 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.767802 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.94442 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.24866 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.976087 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.17923 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.918789 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.07125 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.862611 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.95974 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.01942 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 0.866063 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.04 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.693147 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.28667 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.23792 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 1.06104 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.78527 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.41607 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 1.34928 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.18069 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.5023 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.29817 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 0.878597 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.982987 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 0.957579 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.01117 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.776184 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.743373 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 1.05254 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.830479 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.25688 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.13099 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 1.46805 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.15517 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.22999 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 1.29638 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 0.814436 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.27032 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 0.903212 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.1637 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.15254 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 1.42335 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.1655 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.895078 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.11007 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  1.06374 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.909083 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.23527 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.777841 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 0.958811 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.57435 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.06335 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.51095 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.862341 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.724866 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 1.31604 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 1.03109 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 0.916422 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.61107 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.31205 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.72749 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.12427 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 0.844942 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.672489 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.823957 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.605557 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.20996 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.692683 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.04206 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.07461 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 1.47622 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 0.825079 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 1.38982 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.871214 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.952577 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.03526 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.767292 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.5532 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.15734 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.235 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.19021 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.816085 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 0.872653 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.07063 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.880662 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.61831 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.937518 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.07957 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.556136 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.3715 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 2.43339 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 1.0646 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.937098 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 1.23714 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 1.02325 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.78183 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.91434 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.29559 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.678865 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.6537 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 0.974448 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.560087 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.817702 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.546359 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.69247 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.760447 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.00776 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 1.16079 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.45128 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.510858 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 1.54182 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.539852 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.20761 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.06515 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.679847 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.97893 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.3558 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.01121 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.14239 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.914135 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.21537 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.94232 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.974345 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.06974 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 0.859753 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.01672 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.680346 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.04574 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.22364 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 1.05983 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.776199 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 1.42196 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 1.25774 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.1356 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.28033 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.30464 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.912057 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 1.01585 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 0.894756 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.885068 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.755936 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.646164 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.986705 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.793761 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.28988 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.06288 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.43175 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.08919 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.46827 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.23483 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.812125 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.08688 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 0.833465 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.0939 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 0.985969 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 1.28015 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.1222 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.948676 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.07515 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  1.05505 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.898059 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.1527 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.846224 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 0.937619 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.763139 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.13004 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.23825 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 1.03412 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.811251 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 1.2861 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 1.01399 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 0.936717 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.24764 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.13693 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.852882 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 1.07707 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 0.859455 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.807631 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.907758 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.777002 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.14089 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.8747 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.12102 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.05347 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 1.28648 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 0.988524 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 1.27071 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.961401 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.796727 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 0.997254 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.823573 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.19956 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.03694 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 1.07115 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.213000299612874 z_k1_s0000
                                 - 0.098999813634979 z_k1_s0001
                                 - 0.478998188769529 z_k1_s0002
                                 - 0.432999209470099 z_k1_s0003
                                 - 0.154999646761992 z_k1_s0004
                                 - 0.267999520078467 z_k1_s0005
                                 - 0.230998753950342 z_k1_s0006
                                 - 0.149999625447172 z_k1_s0007
                                 - 0.0779995649419187 z_k1_s0010
                                 - 0.112999412442552 z_k1_s0011
                                 - 0.175999436682104 z_k1_s0012
                                 - 0.256998909764496 z_k1_s0013
                                 - 0.156999673760939 z_k1_s0014
                                 - 0.1689986169265 z_k1_s0015
                                 - 0.0559996640815029 z_k1_s0016
                                 - 0.14299918324362 z_k1_s0017
                                 - 0.0499999629796934 z_k1_s0020
                                 - 0.124999907449233 z_k1_s0021
                                 - 0.0799999220875265 z_k1_s0022
                                 - 0.0929998825742742 z_k1_s0023
                                 - 0.216999757139945 z_k1_s0024
                                 - 0.127999912700008 z_k1_s0025
                                 - 0.0669999616007789 z_k1_s0026
                                 - 0.174000304544936 z_k1_s0027
                                 - 0.135999903549041 z_k1_s0030
                                 - 0.222999544226842 z_k1_s0031
                                 - 0.266999016487003 z_k1_s0032
                                 - 0.146999975014008 z_k1_s0033
                                 - 0.255999148421963 z_k1_s0034
                                 - 0.198999980012156 z_k1_s0035
                                 - 0.400000194908772 z_k1_s0036
                                 - 0.406000492762298 z_k1_s0037
                                 - 0.110999778886204 z_k1_s0040
                                 - 0.0719998615261783 z_k1_s0041
                                 - 0.12799979453536 z_k1_s0042
                                 - 0.14300018551928 z_k1_s0043
                                 - 0.0959993879022839 z_k1_s0044
                                 - 0.0969996582341379 z_k1_s0045
                                 - 0.0969996582341379 z_k1_s0046
                                 - 0.171999781156795 z_k1_s0047
                                 - 0.19999962542254 z_k1_s0050
                                 - 0.061999941180903 z_k1_s0051
                                 - 0.124999765889088 z_k1_s0052
                                 - 0.129000254996807 z_k1_s0053
                                 - 0.0849998408045795 z_k1_s0054
                                 - 0.133999672633214 z_k1_s0055
                                 - 0.104000301818989 z_k1_s0056
                                 - 0.115999859144961 z_k1_s0057
                                 - 0.312999940645841 z_k1_s0060
                                 - 0.188999873035747 z_k1_s0061
                                 - 0.307998663029616 z_k1_s0062
                                 - 0.188999873035747 z_k1_s0063
                                 - 0.242999742003187 z_k1_s0064
                                 - 0.511998610217332 z_k1_s0065
                                 - 0.344999715597045 z_k1_s0066
                                 - 0.111999875628634 z_k1_s0067
                                 - 0.0949999073274549 z_k1_s0070
                                 - 0.165999731077556 z_k1_s0071
                                 - 0.174999701919126 z_k1_s0072
                                 - 0.194000312583976 z_k1_s0073
                                 - 0.279000306087411 z_k1_s0074
                                 - 0.136000231605249 z_k1_s0075
                                 - 0.0969999142922344 z_k1_s0076
                                 - 0.0679998738028707 z_k1_s0077
                                 - 0.105000141357142 z_k1_s0100
                                 - 0.220999667741429 z_k1_s0101
                                 - 0.0870000534386993 z_k1_s0102
                                 - 0.352999718075971 z_k1_s0103
                                 - 0.125000090444124 z_k1_s0104
                                 - 0.219999588556983 z_k1_s0105
                                 - 0.12600016962857 z_k1_s0106
                                 - 0.249000398904155 z_k1_s0107
                                 - 0.28500007802333 z_k1_s0110
                                 - 0.228000178098717 z_k1_s0111
                                 - 0.39299984274363 z_k1_s0112
                                 - 0.238000140246885 z_k1_s0113
                                 - 0.569001258792344 z_k1_s0114
                                 - 0.415999929204496 z_k1_s0115
                                 - 0.385000567105411 z_k1_s0116
                                 - 0.377999031920982 z_k1_s0117
                                 - 0.0800000444620203 z_k1_s0120
                                 - 0.152000373198067 z_k1_s0121
                                 - 0.18300050471219 z_k1_s0122
                                 - 0.200999804945583 z_k1_s0123
                                 - 0.0930000336420844 z_k1_s0124
                                 - 0.216000023807379 z_k1_s0125
                                 - 0.17299995780401 z_k1_s0126
                                 - 0.105000088431425 z_k1_s0127
                                 - 0.586002297301709 z_k1_s0130
                                 - 0.416000011124939 z_k1_s0131
                                 - 0.540000762912414 z_k1_s0132
                                 - 0.662000247073893 z_k1_s0133
                                 - 0.605999907556968 z_k1_s0134
                                 - 0.311000002280902 z_k1_s0135
                                 - 0.365000078572143 z_k1_s0136
                                 - 1.53599926846107 z_k1_s0137
                                 - 0.286998168712923 z_k1_s0140
                                 - 0.563999077866188 z_k1_s0141
                                 - 0.141999495422513 z_k1_s0142
                                 - 0.171999332075341 z_k1_s0143
                                 - 0.682996549728506 z_k1_s0144
                                 - 0.357999662318514 z_k1_s0145
                                 - 0.34100002348104 z_k1_s0146
                                 - 0.13899951175723 z_k1_s0147
                                 - 0.21400058326049 z_k1_s0150
                                 - 0.259999495288076 z_k1_s0151
                                 - 0.33800101907501 z_k1_s0152
                                 - 0.230000842139845 z_k1_s0153
                                 - 0.175000030767087 z_k1_s0154
                                 - 0.230000842139845 z_k1_s0155
                                 - 0.311000974841219 z_k1_s0156
                                 - 0.335999154464532 z_k1_s0157
                                 - 0.354000480902494 z_k1_s0160
                                 - 0.989003244858203 z_k1_s0161
                                 - 0.68900051485458 z_k1_s0162
                                 - 0.316000135102035 z_k1_s0163
                                 - 0.419000250252744 z_k1_s0164
                                 - 1.01499964475602 z_k1_s0165
                                 - 1.1300020615083 z_k1_s0166
                                 - 0.568999422853131 z_k1_s0167
                                 - 0.129000147012255 z_k1_s0170
                                 - 0.132000016682188 z_k1_s0171
                                 - 0.332000639486035 z_k1_s0172
                                 - 0.520001372809782 z_k1_s0173
                                 - 0.250000778504809 z_k1_s0174
                                 - 0.105000162651873 z_k1_s0175
                                 - 0.188000117123199 z_k1_s0176
                                 - 0.397999826226398 z_k1_s0177
                                 - 0.311000179053615 z_k1_s1000
                                 - 0.900996640301573 z_k1_s1001
                                 - 0.709999946550598 z_k1_s1002
                                 - 0.524999924506446 z_k1_s1003
                                 - 1.34699636212277 z_k1_s1004
                                 - 0.834002915623632 z_k1_s1005
                                 - 1.50799729244562 z_k1_s1006
                                 - 1.16099592186128 z_k1_s1007
                                 - 0.108999975053002 z_k1_s1010
                                 - 0.148999882688085 z_k1_s1011
                                 - 0.481000652388608 z_k1_s1012
                                 - 0.121999912641918 z_k1_s1013
                                 - 0.330999379132747 z_k1_s1014
                                 - 0.123999917278668 z_k1_s1015
                                 - 0.119999908005168 z_k1_s1016
                                 - 0.414999573876253 z_k1_s1017
                                 - 0.173999305360447 z_k1_s1020
                                 - 0.864998290722055 z_k1_s1021
                                 - 0.68799657572663 z_k1_s1022
                                 - 0.437998072045941 z_k1_s1023
                                 - 0.53399628366392 z_k1_s1024
                                 - 0.619998023121544 z_k1_s1025
                                 - 0.813997208776446 z_k1_s1026
                                 - 0.201999397871699 z_k1_s1027
                                 - 0.596001132471199 z_k1_s1030
                                 - 0.520997752084151 z_k1_s1031
                                 - 0.847000855757146 z_k1_s1032
                                 - 0.748000053564992 z_k1_s1033
                                 - 0.510999519498453 z_k1_s1034
                                 - 0.315999880289003 z_k1_s1035
                                 - 0.209000076128433 z_k1_s1036
                                 - 0.410999725847446 z_k1_s1037
                                 - 0.427000481427668 z_k1_s1040
                                 - 0.225000061981854 z_k1_s1041
                                 - 0.417999644348664 z_k1_s1042
                                 - 0.417999644348664 z_k1_s1043
                                 - 0.267999924027324 z_k1_s1044
                                 - 0.250999769544299 z_k1_s1045
                                 - 0.34100139933652 z_k1_s1046
                                 - 0.862000280259357 z_k1_s1047
                                 - 1.46200074335624 z_k1_s1050
                                 - 1.09100041516952 z_k1_s1051
                                 - 0.504999777642335 z_k1_s1052
                                 - 1.35399562971314 z_k1_s1053
                                 - 1.94399853465497 z_k1_s1054
                                 - 0.744998956537317 z_k1_s1055
                                 - 0.801998963423455 z_k1_s1056
                                 - 0.597999788877612 z_k1_s1057
                                 - 0.543000455078258 z_k1_s1060
                                 - 0.176000078397369 z_k1_s1061
                                 - 0.549001168432756 z_k1_s1062
                                 - 0.180000137033676 z_k1_s1063
                                 - 0.598001261327482 z_k1_s1064
                                 - 0.26500013225513 z_k1_s1065
                                 - 0.645000074103981 z_k1_s1066
                                 - 0.171000005101985 z_k1_s1067
                                 - 0.595999896110884 z_k1_s1070
                                 - 0.548999444088835 z_k1_s1071
                                 - 0.388999308864478 z_k1_s1072
                                 - 0.551999251324341 z_k1_s1073
                                 - 0.216999944698095 z_k1_s1074
                                 - 0.57400130971717 z_k1_s1075
                                 - 0.452998668554397 z_k1_s1076
                                 - 0.141999903011672 z_k1_s1077
                                 - 0.261999597641687 z_k1_s1100
                                 - 0.349000693283133 z_k1_s1101
                                 - 0.233000004761762 z_k1_s1102
                                 - 0.474000616284021 z_k1_s1103
                                 - 0.516999533243566 z_k1_s1104
                                 - 0.246000701122361 z_k1_s1105
                                 - 0.127999976760949 z_k1_s1106
                                 - 0.108000081760874 z_k1_s1107
                                 - 0.16399987550463 z_k1_s1110
                                 - 0.198999897045227 z_k1_s1111
                                 - 0.107999998839592 z_k1_s1112
                                 - 0.0869999859152336 z_k1_s1113
                                 - 0.20899967777123 z_k1_s1114
                                 - 0.298999282304432 z_k1_s1115
                                 - 0.210000129243581 z_k1_s1116
                                 - 0.301999058722316 z_k1_s1117
                                 - 0.415999448020151 z_k1_s1120
                                 - 0.189000138733286 z_k1_s1121
                                 - 0.393000227600944 z_k1_s1122
                                 - 0.156000152557641 z_k1_s1123
                                 - 0.426999283611995 z_k1_s1124
                                 - 0.139999934087579 z_k1_s1125
                                 - 0.318000476929077 z_k1_s1126
                                 - 0.35799922535381 z_k1_s1127
                                 - 0.999000419334822 z_k1_s1130
                                 - 0.833999484939896 z_k1_s1131
                                 - 0.282999969795 z_k1_s1132
                                 - 0.37700003328347 z_k1_s1133
                                 - 1.33900008142059 z_k1_s1134
                                 - 0.638999835655464 z_k1_s1135
                                 - 0.326000083970605 z_k1_s1136
                                 - 0.537997582358622 z_k1_s1137
                                 - 1.30799756385264 z_k1_s1140
                                 - 0.698999713590153 z_k1_s1141
                                 - 0.588999904409897 z_k1_s1142
                                 - 0.443999285491171 z_k1_s1143
                                 - 1.02199637478533 z_k1_s1144
                                 - 0.979000829502239 z_k1_s1145
                                 - 1.30900102304239 z_k1_s1146
                                 - 1.30999682223833 z_k1_s1147
                                 - 0.140000212915891 z_k1_s1150
                                 - 0.153000103512236 z_k1_s1151
                                 - 0.320999984850925 z_k1_s1152
                                 - 0.139000113400678 z_k1_s1153
                                 - 0.112000076792618 z_k1_s1154
                                 - 0.0990000302961152 z_k1_s1155
                                 - 0.286999719336849 z_k1_s1156
                                 - 0.30800102965553 z_k1_s1157
                                 - 0.564000910933157 z_k1_s1160
                                 - 1.05100018512917 z_k1_s1161
                                 - 0.626001817957345 z_k1_s1162
                                 - 0.708002075530057 z_k1_s1163
                                 - 1.12700161097858 z_k1_s1164
                                 - 0.667999724028307 z_k1_s1165
                                 - 0.466000321160529 z_k1_s1166
                                 - 0.873998145244695 z_k1_s1167
                                 - 0.183000089739859 z_k1_s1170
                                 - 0.297000344941704 z_k1_s1171
                                 - 0.287001049660716 z_k1_s1172
                                 - 0.76800058497052 z_k1_s1173
                                 - 0.662000871193276 z_k1_s1174
                                 - 0.204999920857797 z_k1_s1175
                                 - 0.196999932033101 z_k1_s1176
                                 - 0.164000047206218 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 0.826323 w_k1_ADM_s000
                                 + 1.30927 w_k1_AXP_s000
                                 + 1.16771 w_k1_DIS_s000
                                 + 1.17292 w_k1_FDX_s000
                                 + 1.09913 w_k1_IBM_s000
                                 + 1.32778 w_k1_INTC_s000
                                 + 1.13056 w_k1_KO_s000 + 1.23319 w_k1_MAT_s000
                                 + 1.20179 w_k1_MSFT_s000 + 1.2458 w_k1_PG_s000
                                 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 1.12678 w_k1_ADM_s000
                                 + 1.40997 w_k1_AXP_s000 + 1.4817 w_k1_DIS_s000
                                 + 1.40993 w_k1_FDX_s000
                                 + 0.845539 w_k1_IBM_s000
                                 + 1.46453 w_k1_INTC_s000
                                 + 1.00391 w_k1_KO_s000
                                 + 0.605142 w_k1_MAT_s000
                                 + 1.36796 w_k1_MSFT_s000
                                 + 1.05941 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.814221 w_k1_ADM_s000
                                 + 0.865056 w_k1_AXP_s000
                                 + 0.928326 w_k1_DIS_s000
                                 + 0.856507 w_k1_FDX_s000
                                 + 1.03701 w_k1_IBM_s000
                                 + 1.03583 w_k1_INTC_s000
                                 + 0.903331 w_k1_KO_s000
                                 + 0.581901 w_k1_MAT_s000
                                 + 0.888761 w_k1_MSFT_s000
                                 + 1.05947 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.04311 w_k1_ADM_s000
                                 + 0.994514 w_k1_AXP_s000
                                 + 0.804373 w_k1_DIS_s000
                                 + 0.963141 w_k1_FDX_s000
                                 + 0.834604 w_k1_IBM_s000
                                 + 0.918244 w_k1_INTC_s000
                                 + 0.792691 w_k1_KO_s000
                                 + 0.672543 w_k1_MAT_s000
                                 + 0.835501 w_k1_MSFT_s000
                                 + 1.02834 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.883136 w_k1_ADM_s000
                                 + 0.727927 w_k1_AXP_s000
                                 + 0.839614 w_k1_DIS_s000
                                 + 0.986886 w_k1_FDX_s000
                                 + 0.575303 w_k1_IBM_s000
                                 + 0.902786 w_k1_INTC_s000
                                 + 0.813766 w_k1_KO_s000
                                 + 0.454969 w_k1_MAT_s000
                                 + 0.990181 w_k1_MSFT_s000
                                 + 1.01189 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.07386 w_k1_ADM_s000
                                 + 0.98382 w_k1_AXP_s000
                                 + 1.26891 w_k1_DIS_s000
                                 + 1.12131 w_k1_FDX_s000
                                 + 1.00239 w_k1_IBM_s000
                                 + 1.26721 w_k1_INTC_s000
                                 + 1.25068 w_k1_KO_s000
                                 + 0.963139 w_k1_MAT_s000
                                 + 1.17685 w_k1_MSFT_s000
                                 + 1.19818 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 0.902782 w_k1_ADM_s000
                                 + 0.918051 w_k1_AXP_s000
                                 + 1.00022 w_k1_DIS_s000
                                 + 0.882861 w_k1_FDX_s000
                                 + 1.4513 w_k1_IBM_s000
                                 + 1.06455 w_k1_INTC_s000
                                 + 1.16875 w_k1_KO_s000
                                 + 0.956203 w_k1_MAT_s000
                                 + 1.06667 w_k1_MSFT_s000
                                 + 1.14825 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.13219 w_k1_ADM_s000
                                 + 1.38153 w_k1_AXP_s000
                                 + 1.61525 w_k1_DIS_s000
                                 + 1.34907 w_k1_FDX_s000
                                 + 1.71908 w_k1_IBM_s000
                                 + 1.21426 w_k1_INTC_s000
                                 + 1.42736 w_k1_KO_s000 + 2.01393 w_k1_MAT_s000
                                 + 1.45303 w_k1_MSFT_s000 + 1.3027 w_k1_PG_s000
                                 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 1.16296 w_k1_ADM_s001
                                 + 1.54418 w_k1_AXP_s001
                                 + 1.67782 w_k1_DIS_s001
                                 + 1.42453 w_k1_FDX_s001
                                 + 0.863476 w_k1_IBM_s001
                                 + 1.47125 w_k1_INTC_s001
                                 + 1.31916 w_k1_KO_s001
                                 + 0.767349 w_k1_MAT_s001
                                 + 1.55415 w_k1_MSFT_s001
                                 + 1.23099 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 1.02317 w_k1_ADM_s001
                                 + 1.24069 w_k1_AXP_s001
                                 + 1.60108 w_k1_DIS_s001 + 1.1764 w_k1_FDX_s001
                                 + 1.29666 w_k1_IBM_s001
                                 + 1.38182 w_k1_INTC_s001
                                 + 1.45303 w_k1_KO_s001 + 1.14089 w_k1_MAT_s001
                                 + 1.26238 w_k1_MSFT_s001 + 1.2562 w_k1_PG_s001
                                 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.733852 w_k1_ADM_s001
                                 + 0.961758 w_k1_AXP_s001
                                 + 0.800568 w_k1_DIS_s001
                                 + 0.777628 w_k1_FDX_s001
                                 + 0.844167 w_k1_IBM_s001
                                 + 0.736122 w_k1_INTC_s001
                                 + 0.814622 w_k1_KO_s001
                                 + 0.729388 w_k1_MAT_s001
                                 + 0.813936 w_k1_MSFT_s001
                                 + 0.920667 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.910827 w_k1_ADM_s001
                                 + 0.884235 w_k1_AXP_s001
                                 + 0.971827 w_k1_DIS_s001
                                 + 1.0616 w_k1_FDX_s001
                                 + 0.710763 w_k1_IBM_s001
                                 + 1.05903 w_k1_INTC_s001
                                 + 0.901419 w_k1_KO_s001
                                 + 0.543304 w_k1_MAT_s001
                                 + 1.07967 w_k1_MSFT_s001
                                 + 1.03623 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.04554 w_k1_ADM_s001
                                 + 1.2402 w_k1_AXP_s001 + 1.16436 w_k1_DIS_s001
                                 + 1.10689 w_k1_FDX_s001
                                 + 0.971801 w_k1_IBM_s001
                                 + 1.22924 w_k1_INTC_s001
                                 + 1.15614 w_k1_KO_s001 + 1.17823 w_k1_MAT_s001
                                 + 1.04713 w_k1_MSFT_s001
                                 + 1.11064 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.799318 w_k1_ADM_s001
                                 + 0.90915 w_k1_AXP_s001
                                 + 0.862022 w_k1_DIS_s001
                                 + 0.9001 w_k1_FDX_s001 + 1.01984 w_k1_IBM_s001
                                 + 1.14127 w_k1_INTC_s001
                                 + 1.02031 w_k1_KO_s001
                                 + 0.978752 w_k1_MAT_s001
                                 + 1.03114 w_k1_MSFT_s001
                                 + 1.02972 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.601766 w_k1_ADM_s001
                                 + 0.759904 w_k1_AXP_s001
                                 + 0.544997 w_k1_DIS_s001
                                 + 0.733442 w_k1_FDX_s001
                                 + 0.583947 w_k1_IBM_s001
                                 + 0.836017 w_k1_INTC_s001
                                 + 0.667109 w_k1_KO_s001
                                 + 0.614088 w_k1_MAT_s001
                                 + 0.788725 w_k1_MSFT_s001
                                 + 0.872977 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.952153 w_k1_ADM_s001
                                 + 1.09132 w_k1_AXP_s001
                                 + 1.18199 w_k1_DIS_s001
                                 + 0.950732 w_k1_FDX_s001
                                 + 1.5329 w_k1_IBM_s001 + 1.1203 w_k1_INTC_s001
                                 + 1.20203 w_k1_KO_s001
                                 + 0.975007 w_k1_MAT_s001
                                 + 0.832752 w_k1_MSFT_s001
                                 + 1.0338 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 1.38417 w_k1_ADM_s002
                                 + 1.40878 w_k1_AXP_s002
                                 + 1.11666 w_k1_DIS_s002
                                 + 1.32064 w_k1_FDX_s002
                                 + 0.604122 w_k1_IBM_s002
                                 + 1.07748 w_k1_INTC_s002
                                 + 0.783316 w_k1_KO_s002
                                 + 0.648078 w_k1_MAT_s002
                                 + 1.01996 w_k1_MSFT_s002
                                 + 1.06698 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.09455 w_k1_ADM_s002
                                 + 1.37526 w_k1_AXP_s002
                                 + 1.18934 w_k1_DIS_s002
                                 + 1.18099 w_k1_FDX_s002
                                 + 1.24015 w_k1_IBM_s002
                                 + 1.06167 w_k1_INTC_s002
                                 + 1.03163 w_k1_KO_s002 + 1.38889 w_k1_MAT_s002
                                 + 1.08774 w_k1_MSFT_s002
                                 + 1.06918 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.75507 w_k1_ADM_s002
                                 + 1.38163 w_k1_AXP_s002
                                 + 1.15413 w_k1_DIS_s002
                                 + 1.10385 w_k1_FDX_s002
                                 + 1.87141 w_k1_IBM_s002
                                 + 1.27254 w_k1_INTC_s002 + 1.2576 w_k1_KO_s002
                                 + 2.2897 w_k1_MAT_s002
                                 + 1.07787 w_k1_MSFT_s002
                                 + 1.09455 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.00628 w_k1_ADM_s002
                                 + 0.882623 w_k1_AXP_s002
                                 + 1.32287 w_k1_DIS_s002
                                 + 1.05207 w_k1_FDX_s002
                                 + 0.849273 w_k1_IBM_s002
                                 + 0.946696 w_k1_INTC_s002
                                 + 1.12667 w_k1_KO_s002
                                 + 0.551805 w_k1_MAT_s002
                                 + 1.08744 w_k1_MSFT_s002
                                 + 0.921631 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.855481 w_k1_ADM_s002
                                 + 0.798548 w_k1_AXP_s002
                                 + 0.728992 w_k1_DIS_s002
                                 + 0.825986 w_k1_FDX_s002
                                 + 0.650932 w_k1_IBM_s002
                                 + 0.873851 w_k1_INTC_s002
                                 + 0.640219 w_k1_KO_s002
                                 + 0.593398 w_k1_MAT_s002
                                 + 0.776968 w_k1_MSFT_s002
                                 + 0.759356 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.23449 w_k1_ADM_s002
                                 + 1.09815 w_k1_AXP_s002
                                 + 0.873913 w_k1_DIS_s002
                                 + 1.06418 w_k1_FDX_s002
                                 + 0.797413 w_k1_IBM_s002
                                 + 1.07274 w_k1_INTC_s002
                                 + 0.887906 w_k1_KO_s002
                                 + 0.796833 w_k1_MAT_s002
                                 + 0.924811 w_k1_MSFT_s002
                                 + 0.871109 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.30029 w_k1_ADM_s002
                                 + 1.69895 w_k1_AXP_s002
                                 + 1.63632 w_k1_DIS_s002
                                 + 1.41339 w_k1_FDX_s002
                                 + 1.26155 w_k1_IBM_s002
                                 + 1.2711 w_k1_INTC_s002 + 1.2678 w_k1_KO_s002
                                 + 1.62163 w_k1_MAT_s002
                                 + 1.16188 w_k1_MSFT_s002
                                 + 1.17528 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.0915 w_k1_ADM_s002
                                 + 0.932586 w_k1_AXP_s002
                                 + 0.726252 w_k1_DIS_s002
                                 + 0.831523 w_k1_FDX_s002
                                 + 1.22325 w_k1_IBM_s002
                                 + 0.976185 w_k1_INTC_s002
                                 + 0.970636 w_k1_KO_s002
                                 + 1.31856 w_k1_MAT_s002
                                 + 0.894515 w_k1_MSFT_s002
                                 + 0.999792 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.08984 w_k1_ADM_s003
                                 + 0.959699 w_k1_AXP_s003
                                 + 1.16604 w_k1_DIS_s003
                                 + 0.966074 w_k1_FDX_s003
                                 + 1.11123 w_k1_IBM_s003
                                 + 0.924452 w_k1_INTC_s003
                                 + 1.14404 w_k1_KO_s003
                                 + 0.786613 w_k1_MAT_s003
                                 + 1.01041 w_k1_MSFT_s003
                                 + 0.933342 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 0.919731 w_k1_ADM_s003
                                 + 0.803177 w_k1_AXP_s003
                                 + 0.979557 w_k1_DIS_s003
                                 + 0.931981 w_k1_FDX_s003
                                 + 1.03085 w_k1_IBM_s003
                                 + 0.835849 w_k1_INTC_s003
                                 + 0.908008 w_k1_KO_s003
                                 + 0.987656 w_k1_MAT_s003
                                 + 0.922895 w_k1_MSFT_s003
                                 + 0.91846 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.00804 w_k1_ADM_s003
                                 + 0.845173 w_k1_AXP_s003
                                 + 0.84963 w_k1_DIS_s003
                                 + 0.964787 w_k1_FDX_s003
                                 + 0.815032 w_k1_IBM_s003
                                 + 0.833111 w_k1_INTC_s003
                                 + 0.852072 w_k1_KO_s003
                                 + 0.753236 w_k1_MAT_s003
                                 + 1.05227 w_k1_MSFT_s003
                                 + 0.875338 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 1.59307 w_k1_ADM_s003
                                 + 1.72373 w_k1_AXP_s003
                                 + 1.71684 w_k1_DIS_s003
                                 + 1.45633 w_k1_FDX_s003 + 1.212 w_k1_IBM_s003
                                 + 0.868719 w_k1_INTC_s003
                                 + 1.39657 w_k1_KO_s003 + 1.82538 w_k1_MAT_s003
                                 + 1.26657 w_k1_MSFT_s003
                                 + 1.24687 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.10658 w_k1_ADM_s003
                                 + 1.06707 w_k1_AXP_s003
                                 + 1.04253 w_k1_DIS_s003
                                 + 0.946709 w_k1_FDX_s003
                                 + 1.47797 w_k1_IBM_s003
                                 + 1.03532 w_k1_INTC_s003
                                 + 1.16316 w_k1_KO_s003 + 1.74269 w_k1_MAT_s003
                                 + 1.01455 w_k1_MSFT_s003
                                 + 0.988146 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.29171 w_k1_ADM_s003
                                 + 1.07629 w_k1_AXP_s003
                                 + 1.50701 w_k1_DIS_s003 + 1.2259 w_k1_FDX_s003
                                 + 1.43491 w_k1_IBM_s003
                                 + 0.836163 w_k1_INTC_s003
                                 + 1.40163 w_k1_KO_s003 + 1.23239 w_k1_MAT_s003
                                 + 1.12782 w_k1_MSFT_s003
                                 + 1.21113 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 0.95431 w_k1_ADM_s003
                                 + 0.721839 w_k1_AXP_s003
                                 + 0.663068 w_k1_DIS_s003
                                 + 0.737534 w_k1_FDX_s003
                                 + 0.904874 w_k1_IBM_s003
                                 + 0.656605 w_k1_INTC_s003
                                 + 0.816583 w_k1_KO_s003
                                 + 1.07817 w_k1_MAT_s003
                                 + 0.807515 w_k1_MSFT_s003
                                 + 0.837414 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 1.23405 w_k1_ADM_s003
                                 + 1.1649 w_k1_AXP_s003 + 1.16974 w_k1_DIS_s003
                                 + 1.19039 w_k1_FDX_s003
                                 + 0.868025 w_k1_IBM_s003
                                 + 0.982916 w_k1_INTC_s003
                                 + 1.19691 w_k1_KO_s003 + 1.195 w_k1_MAT_s003
                                 + 1.16607 w_k1_MSFT_s003
                                 + 1.05665 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.769819 w_k1_ADM_s004
                                 + 1.05465 w_k1_AXP_s004
                                 + 0.799742 w_k1_DIS_s004
                                 + 0.892957 w_k1_FDX_s004
                                 + 0.963323 w_k1_IBM_s004
                                 + 0.932876 w_k1_INTC_s004
                                 + 0.848082 w_k1_KO_s004
                                 + 0.892521 w_k1_MAT_s004
                                 + 0.808394 w_k1_MSFT_s004
                                 + 0.91542 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.671497 w_k1_ADM_s004
                                 + 1.04569 w_k1_AXP_s004
                                 + 0.875979 w_k1_DIS_s004
                                 + 0.914387 w_k1_FDX_s004
                                 + 0.866659 w_k1_IBM_s004
                                 + 0.968215 w_k1_INTC_s004
                                 + 0.970232 w_k1_KO_s004
                                 + 1.41632 w_k1_MAT_s004
                                 + 0.97998 w_k1_MSFT_s004
                                 + 1.07541 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.827523 w_k1_ADM_s004
                                 + 1.01649 w_k1_AXP_s004
                                 + 1.43465 w_k1_DIS_s004
                                 + 1.11626 w_k1_FDX_s004
                                 + 1.52055 w_k1_IBM_s004
                                 + 1.23918 w_k1_INTC_s004
                                 + 1.27776 w_k1_KO_s004 + 0.95262 w_k1_MAT_s004
                                 + 1.1868 w_k1_MSFT_s004 + 1.06613 w_k1_PG_s004
                                 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.14035 w_k1_ADM_s004
                                 + 1.37419 w_k1_AXP_s004
                                 + 1.46664 w_k1_DIS_s004
                                 + 1.27349 w_k1_FDX_s004
                                 + 1.13712 w_k1_IBM_s004
                                 + 1.22594 w_k1_INTC_s004
                                 + 1.28101 w_k1_KO_s004 + 1.01933 w_k1_MAT_s004
                                 + 1.23988 w_k1_MSFT_s004
                                 + 1.14253 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 0.540197 w_k1_ADM_s004
                                 + 0.559412 w_k1_AXP_s004
                                 + 0.533501 w_k1_DIS_s004
                                 + 0.679182 w_k1_FDX_s004
                                 + 0.930013 w_k1_IBM_s004
                                 + 0.718087 w_k1_INTC_s004
                                 + 0.732741 w_k1_KO_s004 + 0.6974 w_k1_MAT_s004
                                 + 0.728458 w_k1_MSFT_s004
                                 + 0.841855 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.688815 w_k1_ADM_s004
                                 + 0.701634 w_k1_AXP_s004
                                 + 0.619787 w_k1_DIS_s004
                                 + 0.828565 w_k1_FDX_s004
                                 + 0.67358 w_k1_IBM_s004
                                 + 1.02198 w_k1_INTC_s004
                                 + 0.819852 w_k1_KO_s004
                                 + 0.565302 w_k1_MAT_s004
                                 + 0.950172 w_k1_MSFT_s004
                                 + 0.845748 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.946638 w_k1_ADM_s004
                                 + 1.15761 w_k1_AXP_s004 + 0.9855 w_k1_DIS_s004
                                 + 1.07182 w_k1_FDX_s004
                                 + 0.802604 w_k1_IBM_s004
                                 + 1.13245 w_k1_INTC_s004
                                 + 0.975287 w_k1_KO_s004
                                 + 1.03289 w_k1_MAT_s004
                                 + 1.29524 w_k1_MSFT_s004
                                 + 1.14324 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.812085 w_k1_ADM_s004
                                 + 0.741745 w_k1_AXP_s004
                                 + 0.96465 w_k1_DIS_s004
                                 + 0.830728 w_k1_FDX_s004
                                 + 1.03204 w_k1_IBM_s004
                                 + 0.907072 w_k1_INTC_s004
                                 + 0.932261 w_k1_KO_s004
                                 + 0.711207 w_k1_MAT_s004
                                 + 0.906789 w_k1_MSFT_s004
                                 + 0.944162 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 1.03585 w_k1_ADM_s005
                                 + 1.2453 w_k1_AXP_s005 + 1.20046 w_k1_DIS_s005
                                 + 1.26611 w_k1_FDX_s005
                                 + 0.908183 w_k1_IBM_s005
                                 + 1.17108 w_k1_INTC_s005
                                 + 0.991706 w_k1_KO_s005
                                 + 1.13858 w_k1_MAT_s005
                                 + 1.15866 w_k1_MSFT_s005
                                 + 1.09764 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 1.07058 w_k1_ADM_s005
                                 + 1.20434 w_k1_AXP_s005
                                 + 1.13121 w_k1_DIS_s005
                                 + 0.978726 w_k1_FDX_s005
                                 + 1.43494 w_k1_IBM_s005
                                 + 1.07597 w_k1_INTC_s005
                                 + 1.23628 w_k1_KO_s005 + 1.80046 w_k1_MAT_s005
                                 + 0.928665 w_k1_MSFT_s005
                                 + 1.11923 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 0.763976 w_k1_ADM_s005
                                 + 0.712409 w_k1_AXP_s005
                                 + 0.674829 w_k1_DIS_s005
                                 + 0.775441 w_k1_FDX_s005
                                 + 0.759814 w_k1_IBM_s005
                                 + 0.894091 w_k1_INTC_s005
                                 + 0.771951 w_k1_KO_s005
                                 + 0.849386 w_k1_MAT_s005
                                 + 0.795925 w_k1_MSFT_s005
                                 + 0.789068 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.854958 w_k1_ADM_s005
                                 + 0.751583 w_k1_AXP_s005
                                 + 0.792637 w_k1_DIS_s005
                                 + 0.928714 w_k1_FDX_s005
                                 + 0.871292 w_k1_IBM_s005
                                 + 0.828806 w_k1_INTC_s005
                                 + 0.868617 w_k1_KO_s005
                                 + 0.892293 w_k1_MAT_s005
                                 + 0.871157 w_k1_MSFT_s005
                                 + 0.90488 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.22877 w_k1_ADM_s005
                                 + 1.15669 w_k1_AXP_s005
                                 + 1.76051 w_k1_DIS_s005
                                 + 1.36879 w_k1_FDX_s005 + 1.3064 w_k1_IBM_s005
                                 + 1.31626 w_k1_INTC_s005
                                 + 1.35662 w_k1_KO_s005
                                 + 0.796664 w_k1_MAT_s005
                                 + 1.1388 w_k1_MSFT_s005 + 1.17483 w_k1_PG_s005
                                 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 0.913688 w_k1_ADM_s005
                                 + 0.944206 w_k1_AXP_s005
                                 + 1.06189 w_k1_DIS_s005
                                 + 1.03527 w_k1_FDX_s005
                                 + 0.922376 w_k1_IBM_s005
                                 + 1.05698 w_k1_INTC_s005
                                 + 1.06589 w_k1_KO_s005
                                 + 0.835872 w_k1_MAT_s005
                                 + 0.969852 w_k1_MSFT_s005
                                 + 1.13852 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.816083 w_k1_ADM_s005
                                 + 1.02037 w_k1_AXP_s005
                                 + 0.973304 w_k1_DIS_s005
                                 + 1.0315 w_k1_FDX_s005 + 1.30528 w_k1_IBM_s005
                                 + 1.14696 w_k1_INTC_s005
                                 + 1.19438 w_k1_KO_s005
                                 + 0.944084 w_k1_MAT_s005
                                 + 1.01975 w_k1_MSFT_s005
                                 + 1.03389 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.871452 w_k1_ADM_s005
                                 + 0.846739 w_k1_AXP_s005
                                 + 0.893153 w_k1_DIS_s005
                                 + 0.835882 w_k1_FDX_s005
                                 + 1.18508 w_k1_IBM_s005
                                 + 1.14896 w_k1_INTC_s005
                                 + 1.07738 w_k1_KO_s005
                                 + 0.733852 w_k1_MAT_s005
                                 + 0.835271 w_k1_MSFT_s005
                                 + 0.931157 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.855597 w_k1_ADM_s006
                                 + 0.655034 w_k1_AXP_s006
                                 + 0.689875 w_k1_DIS_s006
                                 + 0.849882 w_k1_FDX_s006
                                 + 0.695945 w_k1_IBM_s006
                                 + 0.896075 w_k1_INTC_s006
                                 + 0.881445 w_k1_KO_s006
                                 + 0.437404 w_k1_MAT_s006
                                 + 0.887742 w_k1_MSFT_s006
                                 + 0.909661 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.889856 w_k1_ADM_s006
                                 + 0.965433 w_k1_AXP_s006
                                 + 0.987457 w_k1_DIS_s006
                                 + 0.886768 w_k1_FDX_s006
                                 + 1.23474 w_k1_IBM_s006
                                 + 1.18069 w_k1_INTC_s006
                                 + 1.31429 w_k1_KO_s006 + 1.15979 w_k1_MAT_s006
                                 + 1.13383 w_k1_MSFT_s006
                                 + 1.11226 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.949518 w_k1_ADM_s006
                                 + 0.995686 w_k1_AXP_s006
                                 + 0.93304 w_k1_DIS_s006
                                 + 1.14208 w_k1_FDX_s006
                                 + 0.788074 w_k1_IBM_s006
                                 + 1.26434 w_k1_INTC_s006
                                 + 0.980192 w_k1_KO_s006
                                 + 0.671064 w_k1_MAT_s006
                                 + 1.2094 w_k1_MSFT_s006
                                 + 0.942317 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.09449 w_k1_ADM_s006
                                 + 1.04158 w_k1_AXP_s006
                                 + 1.06066 w_k1_DIS_s006
                                 + 0.980111 w_k1_FDX_s006
                                 + 1.83094 w_k1_IBM_s006
                                 + 0.943332 w_k1_INTC_s006
                                 + 1.21535 w_k1_KO_s006 + 1.51031 w_k1_MAT_s006
                                 + 1.07175 w_k1_MSFT_s006
                                 + 1.21716 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.826299 w_k1_ADM_s006
                                 + 0.83211 w_k1_AXP_s006
                                 + 0.739261 w_k1_DIS_s006
                                 + 0.777925 w_k1_FDX_s006
                                 + 0.880199 w_k1_IBM_s006
                                 + 0.834783 w_k1_INTC_s006
                                 + 0.913222 w_k1_KO_s006
                                 + 0.718168 w_k1_MAT_s006
                                 + 0.818017 w_k1_MSFT_s006
                                 + 1.0254 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.20889 w_k1_ADM_s006
                                 + 1.30273 w_k1_AXP_s006
                                 + 1.25588 w_k1_DIS_s006
                                 + 1.23011 w_k1_FDX_s006
                                 + 1.30523 w_k1_IBM_s006
                                 + 1.56776 w_k1_INTC_s006
                                 + 1.52101 w_k1_KO_s006 + 1.34936 w_k1_MAT_s006
                                 + 1.31697 w_k1_MSFT_s006
                                 + 1.21377 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.987555 w_k1_ADM_s006
                                 + 1.22019 w_k1_AXP_s006
                                 + 0.787649 w_k1_DIS_s006
                                 + 1.01003 w_k1_FDX_s006
                                 + 0.870991 w_k1_IBM_s006
                                 + 0.865116 w_k1_INTC_s006
                                 + 0.846651 w_k1_KO_s006
                                 + 1.09646 w_k1_MAT_s006
                                 + 0.969822 w_k1_MSFT_s006
                                 + 0.987235 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 0.906889 w_k1_ADM_s006
                                 + 0.884457 w_k1_AXP_s006
                                 + 0.608532 w_k1_DIS_s006
                                 + 0.82182 w_k1_FDX_s006
                                 + 1.26893 w_k1_IBM_s006
                                 + 0.836279 w_k1_INTC_s006
                                 + 0.954833 w_k1_KO_s006
                                 + 0.907205 w_k1_MAT_s006
                                 + 0.764294 w_k1_MSFT_s006
                                 + 1.01918 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.828819 w_k1_ADM_s007
                                 + 1.87667 w_k1_AXP_s007
                                 + 1.43722 w_k1_DIS_s007
                                 + 1.35588 w_k1_FDX_s007
                                 + 1.41271 w_k1_IBM_s007
                                 + 1.27072 w_k1_INTC_s007
                                 + 1.27077 w_k1_KO_s007 + 2.44257 w_k1_MAT_s007
                                 + 1.22971 w_k1_MSFT_s007
                                 + 1.14871 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.92286 w_k1_ADM_s007
                                 + 1.14412 w_k1_AXP_s007
                                 + 0.944653 w_k1_DIS_s007
                                 + 1.15048 w_k1_FDX_s007
                                 + 0.731412 w_k1_IBM_s007
                                 + 1.102 w_k1_INTC_s007 + 1.00853 w_k1_KO_s007
                                 + 0.928036 w_k1_MAT_s007
                                 + 1.09631 w_k1_MSFT_s007
                                 + 0.953282 w_k1_PG_s007 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.15866 w_k1_ADM_s007
                                 + 0.978847 w_k1_AXP_s007
                                 + 1.4882 w_k1_DIS_s007 + 1.16084 w_k1_FDX_s007
                                 + 1.04462 w_k1_IBM_s007
                                 + 1.24962 w_k1_INTC_s007
                                 + 1.17329 w_k1_KO_s007 + 1.1802 w_k1_MAT_s007
                                 + 1.18724 w_k1_MSFT_s007
                                 + 1.10549 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.808305 w_k1_ADM_s007
                                 + 0.754478 w_k1_AXP_s007
                                 + 0.824033 w_k1_DIS_s007
                                 + 0.831335 w_k1_FDX_s007
                                 + 0.930753 w_k1_IBM_s007
                                 + 1.05368 w_k1_INTC_s007
                                 + 0.959727 w_k1_KO_s007
                                 + 0.805667 w_k1_MAT_s007
                                 + 0.902194 w_k1_MSFT_s007
                                 + 0.870361 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.78974 w_k1_ADM_s007
                                 + 1.01326 w_k1_AXP_s007
                                 + 0.973116 w_k1_DIS_s007
                                 + 0.946585 w_k1_FDX_s007
                                 + 1.18802 w_k1_IBM_s007
                                 + 0.94936 w_k1_INTC_s007
                                 + 1.09109 w_k1_KO_s007 + 1.26162 w_k1_MAT_s007
                                 + 0.81724 w_k1_MSFT_s007
                                 + 0.933719 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.542285 w_k1_ADM_s007
                                 + 0.722519 w_k1_AXP_s007
                                 + 0.628933 w_k1_DIS_s007
                                 + 0.694507 w_k1_FDX_s007
                                 + 0.875092 w_k1_IBM_s007
                                 + 0.837725 w_k1_INTC_s007
                                 + 0.813512 w_k1_KO_s007
                                 + 0.933987 w_k1_MAT_s007
                                 + 0.647544 w_k1_MSFT_s007
                                 + 0.91803 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.554108 w_k1_ADM_s007
                                 + 0.563583 w_k1_AXP_s007
                                 + 0.693883 w_k1_DIS_s007
                                 + 0.776922 w_k1_FDX_s007
                                 + 0.641371 w_k1_IBM_s007
                                 + 0.759301 w_k1_INTC_s007
                                 + 0.654843 w_k1_KO_s007
                                 + 0.665736 w_k1_MAT_s007
                                 + 0.816236 w_k1_MSFT_s007
                                 + 0.769566 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.886185 w_k1_ADM_s007
                                 + 1.11915 w_k1_AXP_s007
                                 + 1.25277 w_k1_DIS_s007
                                 + 1.00652 w_k1_FDX_s007 + 1.6173 w_k1_IBM_s007
                                 + 0.931935 w_k1_INTC_s007
                                 + 1.32786 w_k1_KO_s007 + 1.85071 w_k1_MAT_s007
                                 + 0.9961 w_k1_MSFT_s007 + 1.18814 w_k1_PG_s007
                                 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.13207 w_k1_ADM_s010
                                 + 1.23473 w_k1_AXP_s010
                                 + 1.18364 w_k1_DIS_s010 + 1.1095 w_k1_FDX_s010
                                 + 0.991178 w_k1_IBM_s010
                                 + 0.882671 w_k1_INTC_s010
                                 + 0.932625 w_k1_KO_s010
                                 + 1.09619 w_k1_MAT_s010
                                 + 1.05659 w_k1_MSFT_s010
                                 + 1.11091 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.07306 w_k1_ADM_s010
                                 + 0.838933 w_k1_AXP_s010
                                 + 1.24683 w_k1_DIS_s010
                                 + 1.04238 w_k1_FDX_s010
                                 + 0.812605 w_k1_IBM_s010
                                 + 0.970215 w_k1_INTC_s010
                                 + 1.06116 w_k1_KO_s010
                                 + 0.684742 w_k1_MAT_s010
                                 + 1.02887 w_k1_MSFT_s010
                                 + 0.912885 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.14288 w_k1_ADM_s010
                                 + 0.936463 w_k1_AXP_s010
                                 + 1.62721 w_k1_DIS_s010
                                 + 1.27869 w_k1_FDX_s010
                                 + 0.887446 w_k1_IBM_s010
                                 + 1.26566 w_k1_INTC_s010
                                 + 1.03604 w_k1_KO_s010
                                 + 0.657036 w_k1_MAT_s010
                                 + 1.21368 w_k1_MSFT_s010
                                 + 1.10392 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.721298 w_k1_ADM_s010
                                 + 0.868598 w_k1_AXP_s010
                                 + 0.804753 w_k1_DIS_s010
                                 + 0.82508 w_k1_FDX_s010
                                 + 0.888735 w_k1_IBM_s010
                                 + 0.948354 w_k1_INTC_s010
                                 + 0.788234 w_k1_KO_s010
                                 + 1.07804 w_k1_MAT_s010
                                 + 0.789697 w_k1_MSFT_s010
                                 + 0.991739 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.904735 w_k1_ADM_s010
                                 + 0.870271 w_k1_AXP_s010
                                 + 1.1345 w_k1_DIS_s010
                                 + 0.953059 w_k1_FDX_s010
                                 + 1.18643 w_k1_IBM_s010
                                 + 0.815389 w_k1_INTC_s010
                                 + 1.00896 w_k1_KO_s010 + 1.09435 w_k1_MAT_s010
                                 + 0.894087 w_k1_MSFT_s010
                                 + 1.00879 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 0.91176 w_k1_ADM_s010
                                 + 0.810141 w_k1_AXP_s010
                                 + 0.903998 w_k1_DIS_s010
                                 + 0.951904 w_k1_FDX_s010
                                 + 0.686614 w_k1_IBM_s010
                                 + 0.795262 w_k1_INTC_s010
                                 + 0.747236 w_k1_KO_s010
                                 + 0.693224 w_k1_MAT_s010
                                 + 0.830912 w_k1_MSFT_s010
                                 + 0.897852 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 0.788526 w_k1_ADM_s010
                                 + 1.17966 w_k1_AXP_s010
                                 + 1.15358 w_k1_DIS_s010
                                 + 0.94897 w_k1_FDX_s010
                                 + 1.51803 w_k1_IBM_s010
                                 + 0.932307 w_k1_INTC_s010
                                 + 1.05967 w_k1_KO_s010 + 1.59472 w_k1_MAT_s010
                                 + 0.996415 w_k1_MSFT_s010
                                 + 1.0285 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.1169 w_k1_ADM_s010
                                 + 1.33216 w_k1_AXP_s010
                                 + 1.64515 w_k1_DIS_s010 + 1.2135 w_k1_FDX_s010
                                 + 1.1679 w_k1_IBM_s010 + 1.1894 w_k1_INTC_s010
                                 + 1.33218 w_k1_KO_s010 + 1.42281 w_k1_MAT_s010
                                 + 1.12621 w_k1_MSFT_s010
                                 + 1.29016 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.03335 w_k1_ADM_s011
                                 + 0.981 w_k1_AXP_s011 + 0.980879 w_k1_DIS_s011
                                 + 1.09938 w_k1_FDX_s011
                                 + 0.740778 w_k1_IBM_s011
                                 + 1.10364 w_k1_INTC_s011
                                 + 0.962241 w_k1_KO_s011
                                 + 0.820254 w_k1_MAT_s011
                                 + 1.02873 w_k1_MSFT_s011
                                 + 1.00718 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.968845 w_k1_ADM_s011
                                 + 0.712419 w_k1_AXP_s011
                                 + 0.726378 w_k1_DIS_s011
                                 + 0.736796 w_k1_FDX_s011
                                 + 1.13701 w_k1_IBM_s011
                                 + 0.905703 w_k1_INTC_s011
                                 + 0.966464 w_k1_KO_s011
                                 + 1.28642 w_k1_MAT_s011
                                 + 0.856574 w_k1_MSFT_s011
                                 + 0.936709 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.02511 w_k1_ADM_s011
                                 + 1.04946 w_k1_AXP_s011
                                 + 0.932436 w_k1_DIS_s011
                                 + 0.870006 w_k1_FDX_s011
                                 + 1.30158 w_k1_IBM_s011
                                 + 0.948216 w_k1_INTC_s011
                                 + 1.15129 w_k1_KO_s011 + 1.34865 w_k1_MAT_s011
                                 + 0.943382 w_k1_MSFT_s011
                                 + 1.07503 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.891986 w_k1_ADM_s011
                                 + 0.828429 w_k1_AXP_s011
                                 + 0.604919 w_k1_DIS_s011
                                 + 0.855042 w_k1_FDX_s011
                                 + 0.720312 w_k1_IBM_s011
                                 + 0.824414 w_k1_INTC_s011
                                 + 0.796379 w_k1_KO_s011
                                 + 0.935644 w_k1_MAT_s011
                                 + 0.773249 w_k1_MSFT_s011
                                 + 0.89023 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 0.919519 w_k1_ADM_s011
                                 + 0.917775 w_k1_AXP_s011
                                 + 0.891701 w_k1_DIS_s011
                                 + 0.989843 w_k1_FDX_s011
                                 + 0.975551 w_k1_IBM_s011
                                 + 1.13305 w_k1_INTC_s011
                                 + 1.03227 w_k1_KO_s011 + 1.27329 w_k1_MAT_s011
                                 + 1.02849 w_k1_MSFT_s011
                                 + 0.948821 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.956573 w_k1_ADM_s011
                                 + 1.22836 w_k1_AXP_s011
                                 + 1.19077 w_k1_DIS_s011
                                 + 1.13997 w_k1_FDX_s011
                                 + 1.10368 w_k1_IBM_s011
                                 + 0.936995 w_k1_INTC_s011
                                 + 1.17934 w_k1_KO_s011 + 1.50749 w_k1_MAT_s011
                                 + 1.16277 w_k1_MSFT_s011
                                 + 1.12614 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.782883 w_k1_ADM_s011
                                 + 0.755549 w_k1_AXP_s011
                                 + 0.726479 w_k1_DIS_s011
                                 + 0.798093 w_k1_FDX_s011
                                 + 0.737987 w_k1_IBM_s011
                                 + 0.861393 w_k1_INTC_s011
                                 + 0.863163 w_k1_KO_s011
                                 + 0.739431 w_k1_MAT_s011
                                 + 0.786713 w_k1_MSFT_s011
                                 + 0.774569 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 1.09907 w_k1_ADM_s011
                                 + 1.66103 w_k1_AXP_s011
                                 + 1.52375 w_k1_DIS_s011
                                 + 1.27512 w_k1_FDX_s011
                                 + 1.37844 w_k1_IBM_s011
                                 + 1.18598 w_k1_INTC_s011
                                 + 1.59087 w_k1_KO_s011 + 1.57175 w_k1_MAT_s011
                                 + 1.31132 w_k1_MSFT_s011
                                 + 1.26459 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.00076 w_k1_ADM_s012
                                 + 1.41675 w_k1_AXP_s012
                                 + 1.13092 w_k1_DIS_s012
                                 + 1.17826 w_k1_FDX_s012
                                 + 0.933656 w_k1_IBM_s012
                                 + 0.963881 w_k1_INTC_s012
                                 + 1.03503 w_k1_KO_s012 + 1.55252 w_k1_MAT_s012
                                 + 1.16521 w_k1_MSFT_s012
                                 + 1.03113 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.850388 w_k1_ADM_s012
                                 + 0.812853 w_k1_AXP_s012
                                 + 1.15663 w_k1_DIS_s012
                                 + 1.00292 w_k1_FDX_s012
                                 + 0.857842 w_k1_IBM_s012
                                 + 0.879058 w_k1_INTC_s012
                                 + 0.942959 w_k1_KO_s012
                                 + 0.872741 w_k1_MAT_s012
                                 + 1.2604 w_k1_MSFT_s012
                                 + 0.944787 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.682345 w_k1_ADM_s012
                                 + 0.856231 w_k1_AXP_s012
                                 + 0.907879 w_k1_DIS_s012
                                 + 0.975204 w_k1_FDX_s012
                                 + 0.977013 w_k1_IBM_s012
                                 + 1.09723 w_k1_INTC_s012
                                 + 0.930412 w_k1_KO_s012
                                 + 1.17611 w_k1_MAT_s012
                                 + 1.19851 w_k1_MSFT_s012
                                 + 0.997526 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 0.881246 w_k1_ADM_s012
                                 + 0.816412 w_k1_AXP_s012
                                 + 0.685516 w_k1_DIS_s012
                                 + 0.870591 w_k1_FDX_s012
                                 + 0.705449 w_k1_IBM_s012
                                 + 0.789599 w_k1_INTC_s012
                                 + 0.715055 w_k1_KO_s012
                                 + 0.969967 w_k1_MAT_s012
                                 + 1.0729 w_k1_MSFT_s012
                                 + 0.946365 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.966749 w_k1_ADM_s012
                                 + 0.916218 w_k1_AXP_s012 + 0.897 w_k1_DIS_s012
                                 + 0.91179 w_k1_FDX_s012
                                 + 1.31094 w_k1_IBM_s012
                                 + 0.967113 w_k1_INTC_s012
                                 + 1.13089 w_k1_KO_s012 + 1.54021 w_k1_MAT_s012
                                 + 1.05758 w_k1_MSFT_s012
                                 + 1.14807 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 1.124 w_k1_ADM_s012
                                 + 1.23225 w_k1_AXP_s012
                                 + 1.35878 w_k1_DIS_s012
                                 + 1.03965 w_k1_FDX_s012 + 1.619 w_k1_IBM_s012
                                 + 0.839502 w_k1_INTC_s012
                                 + 1.28894 w_k1_KO_s012 + 2.18467 w_k1_MAT_s012
                                 + 1.30318 w_k1_MSFT_s012
                                 + 1.22908 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.979012 w_k1_ADM_s012
                                 + 1.33952 w_k1_AXP_s012 + 1.5991 w_k1_DIS_s012
                                 + 1.30719 w_k1_FDX_s012
                                 + 0.865207 w_k1_IBM_s012
                                 + 1.32714 w_k1_INTC_s012
                                 + 1.17548 w_k1_KO_s012 + 1.15113 w_k1_MAT_s012
                                 + 1.47434 w_k1_MSFT_s012
                                 + 1.25091 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.02732 w_k1_ADM_s012
                                 + 0.656455 w_k1_AXP_s012
                                 + 0.670623 w_k1_DIS_s012
                                 + 0.849695 w_k1_FDX_s012
                                 + 0.513434 w_k1_IBM_s012
                                 + 0.752005 w_k1_INTC_s012
                                 + 0.607275 w_k1_KO_s012
                                 + 0.463745 w_k1_MAT_s012
                                 + 0.877414 w_k1_MSFT_s012
                                 + 0.815306 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.33594 w_k1_ADM_s013
                                 + 1.03854 w_k1_AXP_s013
                                 + 0.990837 w_k1_DIS_s013
                                 + 1.05834 w_k1_FDX_s013
                                 + 0.749212 w_k1_IBM_s013
                                 + 0.959973 w_k1_INTC_s013
                                 + 0.944691 w_k1_KO_s013
                                 + 0.765452 w_k1_MAT_s013
                                 + 1.11538 w_k1_MSFT_s013
                                 + 1.00308 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 0.866955 w_k1_ADM_s013
                                 + 0.920707 w_k1_AXP_s013
                                 + 0.828448 w_k1_DIS_s013
                                 + 0.926639 w_k1_FDX_s013
                                 + 0.703697 w_k1_IBM_s013
                                 + 1.02518 w_k1_INTC_s013
                                 + 0.773182 w_k1_KO_s013
                                 + 1.27221 w_k1_MAT_s013
                                 + 0.947508 w_k1_MSFT_s013
                                 + 0.878504 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.931285 w_k1_ADM_s013
                                 + 0.777825 w_k1_AXP_s013
                                 + 0.748613 w_k1_DIS_s013
                                 + 0.844374 w_k1_FDX_s013
                                 + 0.773689 w_k1_IBM_s013
                                 + 0.97717 w_k1_INTC_s013
                                 + 0.801842 w_k1_KO_s013
                                 + 0.721427 w_k1_MAT_s013
                                 + 0.785732 w_k1_MSFT_s013
                                 + 0.866632 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.966018 w_k1_ADM_s013
                                 + 1.23427 w_k1_AXP_s013
                                 + 1.00726 w_k1_DIS_s013
                                 + 0.985733 w_k1_FDX_s013
                                 + 1.32771 w_k1_IBM_s013
                                 + 0.946297 w_k1_INTC_s013
                                 + 1.03145 w_k1_KO_s013 + 2.14877 w_k1_MAT_s013
                                 + 0.942453 w_k1_MSFT_s013
                                 + 1.0271 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 0.913358 w_k1_ADM_s013
                                 + 0.775021 w_k1_AXP_s013
                                 + 0.835805 w_k1_DIS_s013
                                 + 0.831612 w_k1_FDX_s013
                                 + 1.35926 w_k1_IBM_s013
                                 + 0.824186 w_k1_INTC_s013
                                 + 0.949177 w_k1_KO_s013
                                 + 1.30406 w_k1_MAT_s013
                                 + 0.816486 w_k1_MSFT_s013
                                 + 0.937719 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 0.777113 w_k1_ADM_s013
                                 + 0.786583 w_k1_AXP_s013
                                 + 0.842408 w_k1_DIS_s013
                                 + 0.827317 w_k1_FDX_s013
                                 + 0.953082 w_k1_IBM_s013
                                 + 0.907511 w_k1_INTC_s013
                                 + 0.928906 w_k1_KO_s013
                                 + 1.03886 w_k1_MAT_s013
                                 + 0.906738 w_k1_MSFT_s013
                                 + 0.913486 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.794421 w_k1_ADM_s013
                                 + 0.705884 w_k1_AXP_s013
                                 + 0.537648 w_k1_DIS_s013
                                 + 0.709463 w_k1_FDX_s013
                                 + 0.681786 w_k1_IBM_s013
                                 + 0.703011 w_k1_INTC_s013
                                 + 0.683011 w_k1_KO_s013
                                 + 0.85836 w_k1_MAT_s013
                                 + 0.734859 w_k1_MSFT_s013
                                 + 0.771042 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.22496 w_k1_ADM_s013
                                 + 1.21301 w_k1_AXP_s013
                                 + 1.43435 w_k1_DIS_s013
                                 + 1.24404 w_k1_FDX_s013 + 1.175 w_k1_IBM_s013
                                 + 1.2664 w_k1_INTC_s013 + 1.19164 w_k1_KO_s013
                                 + 1.38961 w_k1_MAT_s013
                                 + 1.08018 w_k1_MSFT_s013
                                 + 1.01616 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 1.49547 w_k1_ADM_s014
                                 + 1.79431 w_k1_AXP_s014
                                 + 1.59399 w_k1_DIS_s014
                                 + 1.47797 w_k1_FDX_s014 + 1.1569 w_k1_IBM_s014
                                 + 1.323 w_k1_INTC_s014 + 1.13795 w_k1_KO_s014
                                 + 1.04476 w_k1_MAT_s014
                                 + 1.17869 w_k1_MSFT_s014
                                 + 1.08752 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.26195 w_k1_ADM_s014
                                 + 0.954501 w_k1_AXP_s014
                                 + 1.1292 w_k1_DIS_s014 + 1.03413 w_k1_FDX_s014
                                 + 1.07026 w_k1_IBM_s014
                                 + 1.12262 w_k1_INTC_s014
                                 + 1.12549 w_k1_KO_s014
                                 + 0.636639 w_k1_MAT_s014
                                 + 1.06769 w_k1_MSFT_s014
                                 + 0.977498 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.657842 w_k1_ADM_s014
                                 + 0.559498 w_k1_AXP_s014
                                 + 0.483997 w_k1_DIS_s014
                                 + 0.5806 w_k1_FDX_s014
                                 + 0.705941 w_k1_IBM_s014
                                 + 0.715116 w_k1_INTC_s014
                                 + 0.593466 w_k1_KO_s014
                                 + 0.483184 w_k1_MAT_s014
                                 + 0.641021 w_k1_MSFT_s014
                                 + 0.637524 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.44483 w_k1_ADM_s014
                                 + 1.46011 w_k1_AXP_s014
                                 + 1.51199 w_k1_DIS_s014
                                 + 1.36043 w_k1_FDX_s014
                                 + 1.25598 w_k1_IBM_s014
                                 + 1.41107 w_k1_INTC_s014
                                 + 1.30905 w_k1_KO_s014 + 1.16407 w_k1_MAT_s014
                                 + 1.21653 w_k1_MSFT_s014
                                 + 1.12999 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.879033 w_k1_ADM_s014
                                 + 0.688224 w_k1_AXP_s014
                                 + 0.729928 w_k1_DIS_s014
                                 + 0.775129 w_k1_FDX_s014
                                 + 0.826167 w_k1_IBM_s014
                                 + 0.902623 w_k1_INTC_s014
                                 + 0.788387 w_k1_KO_s014
                                 + 0.619797 w_k1_MAT_s014
                                 + 0.87482 w_k1_MSFT_s014
                                 + 0.734923 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.989543 w_k1_ADM_s014
                                 + 1.17317 w_k1_AXP_s014
                                 + 1.18645 w_k1_DIS_s014
                                 + 1.03407 w_k1_FDX_s014
                                 + 1.48372 w_k1_IBM_s014
                                 + 1.33028 w_k1_INTC_s014
                                 + 1.27339 w_k1_KO_s014 + 1.29244 w_k1_MAT_s014
                                 + 1.14083 w_k1_MSFT_s014 + 1.1097 w_k1_PG_s014
                                 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.14172 w_k1_ADM_s014
                                 + 0.964251 w_k1_AXP_s014
                                 + 0.924409 w_k1_DIS_s014
                                 + 1.02845 w_k1_FDX_s014
                                 + 0.781234 w_k1_IBM_s014
                                 + 1.12274 w_k1_INTC_s014
                                 + 0.76735 w_k1_KO_s014
                                 + 0.430858 w_k1_MAT_s014
                                 + 0.967687 w_k1_MSFT_s014
                                 + 0.776861 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.20962 w_k1_ADM_s014
                                 + 1.19906 w_k1_AXP_s014
                                 + 1.27353 w_k1_DIS_s014
                                 + 1.40731 w_k1_FDX_s014
                                 + 0.768261 w_k1_IBM_s014
                                 + 1.17825 w_k1_INTC_s014
                                 + 0.93273 w_k1_KO_s014
                                 + 0.881532 w_k1_MAT_s014
                                 + 1.3223 w_k1_MSFT_s014 + 1.06784 w_k1_PG_s014
                                 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.04252 w_k1_ADM_s015
                                 + 1.47878 w_k1_AXP_s015
                                 + 1.31386 w_k1_DIS_s015
                                 + 1.12117 w_k1_FDX_s015 + 1.6792 w_k1_IBM_s015
                                 + 1.15675 w_k1_INTC_s015
                                 + 1.53711 w_k1_KO_s015 + 2.13199 w_k1_MAT_s015
                                 + 1.10438 w_k1_MSFT_s015
                                 + 1.07993 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 0.673011 w_k1_ADM_s015
                                 + 0.953189 w_k1_AXP_s015
                                 + 0.820669 w_k1_DIS_s015
                                 + 0.785081 w_k1_FDX_s015
                                 + 1.12193 w_k1_IBM_s015
                                 + 0.858784 w_k1_INTC_s015
                                 + 0.960765 w_k1_KO_s015
                                 + 1.03493 w_k1_MAT_s015
                                 + 0.726177 w_k1_MSFT_s015
                                 + 0.799845 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.897769 w_k1_ADM_s015
                                 + 0.987655 w_k1_AXP_s015
                                 + 0.993372 w_k1_DIS_s015
                                 + 0.988548 w_k1_FDX_s015
                                 + 0.852566 w_k1_IBM_s015
                                 + 1.0798 w_k1_INTC_s015
                                 + 0.929258 w_k1_KO_s015
                                 + 0.799355 w_k1_MAT_s015
                                 + 0.910423 w_k1_MSFT_s015
                                 + 0.975352 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.925022 w_k1_ADM_s015
                                 + 1.49669 w_k1_AXP_s015
                                 + 1.49952 w_k1_DIS_s015
                                 + 1.33323 w_k1_FDX_s015
                                 + 1.12147 w_k1_IBM_s015
                                 + 1.29322 w_k1_INTC_s015
                                 + 1.17705 w_k1_KO_s015 + 1.50194 w_k1_MAT_s015
                                 + 1.07145 w_k1_MSFT_s015
                                 + 0.950659 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.750509 w_k1_ADM_s015
                                 + 0.617111 w_k1_AXP_s015
                                 + 0.704686 w_k1_DIS_s015
                                 + 0.773722 w_k1_FDX_s015
                                 + 0.661792 w_k1_IBM_s015
                                 + 0.822342 w_k1_INTC_s015
                                 + 0.693914 w_k1_KO_s015
                                 + 0.566208 w_k1_MAT_s015
                                 + 0.759815 w_k1_MSFT_s015
                                 + 0.62069 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.854294 w_k1_ADM_s015
                                 + 1.03184 w_k1_AXP_s015
                                 + 0.994474 w_k1_DIS_s015
                                 + 1.01855 w_k1_FDX_s015
                                 + 0.924154 w_k1_IBM_s015
                                 + 0.914864 w_k1_INTC_s015
                                 + 0.99634 w_k1_KO_s015
                                 + 0.941569 w_k1_MAT_s015
                                 + 0.854987 w_k1_MSFT_s015
                                 + 0.902064 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.949296 w_k1_ADM_s015
                                 + 0.920587 w_k1_AXP_s015
                                 + 0.881958 w_k1_DIS_s015
                                 + 1.01307 w_k1_FDX_s015
                                 + 0.669473 w_k1_IBM_s015
                                 + 0.926341 w_k1_INTC_s015
                                 + 0.749176 w_k1_KO_s015
                                 + 0.507435 w_k1_MAT_s015
                                 + 0.932508 w_k1_MSFT_s015
                                 + 0.799536 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.862565 w_k1_ADM_s015
                                 + 0.967247 w_k1_AXP_s015
                                 + 1.2094 w_k1_DIS_s015 + 1.08554 w_k1_FDX_s015
                                 + 1.09798 w_k1_IBM_s015
                                 + 0.868374 w_k1_INTC_s015
                                 + 1.26508 w_k1_KO_s015 + 1.00979 w_k1_MAT_s015
                                 + 0.985877 w_k1_MSFT_s015
                                 + 0.990121 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.04647 w_k1_ADM_s016
                                 + 1.38708 w_k1_AXP_s016
                                 + 1.13393 w_k1_DIS_s016
                                 + 1.23102 w_k1_FDX_s016
                                 + 0.754437 w_k1_IBM_s016
                                 + 1.13178 w_k1_INTC_s016
                                 + 0.868772 w_k1_KO_s016
                                 + 0.768692 w_k1_MAT_s016
                                 + 1.04511 w_k1_MSFT_s016 + 1.0325 w_k1_PG_s016
                                 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.617115 w_k1_ADM_s016
                                 + 0.78019 w_k1_AXP_s016
                                 + 0.655383 w_k1_DIS_s016
                                 + 0.776884 w_k1_FDX_s016
                                 + 0.934357 w_k1_IBM_s016
                                 + 0.758286 w_k1_INTC_s016
                                 + 0.718616 w_k1_KO_s016
                                 + 0.748042 w_k1_MAT_s016
                                 + 0.728558 w_k1_MSFT_s016
                                 + 0.77247 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.11273 w_k1_ADM_s016
                                 + 1.25537 w_k1_AXP_s016
                                 + 1.14278 w_k1_DIS_s016
                                 + 1.12132 w_k1_FDX_s016
                                 + 1.02103 w_k1_IBM_s016
                                 + 1.21837 w_k1_INTC_s016
                                 + 0.908948 w_k1_KO_s016
                                 + 0.921558 w_k1_MAT_s016
                                 + 1.11604 w_k1_MSFT_s016
                                 + 0.996875 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.0691 w_k1_ADM_s016
                                 + 1.30509 w_k1_AXP_s016
                                 + 1.43364 w_k1_DIS_s016
                                 + 1.09184 w_k1_FDX_s016
                                 + 1.37983 w_k1_IBM_s016
                                 + 0.937617 w_k1_INTC_s016
                                 + 1.15735 w_k1_KO_s016 + 1.20679 w_k1_MAT_s016
                                 + 1.01901 w_k1_MSFT_s016
                                 + 0.911192 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 0.972664 w_k1_ADM_s016
                                 + 1.39546 w_k1_AXP_s016 + 1.4237 w_k1_DIS_s016
                                 + 1.24391 w_k1_FDX_s016
                                 + 1.72594 w_k1_IBM_s016
                                 + 1.02971 w_k1_INTC_s016
                                 + 1.38992 w_k1_KO_s016 + 1.7678 w_k1_MAT_s016
                                 + 0.958138 w_k1_MSFT_s016
                                 + 1.2387 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.03708 w_k1_ADM_s016
                                 + 0.960641 w_k1_AXP_s016
                                 + 1.04672 w_k1_DIS_s016 + 1.0105 w_k1_FDX_s016
                                 + 1.17622 w_k1_IBM_s016
                                 + 0.992373 w_k1_INTC_s016
                                 + 1.1649 w_k1_KO_s016 + 1.08599 w_k1_MAT_s016
                                 + 0.959359 w_k1_MSFT_s016
                                 + 0.962743 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.960318 w_k1_ADM_s016
                                 + 0.926137 w_k1_AXP_s016
                                 + 0.727931 w_k1_DIS_s016
                                 + 0.95743 w_k1_FDX_s016
                                 + 0.745729 w_k1_IBM_s016
                                 + 0.787728 w_k1_INTC_s016
                                 + 0.736485 w_k1_KO_s016
                                 + 0.767832 w_k1_MAT_s016
                                 + 0.86479 w_k1_MSFT_s016
                                 + 0.80172 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.18552 w_k1_ADM_s016
                                 + 1.77086 w_k1_AXP_s016
                                 + 1.53893 w_k1_DIS_s016
                                 + 1.50184 w_k1_FDX_s016
                                 + 1.28226 w_k1_IBM_s016
                                 + 1.35147 w_k1_INTC_s016
                                 + 1.22691 w_k1_KO_s016 + 1.50465 w_k1_MAT_s016
                                 + 1.32903 w_k1_MSFT_s016
                                 + 1.15862 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 0.793808 w_k1_ADM_s017
                                 + 0.68483 w_k1_AXP_s017
                                 + 0.587923 w_k1_DIS_s017
                                 + 0.837558 w_k1_FDX_s017
                                 + 0.658344 w_k1_IBM_s017
                                 + 0.781545 w_k1_INTC_s017
                                 + 0.765613 w_k1_KO_s017
                                 + 0.924942 w_k1_MAT_s017
                                 + 0.78116 w_k1_MSFT_s017
                                 + 0.850137 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.926904 w_k1_ADM_s017
                                 + 0.870762 w_k1_AXP_s017
                                 + 0.799278 w_k1_DIS_s017
                                 + 1.04586 w_k1_FDX_s017
                                 + 0.695535 w_k1_IBM_s017
                                 + 1.00122 w_k1_INTC_s017
                                 + 0.791692 w_k1_KO_s017
                                 + 0.810377 w_k1_MAT_s017
                                 + 0.995661 w_k1_MSFT_s017
                                 + 0.930155 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.27047 w_k1_ADM_s017
                                 + 1.66381 w_k1_AXP_s017
                                 + 1.40181 w_k1_DIS_s017
                                 + 1.33637 w_k1_FDX_s017
                                 + 1.42472 w_k1_IBM_s017
                                 + 1.15969 w_k1_INTC_s017
                                 + 1.42094 w_k1_KO_s017 + 1.97518 w_k1_MAT_s017
                                 + 1.18025 w_k1_MSFT_s017
                                 + 1.21199 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.962775 w_k1_ADM_s017
                                 + 0.763839 w_k1_AXP_s017
                                 + 0.893081 w_k1_DIS_s017
                                 + 0.900862 w_k1_FDX_s017
                                 + 0.960881 w_k1_IBM_s017
                                 + 0.881739 w_k1_INTC_s017
                                 + 1.01282 w_k1_KO_s017 + 1.11339 w_k1_MAT_s017
                                 + 0.882028 w_k1_MSFT_s017
                                 + 1.01864 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.915917 w_k1_ADM_s017
                                 + 1.02613 w_k1_AXP_s017
                                 + 1.02771 w_k1_DIS_s017
                                 + 0.975165 w_k1_FDX_s017
                                 + 1.28862 w_k1_IBM_s017
                                 + 0.913892 w_k1_INTC_s017
                                 + 1.16819 w_k1_KO_s017 + 1.99984 w_k1_MAT_s017
                                 + 0.983357 w_k1_MSFT_s017
                                 + 1.00623 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.679996 w_k1_ADM_s017
                                 + 0.521533 w_k1_AXP_s017
                                 + 0.427499 w_k1_DIS_s017
                                 + 0.626751 w_k1_FDX_s017
                                 + 0.679547 w_k1_IBM_s017
                                 + 0.637999 w_k1_INTC_s017
                                 + 0.50399 w_k1_KO_s017
                                 + 0.660817 w_k1_MAT_s017
                                 + 0.652002 w_k1_MSFT_s017
                                 + 0.639597 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 1.18794 w_k1_ADM_s017
                                 + 0.90611 w_k1_AXP_s017
                                 + 0.743747 w_k1_DIS_s017
                                 + 0.944262 w_k1_FDX_s017
                                 + 0.975602 w_k1_IBM_s017
                                 + 0.891233 w_k1_INTC_s017
                                 + 0.900118 w_k1_KO_s017
                                 + 1.38034 w_k1_MAT_s017
                                 + 0.938804 w_k1_MSFT_s017
                                 + 1.01186 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.11364 w_k1_ADM_s017
                                 + 1.1448 w_k1_AXP_s017 + 1.41292 w_k1_DIS_s017
                                 + 1.27051 w_k1_FDX_s017
                                 + 1.06436 w_k1_IBM_s017
                                 + 1.03829 w_k1_INTC_s017
                                 + 1.23256 w_k1_KO_s017 + 1.72104 w_k1_MAT_s017
                                 + 1.27363 w_k1_MSFT_s017
                                 + 1.19165 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 1.00421 w_k1_ADM_s100
                                 + 1.29781 w_k1_AXP_s100
                                 + 0.924113 w_k1_DIS_s100
                                 + 0.966115 w_k1_FDX_s100
                                 + 1.75883 w_k1_IBM_s100
                                 + 0.974599 w_k1_INTC_s100
                                 + 1.16556 w_k1_KO_s100 + 2.00122 w_k1_MAT_s100
                                 + 0.928782 w_k1_MSFT_s100
                                 + 1.07725 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.14282 w_k1_ADM_s100
                                 + 1.38943 w_k1_AXP_s100
                                 + 1.57341 w_k1_DIS_s100
                                 + 1.18008 w_k1_FDX_s100
                                 + 1.43509 w_k1_IBM_s100
                                 + 1.11798 w_k1_INTC_s100
                                 + 1.51591 w_k1_KO_s100 + 1.65807 w_k1_MAT_s100
                                 + 1.31533 w_k1_MSFT_s100
                                 + 1.15867 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.03318 w_k1_ADM_s100
                                 + 1.08873 w_k1_AXP_s100
                                 + 0.915025 w_k1_DIS_s100
                                 + 1.02819 w_k1_FDX_s100
                                 + 0.847473 w_k1_IBM_s100
                                 + 1.05588 w_k1_INTC_s100
                                 + 0.920577 w_k1_KO_s100
                                 + 1.01534 w_k1_MAT_s100
                                 + 0.950274 w_k1_MSFT_s100
                                 + 0.984389 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.922888 w_k1_ADM_s100
                                 + 1.39813 w_k1_AXP_s100
                                 + 0.95645 w_k1_DIS_s100
                                 + 0.982633 w_k1_FDX_s100
                                 + 1.15886 w_k1_IBM_s100
                                 + 1.01102 w_k1_INTC_s100
                                 + 1.05613 w_k1_KO_s100 + 1.44568 w_k1_MAT_s100
                                 + 1.02158 w_k1_MSFT_s100
                                 + 0.904158 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.667147 w_k1_ADM_s100
                                 + 0.877898 w_k1_AXP_s100
                                 + 0.983335 w_k1_DIS_s100
                                 + 0.867673 w_k1_FDX_s100
                                 + 1.10756 w_k1_IBM_s100
                                 + 0.969848 w_k1_INTC_s100
                                 + 1.14054 w_k1_KO_s100 + 1.25819 w_k1_MAT_s100
                                 + 0.967806 w_k1_MSFT_s100
                                 + 0.926219 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.876644 w_k1_ADM_s100
                                 + 0.878064 w_k1_AXP_s100
                                 + 1.0001 w_k1_DIS_s100 + 1.04028 w_k1_FDX_s100
                                 + 0.685401 w_k1_IBM_s100
                                 + 1.17183 w_k1_INTC_s100
                                 + 0.844088 w_k1_KO_s100
                                 + 0.709904 w_k1_MAT_s100
                                 + 1.22305 w_k1_MSFT_s100
                                 + 0.936275 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 1.03416 w_k1_ADM_s100
                                 + 1.37 w_k1_AXP_s100 + 1.32645 w_k1_DIS_s100
                                 + 1.26043 w_k1_FDX_s100
                                 + 1.05977 w_k1_IBM_s100
                                 + 1.29276 w_k1_INTC_s100 + 1.2608 w_k1_KO_s100
                                 + 1.33995 w_k1_MAT_s100
                                 + 1.32661 w_k1_MSFT_s100
                                 + 1.08058 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 0.678321 w_k1_ADM_s100
                                 + 0.747008 w_k1_AXP_s100
                                 + 0.720124 w_k1_DIS_s100
                                 + 0.76908 w_k1_FDX_s100
                                 + 0.746524 w_k1_IBM_s100
                                 + 0.902311 w_k1_INTC_s100
                                 + 0.837057 w_k1_KO_s100
                                 + 0.586853 w_k1_MAT_s100
                                 + 0.922537 w_k1_MSFT_s100
                                 + 0.776292 w_k1_PG_s100 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.728333 w_k1_ADM_s101
                                 + 0.528523 w_k1_AXP_s101
                                 + 0.637353 w_k1_DIS_s101
                                 + 0.751943 w_k1_FDX_s101
                                 + 0.566867 w_k1_IBM_s101
                                 + 0.711877 w_k1_INTC_s101
                                 + 0.628496 w_k1_KO_s101
                                 + 0.507682 w_k1_MAT_s101
                                 + 0.71988 w_k1_MSFT_s101
                                 + 0.672329 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 0.993694 w_k1_ADM_s101
                                 + 0.857323 w_k1_AXP_s101
                                 + 0.823224 w_k1_DIS_s101
                                 + 0.816692 w_k1_FDX_s101
                                 + 1.3604 w_k1_IBM_s101
                                 + 0.714084 w_k1_INTC_s101
                                 + 1.29839 w_k1_KO_s101 + 1.71902 w_k1_MAT_s101
                                 + 0.902412 w_k1_MSFT_s101
                                 + 0.900609 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.867717 w_k1_ADM_s101
                                 + 0.827481 w_k1_AXP_s101
                                 + 0.799382 w_k1_DIS_s101
                                 + 0.895769 w_k1_FDX_s101
                                 + 0.779213 w_k1_IBM_s101
                                 + 0.789355 w_k1_INTC_s101
                                 + 0.781535 w_k1_KO_s101
                                 + 0.743413 w_k1_MAT_s101
                                 + 0.941132 w_k1_MSFT_s101
                                 + 0.836573 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.866509 w_k1_ADM_s101
                                 + 1.2826 w_k1_AXP_s101 + 1.34314 w_k1_DIS_s101
                                 + 1.05763 w_k1_FDX_s101
                                 + 1.59689 w_k1_IBM_s101
                                 + 1.17976 w_k1_INTC_s101
                                 + 1.45871 w_k1_KO_s101 + 1.77234 w_k1_MAT_s101
                                 + 1.31012 w_k1_MSFT_s101 + 1.1869 w_k1_PG_s101
                                 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 1.13787 w_k1_ADM_s101
                                 + 1.2814 w_k1_AXP_s101 + 1.20625 w_k1_DIS_s101
                                 + 1.31473 w_k1_FDX_s101
                                 + 0.863354 w_k1_IBM_s101
                                 + 1.29339 w_k1_INTC_s101
                                 + 1.06666 w_k1_KO_s101 + 1.24312 w_k1_MAT_s101
                                 + 1.25161 w_k1_MSFT_s101
                                 + 0.992147 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.42693 w_k1_ADM_s101
                                 + 1.28729 w_k1_AXP_s101
                                 + 1.71323 w_k1_DIS_s101
                                 + 1.41271 w_k1_FDX_s101
                                 + 1.18116 w_k1_IBM_s101
                                 + 1.37105 w_k1_INTC_s101
                                 + 1.30684 w_k1_KO_s101 + 1.25062 w_k1_MAT_s101
                                 + 1.40629 w_k1_MSFT_s101
                                 + 1.18883 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.14406 w_k1_ADM_s101
                                 + 1.60191 w_k1_AXP_s101
                                 + 1.35709 w_k1_DIS_s101
                                 + 1.28917 w_k1_FDX_s101
                                 + 1.44925 w_k1_IBM_s101
                                 + 1.22402 w_k1_INTC_s101
                                 + 1.30006 w_k1_KO_s101 + 1.45582 w_k1_MAT_s101
                                 + 1.17433 w_k1_MSFT_s101
                                 + 1.16656 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.884423 w_k1_ADM_s101
                                 + 0.850642 w_k1_AXP_s101
                                 + 0.910044 w_k1_DIS_s101
                                 + 0.843032 w_k1_FDX_s101
                                 + 1.04491 w_k1_IBM_s101
                                 + 1.05746 w_k1_INTC_s101
                                 + 0.940108 w_k1_KO_s101
                                 + 0.709069 w_k1_MAT_s101
                                 + 0.955262 w_k1_MSFT_s101
                                 + 0.92858 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.32422 w_k1_ADM_s102
                                 + 1.75286 w_k1_AXP_s102
                                 + 2.10961 w_k1_DIS_s102
                                 + 1.58185 w_k1_FDX_s102
                                 + 1.30283 w_k1_IBM_s102
                                 + 1.37989 w_k1_INTC_s102
                                 + 1.74801 w_k1_KO_s102 + 1.63258 w_k1_MAT_s102
                                 + 1.44513 w_k1_MSFT_s102
                                 + 1.45262 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.951445 w_k1_ADM_s102
                                 + 0.909292 w_k1_AXP_s102
                                 + 1.19868 w_k1_DIS_s102
                                 + 1.02923 w_k1_FDX_s102
                                 + 0.85622 w_k1_IBM_s102
                                 + 0.927475 w_k1_INTC_s102
                                 + 1.08897 w_k1_KO_s102
                                 + 0.722661 w_k1_MAT_s102
                                 + 1.16399 w_k1_MSFT_s102 + 1.0145 w_k1_PG_s102
                                 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 0.863448 w_k1_ADM_s102
                                 + 0.882859 w_k1_AXP_s102
                                 + 0.895555 w_k1_DIS_s102
                                 + 0.866028 w_k1_FDX_s102
                                 + 1.21631 w_k1_IBM_s102
                                 + 0.940573 w_k1_INTC_s102
                                 + 1.04366 w_k1_KO_s102 + 1.00115 w_k1_MAT_s102
                                 + 1.0011 w_k1_MSFT_s102
                                 + 0.938829 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.738433 w_k1_ADM_s102
                                 + 0.654211 w_k1_AXP_s102
                                 + 0.716026 w_k1_DIS_s102
                                 + 0.838706 w_k1_FDX_s102
                                 + 0.649331 w_k1_IBM_s102
                                 + 0.909909 w_k1_INTC_s102
                                 + 0.798944 w_k1_KO_s102
                                 + 0.630732 w_k1_MAT_s102
                                 + 0.886839 w_k1_MSFT_s102
                                 + 0.817672 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.21828 w_k1_ADM_s102
                                 + 1.38163 w_k1_AXP_s102
                                 + 1.14322 w_k1_DIS_s102
                                 + 1.16248 w_k1_FDX_s102
                                 + 1.14397 w_k1_IBM_s102
                                 + 0.964821 w_k1_INTC_s102
                                 + 1.31227 w_k1_KO_s102 + 1.33983 w_k1_MAT_s102
                                 + 0.991688 w_k1_MSFT_s102
                                 + 1.08658 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.14498 w_k1_ADM_s102
                                 + 1.33112 w_k1_AXP_s102
                                 + 1.53218 w_k1_DIS_s102
                                 + 1.37105 w_k1_FDX_s102
                                 + 1.00714 w_k1_IBM_s102
                                 + 1.31605 w_k1_INTC_s102
                                 + 1.29791 w_k1_KO_s102
                                 + 0.805962 w_k1_MAT_s102
                                 + 1.30908 w_k1_MSFT_s102
                                 + 1.09294 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.959274 w_k1_ADM_s102
                                 + 0.900626 w_k1_AXP_s102
                                 + 0.79676 w_k1_DIS_s102
                                 + 0.881713 w_k1_FDX_s102
                                 + 0.901867 w_k1_IBM_s102
                                 + 0.826264 w_k1_INTC_s102
                                 + 0.896353 w_k1_KO_s102
                                 + 0.892301 w_k1_MAT_s102
                                 + 0.946171 w_k1_MSFT_s102
                                 + 0.914008 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.06745 w_k1_ADM_s102
                                 + 1.34532 w_k1_AXP_s102 + 1.6765 w_k1_DIS_s102
                                 + 1.17212 w_k1_FDX_s102
                                 + 1.76947 w_k1_IBM_s102
                                 + 1.36663 w_k1_INTC_s102 + 1.5478 w_k1_KO_s102
                                 + 1.77638 w_k1_MAT_s102
                                 + 1.33912 w_k1_MSFT_s102
                                 + 1.18976 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 0.969357 w_k1_ADM_s103
                                 + 0.923806 w_k1_AXP_s103
                                 + 0.996846 w_k1_DIS_s103
                                 + 1.00131 w_k1_FDX_s103
                                 + 1.31617 w_k1_IBM_s103
                                 + 1.12859 w_k1_INTC_s103
                                 + 1.23192 w_k1_KO_s103
                                 + 0.963185 w_k1_MAT_s103
                                 + 1.06718 w_k1_MSFT_s103
                                 + 1.05468 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.970325 w_k1_ADM_s103
                                 + 0.808542 w_k1_AXP_s103
                                 + 0.92554 w_k1_DIS_s103
                                 + 1.05754 w_k1_FDX_s103
                                 + 0.688544 w_k1_IBM_s103
                                 + 0.90965 w_k1_INTC_s103
                                 + 0.915795 w_k1_KO_s103
                                 + 0.659986 w_k1_MAT_s103
                                 + 1.13326 w_k1_MSFT_s103
                                 + 0.969889 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.19594 w_k1_ADM_s103
                                 + 0.882588 w_k1_AXP_s103
                                 + 1.05964 w_k1_DIS_s103
                                 + 1.05744 w_k1_FDX_s103
                                 + 0.906301 w_k1_IBM_s103
                                 + 1.01684 w_k1_INTC_s103
                                 + 1.04001 w_k1_KO_s103
                                 + 0.741761 w_k1_MAT_s103
                                 + 1.12571 w_k1_MSFT_s103
                                 + 1.09247 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 0.84414 w_k1_ADM_s103
                                 + 0.600657 w_k1_AXP_s103
                                 + 0.698118 w_k1_DIS_s103
                                 + 0.751807 w_k1_FDX_s103
                                 + 0.847602 w_k1_IBM_s103
                                 + 0.849555 w_k1_INTC_s103
                                 + 0.968897 w_k1_KO_s103
                                 + 0.519868 w_k1_MAT_s103
                                 + 0.855837 w_k1_MSFT_s103
                                 + 0.944518 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.13897 w_k1_ADM_s103
                                 + 1.26095 w_k1_AXP_s103
                                 + 1.43656 w_k1_DIS_s103
                                 + 1.33257 w_k1_FDX_s103
                                 + 1.14838 w_k1_IBM_s103
                                 + 1.27734 w_k1_INTC_s103
                                 + 1.34998 w_k1_KO_s103 + 1.12342 w_k1_MAT_s103
                                 + 1.42703 w_k1_MSFT_s103
                                 + 1.30703 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.952286 w_k1_ADM_s103
                                 + 0.919258 w_k1_AXP_s103
                                 + 0.731663 w_k1_DIS_s103
                                 + 0.868572 w_k1_FDX_s103
                                 + 0.92136 w_k1_IBM_s103
                                 + 0.953434 w_k1_INTC_s103
                                 + 0.915636 w_k1_KO_s103
                                 + 0.886955 w_k1_MAT_s103
                                 + 0.913588 w_k1_MSFT_s103
                                 + 0.932368 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.910187 w_k1_ADM_s103
                                 + 0.71778 w_k1_AXP_s103
                                 + 0.767337 w_k1_DIS_s103
                                 + 0.816401 w_k1_FDX_s103
                                 + 1.21649 w_k1_IBM_s103
                                 + 1.04372 w_k1_INTC_s103
                                 + 1.03303 w_k1_KO_s103
                                 + 0.917699 w_k1_MAT_s103
                                 + 0.917542 w_k1_MSFT_s103
                                 + 0.97376 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.23272 w_k1_ADM_s103
                                 + 1.39652 w_k1_AXP_s103
                                 + 1.41215 w_k1_DIS_s103
                                 + 1.09169 w_k1_FDX_s103
                                 + 1.63847 w_k1_IBM_s103
                                 + 1.26289 w_k1_INTC_s103
                                 + 1.42005 w_k1_KO_s103 + 1.14897 w_k1_MAT_s103
                                 + 1.17041 w_k1_MSFT_s103 + 1.2848 w_k1_PG_s103
                                 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.18274 w_k1_ADM_s104
                                 + 1.46369 w_k1_AXP_s104
                                 + 1.43487 w_k1_DIS_s104
                                 + 1.23762 w_k1_FDX_s104
                                 + 1.37582 w_k1_IBM_s104
                                 + 1.16018 w_k1_INTC_s104
                                 + 1.45064 w_k1_KO_s104 + 1.53659 w_k1_MAT_s104
                                 + 1.16191 w_k1_MSFT_s104
                                 + 1.13834 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 0.896657 w_k1_ADM_s104
                                 + 0.861786 w_k1_AXP_s104
                                 + 1.10336 w_k1_DIS_s104
                                 + 1.04958 w_k1_FDX_s104
                                 + 0.731727 w_k1_IBM_s104
                                 + 0.98183 w_k1_INTC_s104
                                 + 0.960226 w_k1_KO_s104
                                 + 0.593318 w_k1_MAT_s104
                                 + 0.967054 w_k1_MSFT_s104
                                 + 0.920227 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.693082 w_k1_ADM_s104
                                 + 0.733645 w_k1_AXP_s104
                                 + 0.656695 w_k1_DIS_s104
                                 + 0.751827 w_k1_FDX_s104
                                 + 0.793501 w_k1_IBM_s104
                                 + 0.938277 w_k1_INTC_s104
                                 + 0.764987 w_k1_KO_s104
                                 + 0.614451 w_k1_MAT_s104
                                 + 0.771035 w_k1_MSFT_s104
                                 + 0.87032 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.840648 w_k1_ADM_s104
                                 + 0.812219 w_k1_AXP_s104
                                 + 0.969969 w_k1_DIS_s104
                                 + 0.870069 w_k1_FDX_s104
                                 + 0.872678 w_k1_IBM_s104
                                 + 0.94121 w_k1_INTC_s104
                                 + 1.02188 w_k1_KO_s104
                                 + 0.813884 w_k1_MAT_s104
                                 + 0.878767 w_k1_MSFT_s104
                                 + 1.06073 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.789616 w_k1_ADM_s104
                                 + 0.701026 w_k1_AXP_s104
                                 + 0.871045 w_k1_DIS_s104
                                 + 0.732325 w_k1_FDX_s104
                                 + 1.2139 w_k1_IBM_s104
                                 + 0.958768 w_k1_INTC_s104
                                 + 1.05595 w_k1_KO_s104 + 1.04217 w_k1_MAT_s104
                                 + 0.833758 w_k1_MSFT_s104
                                 + 0.879544 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.23957 w_k1_ADM_s104
                                 + 1.06537 w_k1_AXP_s104
                                 + 1.40659 w_k1_DIS_s104
                                 + 1.24781 w_k1_FDX_s104
                                 + 0.816412 w_k1_IBM_s104
                                 + 1.08393 w_k1_INTC_s104
                                 + 1.08389 w_k1_KO_s104
                                 + 0.691143 w_k1_MAT_s104
                                 + 1.11354 w_k1_MSFT_s104
                                 + 1.29283 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.893456 w_k1_ADM_s104
                                 + 1.15206 w_k1_AXP_s104
                                 + 0.816134 w_k1_DIS_s104
                                 + 0.852975 w_k1_FDX_s104
                                 + 1.17332 w_k1_IBM_s104
                                 + 1.06977 w_k1_INTC_s104
                                 + 0.976053 w_k1_KO_s104
                                 + 1.33913 w_k1_MAT_s104
                                 + 0.865213 w_k1_MSFT_s104
                                 + 1.07672 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.16751 w_k1_ADM_s104
                                 + 1.15724 w_k1_AXP_s104
                                 + 1.13471 w_k1_DIS_s104 + 1.0368 w_k1_FDX_s104
                                 + 1.09705 w_k1_IBM_s104
                                 + 1.05497 w_k1_INTC_s104
                                 + 1.20954 w_k1_KO_s104 + 1.09517 w_k1_MAT_s104
                                 + 1.02753 w_k1_MSFT_s104
                                 + 1.21503 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.08357 w_k1_ADM_s105
                                 + 1.02679 w_k1_AXP_s105
                                 + 1.14449 w_k1_DIS_s105
                                 + 1.02226 w_k1_FDX_s105
                                 + 1.09245 w_k1_IBM_s105
                                 + 1.04222 w_k1_INTC_s105
                                 + 1.20602 w_k1_KO_s105 + 1.22195 w_k1_MAT_s105
                                 + 1.01279 w_k1_MSFT_s105
                                 + 1.15982 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.28109 w_k1_ADM_s105
                                 + 1.20661 w_k1_AXP_s105
                                 + 0.882309 w_k1_DIS_s105
                                 + 1.08392 w_k1_FDX_s105
                                 + 0.767317 w_k1_IBM_s105
                                 + 0.915895 w_k1_INTC_s105
                                 + 0.877646 w_k1_KO_s105
                                 + 0.944255 w_k1_MAT_s105
                                 + 1.0161 w_k1_MSFT_s105 + 1.0522 w_k1_PG_s105
                                 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.770414 w_k1_ADM_s105
                                 + 0.766049 w_k1_AXP_s105
                                 + 0.547118 w_k1_DIS_s105
                                 + 0.713534 w_k1_FDX_s105
                                 + 0.694917 w_k1_IBM_s105
                                 + 0.615775 w_k1_INTC_s105
                                 + 0.657981 w_k1_KO_s105
                                 + 0.703127 w_k1_MAT_s105
                                 + 0.680066 w_k1_MSFT_s105
                                 + 0.787886 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.31137 w_k1_ADM_s105
                                 + 1.47093 w_k1_AXP_s105 + 1.4593 w_k1_DIS_s105
                                 + 1.33966 w_k1_FDX_s105
                                 + 1.06706 w_k1_IBM_s105
                                 + 1.0056 w_k1_INTC_s105 + 1.14371 w_k1_KO_s105
                                 + 1.44798 w_k1_MAT_s105
                                 + 1.1299 w_k1_MSFT_s105 + 1.1226 w_k1_PG_s105
                                 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.967255 w_k1_ADM_s105
                                 + 0.855201 w_k1_AXP_s105
                                 + 0.735094 w_k1_DIS_s105
                                 + 0.841352 w_k1_FDX_s105
                                 + 0.985618 w_k1_IBM_s105
                                 + 0.939089 w_k1_INTC_s105
                                 + 0.85007 w_k1_KO_s105
                                 + 0.834119 w_k1_MAT_s105
                                 + 0.80967 w_k1_MSFT_s105
                                 + 0.84065 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.93026 w_k1_ADM_s105
                                 + 0.869602 w_k1_AXP_s105
                                 + 0.971274 w_k1_DIS_s105
                                 + 1.02138 w_k1_FDX_s105
                                 + 0.812646 w_k1_IBM_s105
                                 + 0.842512 w_k1_INTC_s105
                                 + 0.923014 w_k1_KO_s105
                                 + 0.663376 w_k1_MAT_s105
                                 + 0.919122 w_k1_MSFT_s105
                                 + 0.91581 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 1.36911 w_k1_ADM_s105
                                 + 1.22281 w_k1_AXP_s105
                                 + 1.31779 w_k1_DIS_s105
                                 + 1.16593 w_k1_FDX_s105
                                 + 1.52218 w_k1_IBM_s105
                                 + 0.922803 w_k1_INTC_s105
                                 + 1.35389 w_k1_KO_s105 + 1.67521 w_k1_MAT_s105
                                 + 0.999937 w_k1_MSFT_s105
                                 + 1.19641 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.0508 w_k1_ADM_s105
                                 + 0.905906 w_k1_AXP_s105
                                 + 0.936823 w_k1_DIS_s105
                                 + 0.883737 w_k1_FDX_s105
                                 + 0.893005 w_k1_IBM_s105
                                 + 1.06768 w_k1_INTC_s105
                                 + 0.942545 w_k1_KO_s105
                                 + 0.747272 w_k1_MAT_s105
                                 + 0.978278 w_k1_MSFT_s105
                                 + 1.06001 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.00531 w_k1_ADM_s106
                                 + 0.868772 w_k1_AXP_s106
                                 + 1.07649 w_k1_DIS_s106
                                 + 1.01465 w_k1_FDX_s106
                                 + 0.969338 w_k1_IBM_s106
                                 + 0.956875 w_k1_INTC_s106
                                 + 1.04941 w_k1_KO_s106
                                 + 0.901722 w_k1_MAT_s106
                                 + 1.0826 w_k1_MSFT_s106 + 1.01655 w_k1_PG_s106
                                 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.34859 w_k1_ADM_s106
                                 + 1.5971 w_k1_AXP_s106 + 1.57669 w_k1_DIS_s106
                                 + 1.24309 w_k1_FDX_s106
                                 + 1.87846 w_k1_IBM_s106
                                 + 1.18639 w_k1_INTC_s106 + 1.2939 w_k1_KO_s106
                                 + 1.6306 w_k1_MAT_s106
                                 + 1.15568 w_k1_MSFT_s106
                                 + 1.28474 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.737889 w_k1_ADM_s106
                                 + 0.770755 w_k1_AXP_s106
                                 + 0.694476 w_k1_DIS_s106
                                 + 0.781315 w_k1_FDX_s106
                                 + 0.714049 w_k1_IBM_s106
                                 + 0.772554 w_k1_INTC_s106
                                 + 0.676593 w_k1_KO_s106
                                 + 0.73315 w_k1_MAT_s106
                                 + 0.807046 w_k1_MSFT_s106
                                 + 0.866494 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.24117 w_k1_ADM_s106
                                 + 1.51611 w_k1_AXP_s106
                                 + 1.84849 w_k1_DIS_s106
                                 + 1.53774 w_k1_FDX_s106
                                 + 1.07121 w_k1_IBM_s106
                                 + 1.27726 w_k1_INTC_s106
                                 + 1.23764 w_k1_KO_s106 + 1.13061 w_k1_MAT_s106
                                 + 1.60993 w_k1_MSFT_s106
                                 + 1.41479 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.24068 w_k1_ADM_s106
                                 + 1.15656 w_k1_AXP_s106
                                 + 1.18194 w_k1_DIS_s106
                                 + 1.03204 w_k1_FDX_s106
                                 + 1.30718 w_k1_IBM_s106
                                 + 1.03018 w_k1_INTC_s106
                                 + 1.22994 w_k1_KO_s106
                                 + 0.985318 w_k1_MAT_s106
                                 + 1.06215 w_k1_MSFT_s106
                                 + 1.13662 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 1.2876 w_k1_ADM_s106
                                 + 1.17677 w_k1_AXP_s106
                                 + 1.48669 w_k1_DIS_s106
                                 + 1.29012 w_k1_FDX_s106
                                 + 0.963081 w_k1_IBM_s106
                                 + 1.21559 w_k1_INTC_s106
                                 + 1.07802 w_k1_KO_s106
                                 + 0.959762 w_k1_MAT_s106
                                 + 1.32649 w_k1_MSFT_s106 + 1.2885 w_k1_PG_s106
                                 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 0.855961 w_k1_ADM_s106
                                 + 0.981969 w_k1_AXP_s106
                                 + 0.804645 w_k1_DIS_s106
                                 + 0.787055 w_k1_FDX_s106
                                 + 1.19716 w_k1_IBM_s106
                                 + 0.740771 w_k1_INTC_s106
                                 + 0.937812 w_k1_KO_s106
                                 + 1.01886 w_k1_MAT_s106
                                 + 0.763045 w_k1_MSFT_s106
                                 + 0.958158 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.43955 w_k1_ADM_s106
                                 + 1.29813 w_k1_AXP_s106
                                 + 0.987666 w_k1_DIS_s106
                                 + 1.26112 w_k1_FDX_s106
                                 + 0.704572 w_k1_IBM_s106
                                 + 1.03908 w_k1_INTC_s106
                                 + 0.846711 w_k1_KO_s106
                                 + 0.774066 w_k1_MAT_s106
                                 + 1.06056 w_k1_MSFT_s106
                                 + 1.03101 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.849034 w_k1_ADM_s107
                                 + 0.86295 w_k1_AXP_s107
                                 + 0.858431 w_k1_DIS_s107
                                 + 0.810745 w_k1_FDX_s107
                                 + 1.16497 w_k1_IBM_s107
                                 + 0.814783 w_k1_INTC_s107
                                 + 1.01312 w_k1_KO_s107 + 1.3552 w_k1_MAT_s107
                                 + 0.772502 w_k1_MSFT_s107
                                 + 0.871713 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 1.44059 w_k1_ADM_s107
                                 + 1.14447 w_k1_AXP_s107
                                 + 1.65929 w_k1_DIS_s107 + 1.3329 w_k1_FDX_s107
                                 + 1.05158 w_k1_IBM_s107
                                 + 0.904907 w_k1_INTC_s107
                                 + 1.22528 w_k1_KO_s107 + 1.17342 w_k1_MAT_s107
                                 + 1.11553 w_k1_MSFT_s107
                                 + 1.22018 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.07795 w_k1_ADM_s107
                                 + 1.37052 w_k1_AXP_s107
                                 + 0.875352 w_k1_DIS_s107
                                 + 1.00343 w_k1_FDX_s107
                                 + 1.22362 w_k1_IBM_s107
                                 + 0.895476 w_k1_INTC_s107
                                 + 0.988862 w_k1_KO_s107
                                 + 1.91778 w_k1_MAT_s107
                                 + 0.943456 w_k1_MSFT_s107
                                 + 1.07281 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 1.18987 w_k1_ADM_s107
                                 + 1.01574 w_k1_AXP_s107
                                 + 0.919587 w_k1_DIS_s107
                                 + 1.01692 w_k1_FDX_s107
                                 + 0.791334 w_k1_IBM_s107
                                 + 0.763717 w_k1_INTC_s107
                                 + 0.809525 w_k1_KO_s107
                                 + 1.34282 w_k1_MAT_s107
                                 + 0.868283 w_k1_MSFT_s107
                                 + 0.914139 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 0.908726 w_k1_ADM_s107
                                 + 0.681296 w_k1_AXP_s107
                                 + 0.635863 w_k1_DIS_s107
                                 + 0.731378 w_k1_FDX_s107
                                 + 0.787296 w_k1_IBM_s107
                                 + 0.672818 w_k1_INTC_s107
                                 + 0.805777 w_k1_KO_s107
                                 + 1.09779 w_k1_MAT_s107
                                 + 0.699025 w_k1_MSFT_s107
                                 + 0.815171 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.00095 w_k1_ADM_s107
                                 + 1.00506 w_k1_AXP_s107
                                 + 1.15226 w_k1_DIS_s107
                                 + 1.02535 w_k1_FDX_s107
                                 + 1.03691 w_k1_IBM_s107
                                 + 0.994132 w_k1_INTC_s107
                                 + 1.29476 w_k1_KO_s107 + 1.68038 w_k1_MAT_s107
                                 + 1.05986 w_k1_MSFT_s107
                                 + 1.09473 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.8447 w_k1_ADM_s107
                                 + 1.03553 w_k1_AXP_s107
                                 + 1.39472 w_k1_DIS_s107
                                 + 1.01761 w_k1_FDX_s107
                                 + 1.54395 w_k1_IBM_s107
                                 + 0.995224 w_k1_INTC_s107
                                 + 1.42688 w_k1_KO_s107 + 1.93135 w_k1_MAT_s107
                                 + 1.12657 w_k1_MSFT_s107
                                 + 1.10926 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.812805 w_k1_ADM_s107
                                 + 0.62673 w_k1_AXP_s107
                                 + 0.595597 w_k1_DIS_s107
                                 + 0.772977 w_k1_FDX_s107
                                 + 0.580128 w_k1_IBM_s107
                                 + 0.570367 w_k1_INTC_s107
                                 + 0.713938 w_k1_KO_s107
                                 + 1.08274 w_k1_MAT_s107
                                 + 0.693194 w_k1_MSFT_s107
                                 + 0.830633 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.26357 w_k1_ADM_s110
                                 + 1.0732 w_k1_AXP_s110
                                 + 0.976999 w_k1_DIS_s110
                                 + 1.01429 w_k1_FDX_s110
                                 + 1.39746 w_k1_IBM_s110
                                 + 1.01137 w_k1_INTC_s110 + 1.1895 w_k1_KO_s110
                                 + 1.39717 w_k1_MAT_s110
                                 + 0.96518 w_k1_MSFT_s110
                                 + 1.07998 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.2298 w_k1_ADM_s110
                                 + 1.4331 w_k1_AXP_s110 + 1.59721 w_k1_DIS_s110
                                 + 1.3903 w_k1_FDX_s110 + 1.0979 w_k1_IBM_s110
                                 + 1.35601 w_k1_INTC_s110
                                 + 1.45967 w_k1_KO_s110 + 1.21281 w_k1_MAT_s110
                                 + 1.20132 w_k1_MSFT_s110
                                 + 1.31332 w_k1_PG_s110 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.826273 w_k1_ADM_s110
                                 + 0.694006 w_k1_AXP_s110
                                 + 0.610397 w_k1_DIS_s110
                                 + 0.716494 w_k1_FDX_s110
                                 + 0.850007 w_k1_IBM_s110
                                 + 0.950361 w_k1_INTC_s110
                                 + 0.838904 w_k1_KO_s110
                                 + 0.70201 w_k1_MAT_s110
                                 + 0.691631 w_k1_MSFT_s110
                                 + 0.80801 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.01817 w_k1_ADM_s110
                                 + 0.840573 w_k1_AXP_s110
                                 + 0.781613 w_k1_DIS_s110
                                 + 0.873062 w_k1_FDX_s110
                                 + 0.922635 w_k1_IBM_s110
                                 + 1.05798 w_k1_INTC_s110
                                 + 0.925411 w_k1_KO_s110
                                 + 0.736774 w_k1_MAT_s110
                                 + 0.874784 w_k1_MSFT_s110
                                 + 0.808673 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.94772 w_k1_ADM_s110
                                 + 1.24503 w_k1_AXP_s110
                                 + 1.05692 w_k1_DIS_s110 + 1.1747 w_k1_FDX_s110
                                 + 1.02195 w_k1_IBM_s110
                                 + 1.22107 w_k1_INTC_s110
                                 + 1.05263 w_k1_KO_s110 + 1.10096 w_k1_MAT_s110
                                 + 0.973181 w_k1_MSFT_s110
                                 + 1.11358 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.03634 w_k1_ADM_s110
                                 + 0.820058 w_k1_AXP_s110
                                 + 0.946486 w_k1_DIS_s110
                                 + 0.974919 w_k1_FDX_s110
                                 + 0.689648 w_k1_IBM_s110
                                 + 1.22951 w_k1_INTC_s110
                                 + 0.986644 w_k1_KO_s110
                                 + 0.809785 w_k1_MAT_s110
                                 + 0.975055 w_k1_MSFT_s110
                                 + 0.873823 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 0.877418 w_k1_ADM_s110
                                 + 1.03175 w_k1_AXP_s110
                                 + 0.652601 w_k1_DIS_s110
                                 + 0.902183 w_k1_FDX_s110
                                 + 0.691699 w_k1_IBM_s110
                                 + 1.03194 w_k1_INTC_s110
                                 + 0.709871 w_k1_KO_s110
                                 + 0.775131 w_k1_MAT_s110
                                 + 0.754423 w_k1_MSFT_s110
                                 + 0.77944 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 1.0905 w_k1_ADM_s110
                                 + 1.17969 w_k1_AXP_s110
                                 + 1.08919 w_k1_DIS_s110
                                 + 1.22979 w_k1_FDX_s110
                                 + 0.634443 w_k1_IBM_s110
                                 + 1.10615 w_k1_INTC_s110
                                 + 0.992317 w_k1_KO_s110
                                 + 0.858823 w_k1_MAT_s110
                                 + 1.14309 w_k1_MSFT_s110
                                 + 0.924128 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 1.1771 w_k1_ADM_s111
                                 + 1.44305 w_k1_AXP_s111
                                 + 1.11431 w_k1_DIS_s111
                                 + 1.26006 w_k1_FDX_s111
                                 + 0.985985 w_k1_IBM_s111
                                 + 1.23935 w_k1_INTC_s111
                                 + 1.05169 w_k1_KO_s111 + 1.55887 w_k1_MAT_s111
                                 + 1.14342 w_k1_MSFT_s111
                                 + 1.08236 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.16091 w_k1_ADM_s111
                                 + 0.941532 w_k1_AXP_s111
                                 + 1.22137 w_k1_DIS_s111
                                 + 1.07963 w_k1_FDX_s111
                                 + 0.854775 w_k1_IBM_s111
                                 + 1.04835 w_k1_INTC_s111
                                 + 1.09945 w_k1_KO_s111
                                 + 0.904659 w_k1_MAT_s111
                                 + 1.06647 w_k1_MSFT_s111
                                 + 1.02171 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.589256 w_k1_ADM_s111
                                 + 0.604869 w_k1_AXP_s111
                                 + 0.618902 w_k1_DIS_s111
                                 + 0.702598 w_k1_FDX_s111
                                 + 0.703875 w_k1_IBM_s111
                                 + 0.654939 w_k1_INTC_s111
                                 + 0.762379 w_k1_KO_s111
                                 + 1.04449 w_k1_MAT_s111
                                 + 0.724342 w_k1_MSFT_s111
                                 + 0.766257 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.883243 w_k1_ADM_s111
                                 + 0.957951 w_k1_AXP_s111
                                 + 0.928146 w_k1_DIS_s111
                                 + 0.812204 w_k1_FDX_s111
                                 + 1.30747 w_k1_IBM_s111
                                 + 0.687296 w_k1_INTC_s111
                                 + 1.08411 w_k1_KO_s111 + 1.61846 w_k1_MAT_s111
                                 + 0.804828 w_k1_MSFT_s111
                                 + 1.06241 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.19392 w_k1_ADM_s111
                                 + 1.16121 w_k1_AXP_s111
                                 + 0.999063 w_k1_DIS_s111
                                 + 1.07861 w_k1_FDX_s111
                                 + 0.944196 w_k1_IBM_s111
                                 + 0.880983 w_k1_INTC_s111
                                 + 1.05181 w_k1_KO_s111 + 1.49198 w_k1_MAT_s111
                                 + 0.993786 w_k1_MSFT_s111
                                 + 1.09651 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.783846 w_k1_ADM_s111
                                 + 0.972688 w_k1_AXP_s111
                                 + 0.804451 w_k1_DIS_s111
                                 + 0.873587 w_k1_FDX_s111
                                 + 0.777227 w_k1_IBM_s111
                                 + 0.907392 w_k1_INTC_s111
                                 + 0.934977 w_k1_KO_s111
                                 + 1.02255 w_k1_MAT_s111
                                 + 0.819863 w_k1_MSFT_s111
                                 + 0.983163 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 1.06759 w_k1_ADM_s111
                                 + 1.36575 w_k1_AXP_s111
                                 + 1.76332 w_k1_DIS_s111
                                 + 1.28189 w_k1_FDX_s111
                                 + 1.32044 w_k1_IBM_s111
                                 + 1.25527 w_k1_INTC_s111
                                 + 1.45319 w_k1_KO_s111 + 1.87229 w_k1_MAT_s111
                                 + 1.27088 w_k1_MSFT_s111
                                 + 1.31453 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.858811 w_k1_ADM_s111
                                 + 1.03746 w_k1_AXP_s111
                                 + 0.982708 w_k1_DIS_s111
                                 + 0.860053 w_k1_FDX_s111
                                 + 1.18554 w_k1_IBM_s111
                                 + 0.904354 w_k1_INTC_s111
                                 + 1.11987 w_k1_KO_s111 + 1.3132 w_k1_MAT_s111
                                 + 0.949479 w_k1_MSFT_s111
                                 + 1.01975 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.833227 w_k1_ADM_s112
                                 + 0.801733 w_k1_AXP_s112
                                 + 0.899054 w_k1_DIS_s112
                                 + 0.863388 w_k1_FDX_s112
                                 + 1.14306 w_k1_IBM_s112
                                 + 0.956202 w_k1_INTC_s112
                                 + 0.848022 w_k1_KO_s112
                                 + 0.871367 w_k1_MAT_s112
                                 + 0.846695 w_k1_MSFT_s112
                                 + 0.999118 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 0.962315 w_k1_ADM_s112
                                 + 0.93868 w_k1_AXP_s112
                                 + 0.889555 w_k1_DIS_s112
                                 + 1.13325 w_k1_FDX_s112
                                 + 0.711516 w_k1_IBM_s112
                                 + 1.07149 w_k1_INTC_s112
                                 + 0.816728 w_k1_KO_s112
                                 + 0.940592 w_k1_MAT_s112
                                 + 0.853988 w_k1_MSFT_s112
                                 + 1.01375 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.34699 w_k1_ADM_s112
                                 + 1.24477 w_k1_AXP_s112
                                 + 1.38263 w_k1_DIS_s112
                                 + 1.40328 w_k1_FDX_s112
                                 + 1.05319 w_k1_IBM_s112
                                 + 1.4896 w_k1_INTC_s112 + 1.14035 w_k1_KO_s112
                                 + 0.817698 w_k1_MAT_s112
                                 + 1.19097 w_k1_MSFT_s112
                                 + 1.14661 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.850078 w_k1_ADM_s112
                                 + 0.950501 w_k1_AXP_s112
                                 + 0.719823 w_k1_DIS_s112
                                 + 0.999389 w_k1_FDX_s112
                                 + 0.903145 w_k1_IBM_s112
                                 + 0.962487 w_k1_INTC_s112
                                 + 0.732056 w_k1_KO_s112
                                 + 1.10994 w_k1_MAT_s112
                                 + 0.80072 w_k1_MSFT_s112
                                 + 0.992949 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 1.03074 w_k1_ADM_s112
                                 + 1.0355 w_k1_AXP_s112 + 1.14628 w_k1_DIS_s112
                                 + 1.11103 w_k1_FDX_s112
                                 + 1.54348 w_k1_IBM_s112
                                 + 1.10299 w_k1_INTC_s112
                                 + 1.19152 w_k1_KO_s112 + 1.37649 w_k1_MAT_s112
                                 + 1.05303 w_k1_MSFT_s112
                                 + 1.32415 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.951197 w_k1_ADM_s112
                                 + 0.781913 w_k1_AXP_s112
                                 + 0.861495 w_k1_DIS_s112
                                 + 1.02865 w_k1_FDX_s112
                                 + 0.958619 w_k1_IBM_s112
                                 + 0.889227 w_k1_INTC_s112
                                 + 0.867239 w_k1_KO_s112
                                 + 0.815604 w_k1_MAT_s112
                                 + 0.846309 w_k1_MSFT_s112
                                 + 1.00264 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 0.737133 w_k1_ADM_s112
                                 + 0.515437 w_k1_AXP_s112
                                 + 0.695979 w_k1_DIS_s112
                                 + 0.794926 w_k1_FDX_s112
                                 + 0.836125 w_k1_IBM_s112
                                 + 0.786446 w_k1_INTC_s112
                                 + 0.787059 w_k1_KO_s112
                                 + 0.711082 w_k1_MAT_s112
                                 + 0.725598 w_k1_MSFT_s112
                                 + 0.88285 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.02964 w_k1_ADM_s112
                                 + 0.754406 w_k1_AXP_s112
                                 + 0.906096 w_k1_DIS_s112
                                 + 1.03099 w_k1_FDX_s112
                                 + 0.768436 w_k1_IBM_s112
                                 + 0.984786 w_k1_INTC_s112
                                 + 0.768153 w_k1_KO_s112
                                 + 0.751389 w_k1_MAT_s112
                                 + 1.03534 w_k1_MSFT_s112
                                 + 0.979575 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.844886 w_k1_ADM_s113
                                 + 0.959026 w_k1_AXP_s113
                                 + 0.841263 w_k1_DIS_s113
                                 + 0.970588 w_k1_FDX_s113
                                 + 0.852839 w_k1_IBM_s113
                                 + 0.942034 w_k1_INTC_s113
                                 + 0.806407 w_k1_KO_s113
                                 + 0.838836 w_k1_MAT_s113
                                 + 0.897392 w_k1_MSFT_s113
                                 + 0.934951 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.861187 w_k1_ADM_s113
                                 + 0.704406 w_k1_AXP_s113
                                 + 0.854357 w_k1_DIS_s113
                                 + 0.785575 w_k1_FDX_s113
                                 + 0.886649 w_k1_IBM_s113
                                 + 0.786848 w_k1_INTC_s113
                                 + 0.914539 w_k1_KO_s113
                                 + 0.865748 w_k1_MAT_s113
                                 + 0.865943 w_k1_MSFT_s113
                                 + 0.968504 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.61996 w_k1_ADM_s113
                                 + 1.24503 w_k1_AXP_s113
                                 + 1.55356 w_k1_DIS_s113
                                 + 1.45749 w_k1_FDX_s113
                                 + 1.12548 w_k1_IBM_s113
                                 + 1.07422 w_k1_INTC_s113
                                 + 1.19332 w_k1_KO_s113 + 1.03029 w_k1_MAT_s113
                                 + 1.39005 w_k1_MSFT_s113
                                 + 1.19954 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.11277 w_k1_ADM_s113
                                 + 1.32812 w_k1_AXP_s113
                                 + 1.14829 w_k1_DIS_s113
                                 + 1.03684 w_k1_FDX_s113
                                 + 1.65785 w_k1_IBM_s113
                                 + 0.915436 w_k1_INTC_s113
                                 + 1.44213 w_k1_KO_s113 + 2.05656 w_k1_MAT_s113
                                 + 1.13431 w_k1_MSFT_s113
                                 + 1.12961 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.2978 w_k1_ADM_s113
                                 + 0.936613 w_k1_AXP_s113
                                 + 1.19173 w_k1_DIS_s113
                                 + 1.00481 w_k1_FDX_s113
                                 + 1.13436 w_k1_IBM_s113
                                 + 0.772071 w_k1_INTC_s113
                                 + 1.01633 w_k1_KO_s113
                                 + 0.813507 w_k1_MAT_s113
                                 + 0.967418 w_k1_MSFT_s113
                                 + 0.96911 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.958334 w_k1_ADM_s113
                                 + 0.648371 w_k1_AXP_s113
                                 + 0.69109 w_k1_DIS_s113
                                 + 0.79429 w_k1_FDX_s113
                                 + 0.682144 w_k1_IBM_s113
                                 + 0.885408 w_k1_INTC_s113
                                 + 0.762708 w_k1_KO_s113
                                 + 0.486014 w_k1_MAT_s113
                                 + 0.888925 w_k1_MSFT_s113
                                 + 0.862774 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.15964 w_k1_ADM_s113
                                 + 1.06648 w_k1_AXP_s113
                                 + 1.33586 w_k1_DIS_s113
                                 + 1.19207 w_k1_FDX_s113
                                 + 0.766342 w_k1_IBM_s113
                                 + 0.987563 w_k1_INTC_s113
                                 + 1.00768 w_k1_KO_s113
                                 + 0.817057 w_k1_MAT_s113
                                 + 1.13596 w_k1_MSFT_s113
                                 + 1.10757 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 1.3597 w_k1_ADM_s113
                                 + 1.4628 w_k1_AXP_s113 + 1.48381 w_k1_DIS_s113
                                 + 1.13542 w_k1_FDX_s113
                                 + 1.30708 w_k1_IBM_s113
                                 + 1.02211 w_k1_INTC_s113
                                 + 1.16499 w_k1_KO_s113 + 1.64422 w_k1_MAT_s113
                                 + 1.14981 w_k1_MSFT_s113
                                 + 1.13365 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.15671 w_k1_ADM_s114
                                 + 1.20678 w_k1_AXP_s114
                                 + 1.26614 w_k1_DIS_s114
                                 + 1.18698 w_k1_FDX_s114
                                 + 1.04352 w_k1_IBM_s114
                                 + 1.10958 w_k1_INTC_s114
                                 + 1.08424 w_k1_KO_s114
                                 + 0.957923 w_k1_MAT_s114
                                 + 1.19148 w_k1_MSFT_s114
                                 + 1.06005 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.616748 w_k1_ADM_s114
                                 + 0.591343 w_k1_AXP_s114
                                 + 0.581767 w_k1_DIS_s114
                                 + 0.732324 w_k1_FDX_s114
                                 + 0.598666 w_k1_IBM_s114
                                 + 0.731463 w_k1_INTC_s114
                                 + 0.621384 w_k1_KO_s114
                                 + 0.58156 w_k1_MAT_s114
                                 + 0.778203 w_k1_MSFT_s114
                                 + 0.805592 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.874122 w_k1_ADM_s114
                                 + 0.924008 w_k1_AXP_s114
                                 + 1.21248 w_k1_DIS_s114
                                 + 0.955515 w_k1_FDX_s114
                                 + 1.19592 w_k1_IBM_s114
                                 + 1.12208 w_k1_INTC_s114
                                 + 1.31344 w_k1_KO_s114 + 1.09298 w_k1_MAT_s114
                                 + 1.05926 w_k1_MSFT_s114 + 1.0021 w_k1_PG_s114
                                 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.866118 w_k1_ADM_s114
                                 + 0.92877 w_k1_AXP_s114
                                 + 0.956939 w_k1_DIS_s114
                                 + 0.850174 w_k1_FDX_s114
                                 + 1.54533 w_k1_IBM_s114
                                 + 0.821772 w_k1_INTC_s114
                                 + 1.21343 w_k1_KO_s114 + 1.39527 w_k1_MAT_s114
                                 + 0.950755 w_k1_MSFT_s114
                                 + 0.972772 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.14049 w_k1_ADM_s114
                                 + 1.32943 w_k1_AXP_s114
                                 + 1.55828 w_k1_DIS_s114
                                 + 1.31004 w_k1_FDX_s114
                                 + 1.38883 w_k1_IBM_s114
                                 + 1.17772 w_k1_INTC_s114
                                 + 1.38314 w_k1_KO_s114 + 1.35571 w_k1_MAT_s114
                                 + 1.32496 w_k1_MSFT_s114
                                 + 1.18359 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 0.643123 w_k1_ADM_s114
                                 + 0.611849 w_k1_AXP_s114
                                 + 0.772667 w_k1_DIS_s114
                                 + 0.707602 w_k1_FDX_s114
                                 + 0.88527 w_k1_IBM_s114
                                 + 0.861351 w_k1_INTC_s114
                                 + 0.880011 w_k1_KO_s114
                                 + 0.894073 w_k1_MAT_s114
                                 + 0.844789 w_k1_MSFT_s114
                                 + 0.879314 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.829389 w_k1_ADM_s114
                                 + 0.860225 w_k1_AXP_s114
                                 + 0.838234 w_k1_DIS_s114
                                 + 0.849917 w_k1_FDX_s114
                                 + 0.977885 w_k1_IBM_s114
                                 + 1.04762 w_k1_INTC_s114
                                 + 0.952624 w_k1_KO_s114
                                 + 0.754379 w_k1_MAT_s114
                                 + 1.01286 w_k1_MSFT_s114
                                 + 0.866979 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 1.03071 w_k1_ADM_s114
                                 + 1.1003 w_k1_AXP_s114 + 1.06098 w_k1_DIS_s114
                                 + 1.06592 w_k1_FDX_s114
                                 + 0.767213 w_k1_IBM_s114
                                 + 1.06584 w_k1_INTC_s114
                                 + 1.01707 w_k1_KO_s114
                                 + 0.990696 w_k1_MAT_s114
                                 + 1.09086 w_k1_MSFT_s114
                                 + 1.03032 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 0.807986 w_k1_ADM_s115
                                 + 0.703036 w_k1_AXP_s115
                                 + 0.818746 w_k1_DIS_s115
                                 + 0.908668 w_k1_FDX_s115
                                 + 0.767241 w_k1_IBM_s115
                                 + 0.791519 w_k1_INTC_s115
                                 + 0.847726 w_k1_KO_s115
                                 + 0.64232 w_k1_MAT_s115
                                 + 1.0924 w_k1_MSFT_s115
                                 + 0.921265 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.766156 w_k1_ADM_s115
                                 + 0.611096 w_k1_AXP_s115
                                 + 0.60333 w_k1_DIS_s115
                                 + 0.674309 w_k1_FDX_s115
                                 + 0.679695 w_k1_IBM_s115
                                 + 0.759191 w_k1_INTC_s115
                                 + 0.684219 w_k1_KO_s115
                                 + 0.551263 w_k1_MAT_s115
                                 + 0.833413 w_k1_MSFT_s115
                                 + 0.75658 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.838832 w_k1_ADM_s115
                                 + 0.772862 w_k1_AXP_s115
                                 + 0.787471 w_k1_DIS_s115
                                 + 0.818847 w_k1_FDX_s115
                                 + 0.997839 w_k1_IBM_s115
                                 + 0.796444 w_k1_INTC_s115
                                 + 0.925644 w_k1_KO_s115
                                 + 0.925388 w_k1_MAT_s115
                                 + 0.896771 w_k1_MSFT_s115
                                 + 0.89234 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.836009 w_k1_ADM_s115
                                 + 1.23481 w_k1_AXP_s115
                                 + 1.22459 w_k1_DIS_s115
                                 + 1.24034 w_k1_FDX_s115
                                 + 0.713878 w_k1_IBM_s115
                                 + 1.2364 w_k1_INTC_s115 + 1.00982 w_k1_KO_s115
                                 + 0.692995 w_k1_MAT_s115
                                 + 1.21306 w_k1_MSFT_s115
                                 + 1.11752 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.12559 w_k1_ADM_s115
                                 + 1.34916 w_k1_AXP_s115
                                 + 1.39765 w_k1_DIS_s115
                                 + 1.13992 w_k1_FDX_s115
                                 + 1.72895 w_k1_IBM_s115
                                 + 1.27076 w_k1_INTC_s115
                                 + 1.50047 w_k1_KO_s115 + 2.32678 w_k1_MAT_s115
                                 + 1.27925 w_k1_MSFT_s115
                                 + 1.12418 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 0.812367 w_k1_ADM_s115
                                 + 1.20139 w_k1_AXP_s115
                                 + 0.934945 w_k1_DIS_s115
                                 + 0.971259 w_k1_FDX_s115
                                 + 0.857282 w_k1_IBM_s115
                                 + 0.994777 w_k1_INTC_s115
                                 + 0.919622 w_k1_KO_s115
                                 + 1.17195 w_k1_MAT_s115
                                 + 1.1498 w_k1_MSFT_s115
                                 + 0.916507 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 0.672175 w_k1_ADM_s115
                                 + 0.942053 w_k1_AXP_s115
                                 + 1.01821 w_k1_DIS_s115
                                 + 0.958409 w_k1_FDX_s115
                                 + 1.31251 w_k1_IBM_s115
                                 + 0.972913 w_k1_INTC_s115
                                 + 0.961072 w_k1_KO_s115
                                 + 1.19493 w_k1_MAT_s115
                                 + 1.0097 w_k1_MSFT_s115 + 0.94152 w_k1_PG_s115
                                 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.33014 w_k1_ADM_s115
                                 + 1.48247 w_k1_AXP_s115
                                 + 1.70298 w_k1_DIS_s115
                                 + 1.39193 w_k1_FDX_s115
                                 + 1.12635 w_k1_IBM_s115
                                 + 1.04524 w_k1_INTC_s115
                                 + 1.17332 w_k1_KO_s115 + 1.3627 w_k1_MAT_s115
                                 + 1.29068 w_k1_MSFT_s115
                                 + 1.29157 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.684278 w_k1_ADM_s116
                                 + 0.709282 w_k1_AXP_s116
                                 + 0.752758 w_k1_DIS_s116
                                 + 0.799865 w_k1_FDX_s116
                                 + 0.959473 w_k1_IBM_s116
                                 + 0.683088 w_k1_INTC_s116
                                 + 0.869882 w_k1_KO_s116
                                 + 0.68251 w_k1_MAT_s116
                                 + 0.768106 w_k1_MSFT_s116
                                 + 0.752314 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.87179 w_k1_ADM_s116
                                 + 1.16577 w_k1_AXP_s116
                                 + 0.965431 w_k1_DIS_s116
                                 + 0.983826 w_k1_FDX_s116
                                 + 1.07231 w_k1_IBM_s116
                                 + 0.873514 w_k1_INTC_s116
                                 + 0.909161 w_k1_KO_s116 + 1.0185 w_k1_MAT_s116
                                 + 1.02525 w_k1_MSFT_s116
                                 + 0.882855 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.1407 w_k1_ADM_s116
                                 + 1.39682 w_k1_AXP_s116
                                 + 1.44771 w_k1_DIS_s116
                                 + 1.37657 w_k1_FDX_s116
                                 + 1.07348 w_k1_IBM_s116
                                 + 0.956306 w_k1_INTC_s116
                                 + 1.04939 w_k1_KO_s116
                                 + 0.964924 w_k1_MAT_s116
                                 + 1.14941 w_k1_MSFT_s116
                                 + 1.04664 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.14408 w_k1_ADM_s116
                                 + 0.993895 w_k1_AXP_s116
                                 + 1.15943 w_k1_DIS_s116
                                 + 1.11764 w_k1_FDX_s116
                                 + 0.938035 w_k1_IBM_s116
                                 + 1.01805 w_k1_INTC_s116
                                 + 1.02342 w_k1_KO_s116
                                 + 0.683442 w_k1_MAT_s116
                                 + 1.12831 w_k1_MSFT_s116 + 1.0351 w_k1_PG_s116
                                 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.904479 w_k1_ADM_s116
                                 + 0.908504 w_k1_AXP_s116
                                 + 0.810326 w_k1_DIS_s116
                                 + 0.961156 w_k1_FDX_s116
                                 + 0.766729 w_k1_IBM_s116
                                 + 0.735084 w_k1_INTC_s116
                                 + 0.726936 w_k1_KO_s116
                                 + 0.632137 w_k1_MAT_s116
                                 + 0.878269 w_k1_MSFT_s116
                                 + 0.88419 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.79267 w_k1_ADM_s116
                                 + 0.940977 w_k1_AXP_s116
                                 + 1.02705 w_k1_DIS_s116
                                 + 0.860405 w_k1_FDX_s116
                                 + 1.49263 w_k1_IBM_s116
                                 + 0.965906 w_k1_INTC_s116
                                 + 1.15444 w_k1_KO_s116 + 1.18686 w_k1_MAT_s116
                                 + 1.02633 w_k1_MSFT_s116
                                 + 0.946077 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 0.643728 w_k1_ADM_s116
                                 + 0.608871 w_k1_AXP_s116
                                 + 0.870143 w_k1_DIS_s116
                                 + 0.858653 w_k1_FDX_s116
                                 + 0.620259 w_k1_IBM_s116
                                 + 0.911138 w_k1_INTC_s116
                                 + 0.733487 w_k1_KO_s116
                                 + 0.481792 w_k1_MAT_s116
                                 + 0.989421 w_k1_MSFT_s116
                                 + 0.847407 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.03266 w_k1_ADM_s116
                                 + 1.13152 w_k1_AXP_s116
                                 + 1.59741 w_k1_DIS_s116 + 1.1765 w_k1_FDX_s116
                                 + 1.28372 w_k1_IBM_s116
                                 + 0.903346 w_k1_INTC_s116
                                 + 1.29361 w_k1_KO_s116 + 1.07332 w_k1_MAT_s116
                                 + 1.20282 w_k1_MSFT_s116
                                 + 1.16852 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.73371 w_k1_ADM_s117
                                 + 0.578161 w_k1_AXP_s117
                                 + 0.690336 w_k1_DIS_s117
                                 + 0.705307 w_k1_FDX_s117
                                 + 0.724114 w_k1_IBM_s117
                                 + 0.67386 w_k1_INTC_s117
                                 + 0.716556 w_k1_KO_s117
                                 + 0.737895 w_k1_MAT_s117
                                 + 0.8025 w_k1_MSFT_s117
                                 + 0.883849 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.56826 w_k1_ADM_s117
                                 + 1.37626 w_k1_AXP_s117
                                 + 1.35831 w_k1_DIS_s117
                                 + 1.42125 w_k1_FDX_s117
                                 + 0.916571 w_k1_IBM_s117
                                 + 1.32266 w_k1_INTC_s117
                                 + 1.05966 w_k1_KO_s117 + 1.07248 w_k1_MAT_s117
                                 + 1.31853 w_k1_MSFT_s117
                                 + 1.18831 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.50467 w_k1_ADM_s117
                                 + 1.42366 w_k1_AXP_s117
                                 + 1.69554 w_k1_DIS_s117
                                 + 1.28644 w_k1_FDX_s117
                                 + 1.42275 w_k1_IBM_s117
                                 + 1.05641 w_k1_INTC_s117
                                 + 1.36282 w_k1_KO_s117 + 1.85957 w_k1_MAT_s117
                                 + 1.22318 w_k1_MSFT_s117
                                 + 1.30307 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.04097 w_k1_ADM_s117
                                 + 1.02933 w_k1_AXP_s117
                                 + 1.04042 w_k1_DIS_s117
                                 + 0.972031 w_k1_FDX_s117
                                 + 1.24673 w_k1_IBM_s117
                                 + 0.94488 w_k1_INTC_s117
                                 + 0.896983 w_k1_KO_s117
                                 + 0.931471 w_k1_MAT_s117
                                 + 0.958355 w_k1_MSFT_s117
                                 + 1.04277 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.00999 w_k1_ADM_s117
                                 + 0.858922 w_k1_AXP_s117
                                 + 1.10604 w_k1_DIS_s117
                                 + 1.06947 w_k1_FDX_s117
                                 + 0.836041 w_k1_IBM_s117
                                 + 1.05214 w_k1_INTC_s117
                                 + 0.918714 w_k1_KO_s117
                                 + 0.85649 w_k1_MAT_s117
                                 + 1.17518 w_k1_MSFT_s117
                                 + 1.03756 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 1.06359 w_k1_ADM_s117
                                 + 0.968408 w_k1_AXP_s117
                                 + 0.818158 w_k1_DIS_s117
                                 + 0.955476 w_k1_FDX_s117
                                 + 0.75655 w_k1_IBM_s117
                                 + 1.03015 w_k1_INTC_s117
                                 + 0.899797 w_k1_KO_s117
                                 + 0.904726 w_k1_MAT_s117
                                 + 0.963349 w_k1_MSFT_s117
                                 + 0.965575 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.02016 w_k1_ADM_s117
                                 + 0.68323 w_k1_AXP_s117
                                 + 0.735777 w_k1_DIS_s117
                                 + 0.780389 w_k1_FDX_s117
                                 + 0.83767 w_k1_IBM_s117
                                 + 0.767543 w_k1_INTC_s117
                                 + 0.823896 w_k1_KO_s117
                                 + 0.838747 w_k1_MAT_s117
                                 + 0.847753 w_k1_MSFT_s117
                                 + 0.783677 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.669531 w_k1_ADM_s117
                                 + 0.603827 w_k1_AXP_s117
                                 + 0.692374 w_k1_DIS_s117
                                 + 0.711863 w_k1_FDX_s117
                                 + 1.11498 w_k1_IBM_s117
                                 + 0.828987 w_k1_INTC_s117
                                 + 0.814346 w_k1_KO_s117
                                 + 0.997864 w_k1_MAT_s117
                                 + 0.891986 w_k1_MSFT_s117
                                 + 0.918218 w_k1_PG_s117 >= 1
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
