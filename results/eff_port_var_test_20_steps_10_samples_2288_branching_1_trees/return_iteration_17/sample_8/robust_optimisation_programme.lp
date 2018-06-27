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
 w_k1_ADM_s0_return_constraint:  1.19063 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.973663 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.13352 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.803899 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 1.15969 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 0.830812 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 0.84604 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.48682 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.757994 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.08703 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.0039 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.753669 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 0.895905 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.940041 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.752902 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.19136 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 1.22006 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 1.31855 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.33074 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 0.9436 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.61818 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.931008 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.898218 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 1.0985 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.0722 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 0.763376 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.28477 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 1.11206 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.821603 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.02187 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 0.759005 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 1.21687 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.10113 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.07087 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.31607 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.00723 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.22439 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 1.02211 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.30165 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.953984 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.15556 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.779066 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 1.16478 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 0.893504 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 0.934153 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.44997 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.611454 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.10005 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.39682 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.827331 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 0.644913 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.795833 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.60487 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.17519 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 1.29897 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.34731 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 0.93826 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.812511 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.67319 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.88376 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.858045 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 1.01847 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.14622 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 0.674908 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.71093 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 1.68676 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.896831 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.09379 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.670186 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.97135 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 1.45252 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 0.812603 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.38147 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.06735 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.05704 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.788387 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.38793 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.909732 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.09471 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.916577 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 1.19646 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 0.827903 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 0.940036 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.95273 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 0.48538 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.17023 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 1.53514 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.639665 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 0.732083 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.795924 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.564963 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.32809 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 1.22105 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 1.39188 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 1.01391 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.679682 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.99851 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.748636 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.922204 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 1.21465 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.24063 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 0.694211 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.8663 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 1.73805 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.814105 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.34967 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 0.652628 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.933775 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 1.59677 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 0.85506 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 1.50288 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 0.932336 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.29872 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.76099 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.27436 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.92645 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.07671 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.839993 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 1.178 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 0.879648 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.01704 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.51583 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.680302 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.0745 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.12637 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.710597 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 0.782225 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.9837 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.704067 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.31349 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 1.13071 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.13126 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.04844 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.928682 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.3628 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.852827 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.880897 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 1.16365 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.05137 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 0.839307 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.4642 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 1.26229 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 0.887574 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.14769 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 0.693315 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 1.1382 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 1.21653 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 0.918519 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.25366 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 0.908259 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.16787 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.913655 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.08123 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  1.00701 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.0139 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 1.11819 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.997835 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.06654 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.891284 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.35633 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 0.515911 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 1.1861 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 1.99976 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 1.00232 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 0.71823 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.616833 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.562081 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 0.926971 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 1.21149 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.61774 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.800865 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 0.550423 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 2.05679 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 1.00275 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.21353 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.878035 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.50255 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 0.638019 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 1.16542 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 1.90877 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.897997 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 1.26676 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.728594 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.567627 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 1.77829 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 0.87582 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 1.13775 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 1.16661 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 0.848127 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 0.546185 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.15788 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 1.01738 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.02528 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.876695 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 1.15788 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 0.841128 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.04504 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.37547 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.72255 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.17259 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.16333 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.863839 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 0.788884 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 1.09262 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.772392 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.4715 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 1.25808 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.22268 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.0886 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 0.945166 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.43287 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 0.931371 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.910808 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.936278 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.942865 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 0.802006 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.39796 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 1.10149 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.805598 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.00981 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 0.738315 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.911188 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.24862 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 0.939012 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.0509 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.03758 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.25795 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.868361 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.13988 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.985604 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.07055 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.99957 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  1.0779 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  0.880596 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 0.937823 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.58233 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 0.57303 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.06611 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 1.55623 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.875618 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 0.826011 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.775115 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.641328 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.16297 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 1.24616 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 1.49251 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 1.00547 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 0.692274 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.86405 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.873715 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 0.957955 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 1.09181 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.16988 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 0.625613 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 1.32733 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 1.49045 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.81859 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.19664 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.73848 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.867097 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 1.61299 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 0.985297 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 1.18333 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.05893 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.16766 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 0.737172 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.09817 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.930941 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.15326 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 1.20694 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.953424 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.05189 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.821097 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.05838 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 0.623788 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.05387 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 2.46002 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.958642 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 0.64372 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.555348 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.356876 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 0.784208 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 1.08988 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.70317 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.760296 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 0.547644 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 2.1054 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.84382 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 1.33392 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.984288 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.64118 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 0.636993 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 1.38481 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 2.31111 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 1.08176 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 1.24688 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.746992 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.806788 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 1.93452 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 0.856336 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 1.3172 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 1.1011 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 0.723151 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 0.595114 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.18632 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.911045 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.08257 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.933209 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 1.22474 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 0.893183 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.05306 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.50271 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.652689 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.00806 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.13411 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.749938 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 0.899035 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.964491 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.733076 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.28657 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 1.13502 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 1.18569 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 0.991846 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.900819 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.3833 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 0.884675 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.827159 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 1.08419 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 0.982006 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 0.815064 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.46508 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 1.1416 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 0.833658 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.09879 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 0.734582 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 1.05262 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 1.20501 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 0.95488 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.19156 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 0.968457 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.15418 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.954666 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.11818 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.972033 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.07345 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.962356 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  1.06612 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  0.911931 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.935465 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.31333 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 0.722292 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.02451 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 1.19643 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.925255 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 0.876264 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.826911 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.740622 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.12988 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 1.09259 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.2225 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 0.951327 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.787974 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.39847 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.979155 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.997943 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.997086 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.09765 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 0.827659 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.27149 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 1.26033 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.897369 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.19719 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.810634 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.972548 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 1.29982 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.02192 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 1.2434 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.04257 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.05679 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.868923 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.263999704688974 z_k1_s0000
                                 - 0.420998328299178 z_k1_s0001
                                 - 0.340998220333025 z_k1_s0002
                                 - 0.57199995745883 z_k1_s0003
                                 - 0.448000604610458 z_k1_s0004
                                 - 0.21499976073458 z_k1_s0005
                                 - 0.782997595097274 z_k1_s0006
                                 - 0.213999881559877 z_k1_s0007
                                 - 0.225000394268183 z_k1_s0010
                                 - 0.213001814967071 z_k1_s0011
                                 - 0.279000640653233 z_k1_s0012
                                 - 0.258000755615556 z_k1_s0013
                                 - 0.0930004664855558 z_k1_s0014
                                 - 0.355002392608923 z_k1_s0015
                                 - 0.330001716465152 z_k1_s0016
                                 - 0.144000593793182 z_k1_s0017
                                 - 0.0849999191447905 z_k1_s0020
                                 - 0.271999546863382 z_k1_s0021
                                 - 0.169000379558321 z_k1_s0022
                                 - 0.175000047945099 z_k1_s0023
                                 - 0.0630000415602292 z_k1_s0024
                                 - 0.15499993832283 z_k1_s0025
                                 - 0.17200021375171 z_k1_s0026
                                 - 0.124000436658133 z_k1_s0027
                                 - 0.424000059269981 z_k1_s0030
                                 - 0.285000061901755 z_k1_s0031
                                 - 1.27599960876158 z_k1_s0032
                                 - 0.555000769534955 z_k1_s0033
                                 - 0.347999815989527 z_k1_s0034
                                 - 0.68199949764845 z_k1_s0035
                                 - 0.304000122721868 z_k1_s0036
                                 - 0.377999800126555 z_k1_s0037
                                 - 0.165000498821953 z_k1_s0040
                                 - 0.117999949400069 z_k1_s0041
                                 - 0.113000015750893 z_k1_s0042
                                 - 0.150000453474503 z_k1_s0043
                                 - 0.253000357527121 z_k1_s0044
                                 - 0.17800013078987 z_k1_s0045
                                 - 0.0909999899746202 z_k1_s0046
                                 - 0.154000058233949 z_k1_s0047
                                 - 0.196000680020974 z_k1_s0050
                                 - 0.311000071977369 z_k1_s0051
                                 - 0.282999251831666 z_k1_s0052
                                 - 0.124000041174575 z_k1_s0053
                                 - 0.242000099403603 z_k1_s0054
                                 - 0.156999947375089 z_k1_s0055
                                 - 0.14900002634768 z_k1_s0056
                                 - 0.225000203957872 z_k1_s0057
                                 - 0.362000180574102 z_k1_s0060
                                 - 0.311999826527784 z_k1_s0061
                                 - 0.155000046343038 z_k1_s0062
                                 - 0.19999966418436 z_k1_s0063
                                 - 0.279000889337889 z_k1_s0064
                                 - 0.0970000561295285 z_k1_s0065
                                 - 0.170000152556934 z_k1_s0066
                                 - 0.176999396202999 z_k1_s0067
                                 - 0.217999976291142 z_k1_s0070
                                 - 0.474999099303774 z_k1_s0071
                                 - 0.501999949496118 z_k1_s0072
                                 - 0.158000019642218 z_k1_s0073
                                 - 0.212000159026317 z_k1_s0074
                                 - 0.16900016779458 z_k1_s0075
                                 - 0.329000944192048 z_k1_s0076
                                 - 0.167000080039582 z_k1_s0077
                                 - 0.0690001257128479 z_k1_s0100
                                 - 0.0760001164373059 z_k1_s0101
                                 - 0.0820001356297402 z_k1_s0102
                                 - 0.0730001068410888 z_k1_s0103
                                 - 0.283000683576141 z_k1_s0104
                                 - 0.204000462542472 z_k1_s0105
                                 - 0.0820001356297402 z_k1_s0106
                                 - 0.0810001640977163 z_k1_s0107
                                 - 0.112000234043167 z_k1_s0110
                                 - 0.0589999690406987 z_k1_s0111
                                 - 0.0899998795706057 z_k1_s0112
                                 - 0.0419999643662691 z_k1_s0113
                                 - 0.127999650478312 z_k1_s0114
                                 - 0.0750000024754368 z_k1_s0115
                                 - 0.0800001671736906 z_k1_s0116
                                 - 0.0309999722298662 z_k1_s0117
                                 - 0.165000297444715 z_k1_s0120
                                 - 0.166000466593057 z_k1_s0121
                                 - 0.425999959110882 z_k1_s0122
                                 - 0.133000100303762 z_k1_s0123
                                 - 0.132000084555596 z_k1_s0124
                                 - 0.21800005829627 z_k1_s0125
                                 - 0.10200007231115 z_k1_s0126
                                 - 0.192000262444663 z_k1_s0127
                                 - 0.0790000323778965 z_k1_s0130
                                 - 0.060999985995594 z_k1_s0131
                                 - 0.110000007336315 z_k1_s0132
                                 - 0.154000407870809 z_k1_s0133
                                 - 0.0749999778929439 z_k1_s0134
                                 - 0.138999955052256 z_k1_s0135
                                 - 0.0979999432814489 z_k1_s0136
                                 - 0.277999910104513 z_k1_s0137
                                 - 0.324000257722625 z_k1_s0140
                                 - 0.0870000362847445 z_k1_s0141
                                 - 0.0700000217647292 z_k1_s0142
                                 - 0.209999418893514 z_k1_s0143
                                 - 0.171000450242824 z_k1_s0144
                                 - 0.171000450242824 z_k1_s0145
                                 - 0.305999614959601 z_k1_s0146
                                 - 0.277000464732252 z_k1_s0147
                                 - 0.0459999423199416 z_k1_s0150
                                 - 0.0910000836327315 z_k1_s0151
                                 - 0.079999998556322 z_k1_s0152
                                 - 0.0749999323216177 z_k1_s0153
                                 - 0.138999627966976 z_k1_s0154
                                 - 0.052999913768648 z_k1_s0155
                                 - 0.125999940876264 z_k1_s0156
                                 - 0.148000111029083 z_k1_s0157
                                 - 0.0569999260031839 z_k1_s0160
                                 - 0.0299999670964069 z_k1_s0161
                                 - 0.030999988959599 z_k1_s0162
                                 - 0.135999988596916 z_k1_s0163
                                 - 0.0459999725078024 z_k1_s0164
                                 - 0.0649996991090967 z_k1_s0165
                                 - 0.0649996991090967 z_k1_s0166
                                 - 0.143999819102775 z_k1_s0167
                                 - 0.0889999816566792 z_k1_s0170
                                 - 0.242999216865644 z_k1_s0171
                                 - 0.405000393774269 z_k1_s0172
                                 - 0.130999896640811 z_k1_s0173
                                 - 0.151999854132877 z_k1_s0174
                                 - 0.13699990876709 z_k1_s0175
                                 - 0.336999690010378 z_k1_s0176
                                 - 0.204999423232245 z_k1_s0177
                                 - 0.75799837267139 z_k1_s1000
                                 - 0.359999800371605 z_k1_s1001
                                 - 0.479999733828807 z_k1_s1002
                                 - 0.379998582281204 z_k1_s1003
                                 - 0.829998332745711 z_k1_s1004
                                 - 0.318999943807055 z_k1_s1005
                                 - 0.203999886877243 z_k1_s1006
                                 - 0.290000080588224 z_k1_s1007
                                 - 1.00500055106724 z_k1_s1010
                                 - 0.754998380879512 z_k1_s1011
                                 - 0.612999160049787 z_k1_s1012
                                 - 0.687997925009008 z_k1_s1013
                                 - 1.02399983582286 z_k1_s1014
                                 - 0.784000267238773 z_k1_s1015
                                 - 0.881995585400877 z_k1_s1016
                                 - 0.535999246070807 z_k1_s1017
                                 - 0.236999952509703 z_k1_s1020
                                 - 0.520998720063649 z_k1_s1021
                                 - 0.300000108366695 z_k1_s1022
                                 - 0.280000020475591 z_k1_s1023
                                 - 0.301999935655825 z_k1_s1024
                                 - 0.566999890213087 z_k1_s1025
                                 - 0.416999231029803 z_k1_s1026
                                 - 0.401000007716831 z_k1_s1027
                                 - 1.92500266024351 z_k1_s1030
                                 - 0.956999779509487 z_k1_s1031
                                 - 0.732000260859021 z_k1_s1032
                                 - 0.820000068152981 z_k1_s1033
                                 - 2.45899610396545 z_k1_s1034
                                 - 0.7779998646422 z_k1_s1035
                                 - 1.15999609601116 z_k1_s1036
                                 - 1.16999709694239 z_k1_s1037
                                 - 0.226999806184012 z_k1_s1040
                                 - 0.239999939853047 z_k1_s1041
                                 - 0.54600088190252 z_k1_s1042
                                 - 0.396999774534079 z_k1_s1043
                                 - 0.305000608198219 z_k1_s1044
                                 - 0.193999830885457 z_k1_s1045
                                 - 0.522000756467301 z_k1_s1046
                                 - 0.197999851791327 z_k1_s1047
                                 - 0.514000198988124 z_k1_s1050
                                 - 0.568999371845264 z_k1_s1051
                                 - 0.619998728749094 z_k1_s1052
                                 - 0.275999906671936 z_k1_s1053
                                 - 0.197999747884309 z_k1_s1054
                                 - 0.477000197920879 z_k1_s1055
                                 - 0.200999950549118 z_k1_s1056
                                 - 0.551998450544569 z_k1_s1057
                                 - 0.469001459772254 z_k1_s1060
                                 - 1.30300339500587 z_k1_s1061
                                 - 0.279000840626911 z_k1_s1062
                                 - 0.872004120948629 z_k1_s1063
                                 - 0.50600146039916 z_k1_s1064
                                 - 0.852003827311619 z_k1_s1065
                                 - 0.836002507198883 z_k1_s1066
                                 - 0.309000738480863 z_k1_s1067
                                 - 0.243999788006215 z_k1_s1070
                                 - 0.225999995856593 z_k1_s1071
                                 - 0.279999832105499 z_k1_s1072
                                 - 0.636000012753954 z_k1_s1073
                                 - 0.284000092449886 z_k1_s1074
                                 - 0.191000016843406 z_k1_s1075
                                 - 0.157999938102392 z_k1_s1076
                                 - 0.279999832105499 z_k1_s1077
                                 - 0.888000558367533 z_k1_s1100
                                 - 0.954997683813472 z_k1_s1101
                                 - 0.516000189516288 z_k1_s1102
                                 - 0.702000061891955 z_k1_s1103
                                 - 0.454000232057733 z_k1_s1104
                                 - 0.474999945863089 z_k1_s1105
                                 - 0.546999856195611 z_k1_s1106
                                 - 1.7039987100029 z_k1_s1107
                                 - 0.322000263465764 z_k1_s1110
                                 - 0.187000236857463 z_k1_s1111
                                 - 0.264000652034129 z_k1_s1112
                                 - 0.219999643361596 z_k1_s1113
                                 - 0.0700000337969006 z_k1_s1114
                                 - 0.131999966016993 z_k1_s1115
                                 - 0.266000452428286 z_k1_s1116
                                 - 0.338999466816281 z_k1_s1117
                                 - 0.826001304688628 z_k1_s1120
                                 - 0.537000996633924 z_k1_s1121
                                 - 0.871000452404813 z_k1_s1122
                                 - 1.6550035029601 z_k1_s1123
                                 - 0.912998706538152 z_k1_s1124
                                 - 0.70800132196206 z_k1_s1125
                                 - 0.424000602408565 z_k1_s1126
                                 - 1.19399924531013 z_k1_s1127
                                 - 0.478997063456382 z_k1_s1130
                                 - 0.996996470725871 z_k1_s1131
                                 - 0.3419988790214 z_k1_s1132
                                 - 0.427999801526669 z_k1_s1133
                                 - 0.775995762407689 z_k1_s1134
                                 - 0.311998758974671 z_k1_s1135
                                 - 0.218999292106651 z_k1_s1136
                                 - 0.218999292106651 z_k1_s1137
                                 - 0.762997790193872 z_k1_s1140
                                 - 0.588998850158253 z_k1_s1141
                                 - 1.12399786303016 z_k1_s1142
                                 - 0.574998713788326 z_k1_s1143
                                 - 0.851997968405042 z_k1_s1144
                                 - 0.382998703094417 z_k1_s1145
                                 - 0.306998789254854 z_k1_s1146
                                 - 0.227999259420329 z_k1_s1147
                                 - 0.445998774065451 z_k1_s1150
                                 - 1.41199582620994 z_k1_s1151
                                 - 0.551997398714077 z_k1_s1152
                                 - 0.773997515149708 z_k1_s1153
                                 - 0.541996670768843 z_k1_s1154
                                 - 0.326999203088019 z_k1_s1155
                                 - 0.954994647009877 z_k1_s1156
                                 - 0.339999079820253 z_k1_s1157
                                 - 0.221000193399077 z_k1_s1160
                                 - 0.388999011424013 z_k1_s1161
                                 - 0.339001180619801 z_k1_s1162
                                 - 0.383000568688099 z_k1_s1163
                                 - 0.451998872158503 z_k1_s1164
                                 - 0.172999952105974 z_k1_s1165
                                 - 0.402001041354291 z_k1_s1166
                                 - 0.343000142443744 z_k1_s1167
                                 - 1.30600561515415 z_k1_s1170
                                 - 0.314000872767387 z_k1_s1171
                                 - 0.488001559918185 z_k1_s1172
                                 - 0.409001241915215 z_k1_s1173
                                 - 0.29200119529459 z_k1_s1174
                                 - 1.28300547615818 z_k1_s1175
                                 - 0.761000271894724 z_k1_s1176
                                 - 0.385001165197927 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.63476 w_k1_ADM_s000
                                 + 1.4642 w_k1_AXP_s000 + 1.36449 w_k1_DIS_s000
                                 + 1.41105 w_k1_FDX_s000
                                 + 0.697115 w_k1_IBM_s000
                                 + 1.07177 w_k1_INTC_s000
                                 + 1.02842 w_k1_KO_s000
                                 + 0.834673 w_k1_MAT_s000
                                 + 1.06413 w_k1_MSFT_s000
                                 + 1.14414 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.986041 w_k1_ADM_s000
                                 + 0.704436 w_k1_AXP_s000
                                 + 0.662583 w_k1_DIS_s000
                                 + 0.79988 w_k1_FDX_s000
                                 + 0.695914 w_k1_IBM_s000
                                 + 0.779915 w_k1_INTC_s000
                                 + 0.655139 w_k1_KO_s000
                                 + 0.660994 w_k1_MAT_s000
                                 + 0.799104 w_k1_MSFT_s000
                                 + 0.810864 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 1.29724 w_k1_ADM_s000
                                 + 0.857136 w_k1_AXP_s000
                                 + 1.44781 w_k1_DIS_s000
                                 + 1.16901 w_k1_FDX_s000
                                 + 0.924932 w_k1_IBM_s000
                                 + 1.23989 w_k1_INTC_s000
                                 + 1.17996 w_k1_KO_s000
                                 + 0.787555 w_k1_MAT_s000
                                 + 1.11076 w_k1_MSFT_s000
                                 + 1.18179 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.02412 w_k1_ADM_s000
                                 + 1.08743 w_k1_AXP_s000
                                 + 1.04663 w_k1_DIS_s000
                                 + 1.05405 w_k1_FDX_s000 + 1.1344 w_k1_IBM_s000
                                 + 1.02177 w_k1_INTC_s000
                                 + 1.09227 w_k1_KO_s000 + 1.09001 w_k1_MAT_s000
                                 + 0.912516 w_k1_MSFT_s000
                                 + 1.02138 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.991235 w_k1_ADM_s000
                                 + 0.836458 w_k1_AXP_s000
                                 + 0.834258 w_k1_DIS_s000
                                 + 0.821661 w_k1_FDX_s000
                                 + 1.27806 w_k1_IBM_s000
                                 + 0.841499 w_k1_INTC_s000
                                 + 0.963195 w_k1_KO_s000
                                 + 1.41719 w_k1_MAT_s000
                                 + 0.872704 w_k1_MSFT_s000
                                 + 1.13916 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.31947 w_k1_ADM_s000
                                 + 1.45978 w_k1_AXP_s000
                                 + 1.21686 w_k1_DIS_s000 + 1.1392 w_k1_FDX_s000
                                 + 1.59435 w_k1_IBM_s000
                                 + 1.05795 w_k1_INTC_s000
                                 + 1.19426 w_k1_KO_s000 + 2.25578 w_k1_MAT_s000
                                 + 1.00826 w_k1_MSFT_s000
                                 + 1.18893 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.01779 w_k1_ADM_s000
                                 + 0.922821 w_k1_AXP_s000
                                 + 0.780978 w_k1_DIS_s000
                                 + 0.889451 w_k1_FDX_s000
                                 + 0.893366 w_k1_IBM_s000
                                 + 1.00475 w_k1_INTC_s000
                                 + 0.875192 w_k1_KO_s000
                                 + 0.970651 w_k1_MAT_s000
                                 + 0.90522 w_k1_MSFT_s000
                                 + 0.997791 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 1.36551 w_k1_ADM_s000
                                 + 1.59171 w_k1_AXP_s000
                                 + 1.51371 w_k1_DIS_s000 + 1.3244 w_k1_FDX_s000
                                 + 1.28158 w_k1_IBM_s000
                                 + 1.13347 w_k1_INTC_s000
                                 + 1.09611 w_k1_KO_s000 + 1.56324 w_k1_MAT_s000
                                 + 1.03431 w_k1_MSFT_s000
                                 + 1.31485 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.96497 w_k1_ADM_s001
                                 + 1.09499 w_k1_AXP_s001
                                 + 1.24003 w_k1_DIS_s001
                                 + 1.14151 w_k1_FDX_s001
                                 + 0.827083 w_k1_IBM_s001
                                 + 1.20982 w_k1_INTC_s001
                                 + 0.999333 w_k1_KO_s001
                                 + 1.01999 w_k1_MAT_s001
                                 + 0.980401 w_k1_MSFT_s001
                                 + 1.07521 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.985673 w_k1_ADM_s001
                                 + 1.13594 w_k1_AXP_s001
                                 + 0.960378 w_k1_DIS_s001
                                 + 0.979549 w_k1_FDX_s001
                                 + 1.26607 w_k1_IBM_s001
                                 + 1.07261 w_k1_INTC_s001
                                 + 1.12036 w_k1_KO_s001 + 2.46217 w_k1_MAT_s001
                                 + 0.958253 w_k1_MSFT_s001
                                 + 1.15771 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 1.00201 w_k1_ADM_s001
                                 + 1.03472 w_k1_AXP_s001
                                 + 0.844195 w_k1_DIS_s001
                                 + 0.996609 w_k1_FDX_s001
                                 + 0.580875 w_k1_IBM_s001
                                 + 0.917354 w_k1_INTC_s001
                                 + 0.677834 w_k1_KO_s001
                                 + 0.80854 w_k1_MAT_s001
                                 + 0.881894 w_k1_MSFT_s001
                                 + 0.914846 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 1.11753 w_k1_ADM_s001
                                 + 1.50352 w_k1_AXP_s001
                                 + 1.34114 w_k1_DIS_s001
                                 + 1.30133 w_k1_FDX_s001
                                 + 1.14539 w_k1_IBM_s001
                                 + 1.15307 w_k1_INTC_s001
                                 + 1.19756 w_k1_KO_s001 + 2.10386 w_k1_MAT_s001
                                 + 1.14159 w_k1_MSFT_s001
                                 + 1.06747 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 0.640513 w_k1_ADM_s001
                                 + 0.693041 w_k1_AXP_s001
                                 + 0.491385 w_k1_DIS_s001
                                 + 0.751784 w_k1_FDX_s001
                                 + 0.452351 w_k1_IBM_s001
                                 + 0.791668 w_k1_INTC_s001
                                 + 0.494001 w_k1_KO_s001
                                 + 0.713002 w_k1_MAT_s001
                                 + 0.709147 w_k1_MSFT_s001
                                 + 0.718736 w_k1_PG_s001 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 0.776856 w_k1_ADM_s001
                                 + 0.756257 w_k1_AXP_s001
                                 + 0.676995 w_k1_DIS_s001
                                 + 0.76177 w_k1_FDX_s001
                                 + 0.870362 w_k1_IBM_s001
                                 + 1.12203 w_k1_INTC_s001
                                 + 0.752368 w_k1_KO_s001
                                 + 0.923824 w_k1_MAT_s001
                                 + 0.795443 w_k1_MSFT_s001
                                 + 0.877049 w_k1_PG_s001 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.914873 w_k1_ADM_s001
                                 + 0.963771 w_k1_AXP_s001
                                 + 0.94585 w_k1_DIS_s001
                                 + 0.925386 w_k1_FDX_s001
                                 + 1.02388 w_k1_IBM_s001
                                 + 0.82229 w_k1_INTC_s001
                                 + 0.939816 w_k1_KO_s001
                                 + 1.20354 w_k1_MAT_s001
                                 + 0.873606 w_k1_MSFT_s001
                                 + 0.932039 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 1.02366 w_k1_ADM_s001
                                 + 1.25537 w_k1_AXP_s001
                                 + 1.60313 w_k1_DIS_s001
                                 + 1.17921 w_k1_FDX_s001
                                 + 1.86009 w_k1_IBM_s001
                                 + 1.35618 w_k1_INTC_s001
                                 + 1.47331 w_k1_KO_s001 + 2.59597 w_k1_MAT_s001
                                 + 1.19751 w_k1_MSFT_s001 + 1.1024 w_k1_PG_s001
                                 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.771744 w_k1_ADM_s002
                                 + 1.00847 w_k1_AXP_s002
                                 + 1.12309 w_k1_DIS_s002
                                 + 0.976265 w_k1_FDX_s002
                                 + 1.57588 w_k1_IBM_s002
                                 + 1.29145 w_k1_INTC_s002
                                 + 1.24148 w_k1_KO_s002 + 1.29782 w_k1_MAT_s002
                                 + 1.10457 w_k1_MSFT_s002
                                 + 1.01478 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 0.790095 w_k1_ADM_s002
                                 + 0.906189 w_k1_AXP_s002
                                 + 0.997614 w_k1_DIS_s002
                                 + 1.01789 w_k1_FDX_s002
                                 + 0.875626 w_k1_IBM_s002
                                 + 1.02077 w_k1_INTC_s002
                                 + 0.797669 w_k1_KO_s002
                                 + 0.527529 w_k1_MAT_s002
                                 + 0.94221 w_k1_MSFT_s002
                                 + 0.875366 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 0.953039 w_k1_ADM_s002
                                 + 1.2504 w_k1_AXP_s002 + 1.09714 w_k1_DIS_s002
                                 + 1.17587 w_k1_FDX_s002
                                 + 0.949138 w_k1_IBM_s002
                                 + 1.02549 w_k1_INTC_s002
                                 + 0.966764 w_k1_KO_s002
                                 + 0.921302 w_k1_MAT_s002
                                 + 1.04578 w_k1_MSFT_s002
                                 + 1.01464 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 0.685047 w_k1_ADM_s002
                                 + 0.537204 w_k1_AXP_s002
                                 + 0.672512 w_k1_DIS_s002
                                 + 0.699937 w_k1_FDX_s002
                                 + 0.990134 w_k1_IBM_s002
                                 + 0.855325 w_k1_INTC_s002
                                 + 0.943548 w_k1_KO_s002
                                 + 0.714647 w_k1_MAT_s002
                                 + 0.885969 w_k1_MSFT_s002
                                 + 0.857409 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 1.34286 w_k1_ADM_s002
                                 + 1.0779 w_k1_AXP_s002 + 1.27578 w_k1_DIS_s002
                                 + 1.31024 w_k1_FDX_s002
                                 + 0.75121 w_k1_IBM_s002
                                 + 1.21634 w_k1_INTC_s002
                                 + 1.01933 w_k1_KO_s002
                                 + 0.659927 w_k1_MAT_s002
                                 + 1.32733 w_k1_MSFT_s002
                                 + 1.01587 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 0.986235 w_k1_ADM_s002
                                 + 1.02899 w_k1_AXP_s002
                                 + 1.17057 w_k1_DIS_s002
                                 + 1.04995 w_k1_FDX_s002
                                 + 1.14484 w_k1_IBM_s002
                                 + 1.3805 w_k1_INTC_s002 + 1.22104 w_k1_KO_s002
                                 + 0.854063 w_k1_MAT_s002
                                 + 1.30769 w_k1_MSFT_s002
                                 + 1.08856 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 0.811449 w_k1_ADM_s002
                                 + 0.796488 w_k1_AXP_s002
                                 + 0.636034 w_k1_DIS_s002
                                 + 0.82755 w_k1_FDX_s002
                                 + 0.782802 w_k1_IBM_s002
                                 + 1.00858 w_k1_INTC_s002
                                 + 0.702212 w_k1_KO_s002
                                 + 0.42459 w_k1_MAT_s002
                                 + 0.891937 w_k1_MSFT_s002
                                 + 0.838051 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.09225 w_k1_ADM_s002
                                 + 1.5061 w_k1_AXP_s002 + 1.6035 w_k1_DIS_s002
                                 + 1.36806 w_k1_FDX_s002
                                 + 1.53667 w_k1_IBM_s002
                                 + 1.25418 w_k1_INTC_s002
                                 + 1.34759 w_k1_KO_s002 + 1.2316 w_k1_MAT_s002
                                 + 1.21021 w_k1_MSFT_s002
                                 + 1.14386 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 0.7406 w_k1_ADM_s003
                                 + 1.03358 w_k1_AXP_s003
                                 + 0.652959 w_k1_DIS_s003
                                 + 0.851189 w_k1_FDX_s003
                                 + 0.911772 w_k1_IBM_s003
                                 + 0.781146 w_k1_INTC_s003
                                 + 0.808396 w_k1_KO_s003
                                 + 1.13489 w_k1_MAT_s003
                                 + 0.878637 w_k1_MSFT_s003
                                 + 0.903242 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.31567 w_k1_ADM_s003
                                 + 1.3885 w_k1_AXP_s003
                                 + 0.948292 w_k1_DIS_s003
                                 + 1.15783 w_k1_FDX_s003
                                 + 0.945688 w_k1_IBM_s003
                                 + 1.02163 w_k1_INTC_s003
                                 + 1.11053 w_k1_KO_s003 + 1.27642 w_k1_MAT_s003
                                 + 1.03828 w_k1_MSFT_s003
                                 + 1.09847 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 0.93429 w_k1_ADM_s003
                                 + 1.08309 w_k1_AXP_s003
                                 + 1.12507 w_k1_DIS_s003
                                 + 1.05011 w_k1_FDX_s003
                                 + 0.948701 w_k1_IBM_s003
                                 + 0.930554 w_k1_INTC_s003
                                 + 1.13352 w_k1_KO_s003 + 1.31738 w_k1_MAT_s003
                                 + 1.06099 w_k1_MSFT_s003
                                 + 1.03482 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.671343 w_k1_ADM_s003
                                 + 0.928937 w_k1_AXP_s003
                                 + 0.832435 w_k1_DIS_s003
                                 + 0.771991 w_k1_FDX_s003
                                 + 1.30205 w_k1_IBM_s003
                                 + 0.844933 w_k1_INTC_s003
                                 + 1.07713 w_k1_KO_s003 + 1.46426 w_k1_MAT_s003
                                 + 0.932112 w_k1_MSFT_s003
                                 + 1.03712 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.37956 w_k1_ADM_s003
                                 + 1.51469 w_k1_AXP_s003
                                 + 1.54479 w_k1_DIS_s003
                                 + 1.42374 w_k1_FDX_s003
                                 + 1.15687 w_k1_IBM_s003
                                 + 1.25679 w_k1_INTC_s003
                                 + 1.43932 w_k1_KO_s003
                                 + 0.998603 w_k1_MAT_s003
                                 + 1.4524 w_k1_MSFT_s003 + 1.33021 w_k1_PG_s003
                                 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 0.690578 w_k1_ADM_s003
                                 + 0.680324 w_k1_AXP_s003
                                 + 0.759426 w_k1_DIS_s003
                                 + 0.792834 w_k1_FDX_s003
                                 + 0.858022 w_k1_IBM_s003
                                 + 0.834178 w_k1_INTC_s003
                                 + 0.989041 w_k1_KO_s003
                                 + 0.994221 w_k1_MAT_s003
                                 + 0.846778 w_k1_MSFT_s003
                                 + 0.870459 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.0347 w_k1_ADM_s003
                                 + 0.888704 w_k1_AXP_s003
                                 + 1.41231 w_k1_DIS_s003
                                 + 1.14389 w_k1_FDX_s003
                                 + 0.935207 w_k1_IBM_s003
                                 + 1.22396 w_k1_INTC_s003
                                 + 1.26441 w_k1_KO_s003
                                 + 0.808257 w_k1_MAT_s003
                                 + 1.41893 w_k1_MSFT_s003
                                 + 1.11646 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.989675 w_k1_ADM_s003
                                 + 1.2719 w_k1_AXP_s003 + 1.12678 w_k1_DIS_s003
                                 + 0.973746 w_k1_FDX_s003
                                 + 1.68273 w_k1_IBM_s003
                                 + 0.944252 w_k1_INTC_s003
                                 + 1.29754 w_k1_KO_s003 + 1.9881 w_k1_MAT_s003
                                 + 0.897948 w_k1_MSFT_s003
                                 + 1.12329 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 0.943624 w_k1_ADM_s004
                                 + 0.9752 w_k1_AXP_s004 + 1.49962 w_k1_DIS_s004
                                 + 1.19771 w_k1_FDX_s004
                                 + 1.15016 w_k1_IBM_s004
                                 + 1.46179 w_k1_INTC_s004
                                 + 1.24202 w_k1_KO_s004
                                 + 0.859212 w_k1_MAT_s004
                                 + 1.41171 w_k1_MSFT_s004
                                 + 1.16614 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 1.03054 w_k1_ADM_s004
                                 + 1.16102 w_k1_AXP_s004
                                 + 0.897397 w_k1_DIS_s004
                                 + 0.89839 w_k1_FDX_s004
                                 + 1.53772 w_k1_IBM_s004
                                 + 1.09581 w_k1_INTC_s004
                                 + 1.26569 w_k1_KO_s004 + 1.6441 w_k1_MAT_s004
                                 + 1.07289 w_k1_MSFT_s004
                                 + 1.08343 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 1.63853 w_k1_ADM_s004
                                 + 1.81212 w_k1_AXP_s004
                                 + 1.35805 w_k1_DIS_s004
                                 + 1.40439 w_k1_FDX_s004
                                 + 1.37597 w_k1_IBM_s004
                                 + 0.945514 w_k1_INTC_s004
                                 + 1.26357 w_k1_KO_s004 + 1.53886 w_k1_MAT_s004
                                 + 1.13813 w_k1_MSFT_s004
                                 + 1.17077 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 1.00593 w_k1_ADM_s004
                                 + 0.683827 w_k1_AXP_s004
                                 + 0.795527 w_k1_DIS_s004
                                 + 0.854949 w_k1_FDX_s004
                                 + 0.812782 w_k1_IBM_s004
                                 + 0.873048 w_k1_INTC_s004
                                 + 0.884501 w_k1_KO_s004
                                 + 0.601659 w_k1_MAT_s004
                                 + 0.898971 w_k1_MSFT_s004
                                 + 0.904432 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.13291 w_k1_ADM_s004
                                 + 1.04619 w_k1_AXP_s004
                                 + 0.962244 w_k1_DIS_s004
                                 + 1.09091 w_k1_FDX_s004
                                 + 0.781018 w_k1_IBM_s004
                                 + 1.16335 w_k1_INTC_s004
                                 + 0.969357 w_k1_KO_s004
                                 + 0.794978 w_k1_MAT_s004
                                 + 1.26648 w_k1_MSFT_s004
                                 + 1.01078 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 0.977424 w_k1_ADM_s004
                                 + 1.0852 w_k1_AXP_s004
                                 + 0.721921 w_k1_DIS_s004
                                 + 0.888183 w_k1_FDX_s004
                                 + 1.02244 w_k1_IBM_s004
                                 + 0.933922 w_k1_INTC_s004
                                 + 0.779724 w_k1_KO_s004 + 0.8741 w_k1_MAT_s004
                                 + 0.848605 w_k1_MSFT_s004
                                 + 0.989448 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 0.959564 w_k1_ADM_s004
                                 + 0.686734 w_k1_AXP_s004
                                 + 0.608654 w_k1_DIS_s004
                                 + 0.701551 w_k1_FDX_s004
                                 + 0.70571 w_k1_IBM_s004
                                 + 0.755395 w_k1_INTC_s004
                                 + 0.762614 w_k1_KO_s004
                                 + 0.402243 w_k1_MAT_s004
                                 + 0.731914 w_k1_MSFT_s004
                                 + 0.748462 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 1.08945 w_k1_ADM_s004
                                 + 0.900355 w_k1_AXP_s004
                                 + 1.13013 w_k1_DIS_s004
                                 + 0.911173 w_k1_FDX_s004
                                 + 0.98712 w_k1_IBM_s004
                                 + 1.15972 w_k1_INTC_s004
                                 + 1.12079 w_k1_KO_s004
                                 + 0.834358 w_k1_MAT_s004
                                 + 1.0671 w_k1_MSFT_s004 + 1.00072 w_k1_PG_s004
                                 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.871898 w_k1_ADM_s005
                                 + 0.657641 w_k1_AXP_s005
                                 + 0.597297 w_k1_DIS_s005
                                 + 0.867036 w_k1_FDX_s005
                                 + 0.595558 w_k1_IBM_s005
                                 + 0.990581 w_k1_INTC_s005
                                 + 0.648163 w_k1_KO_s005
                                 + 0.545923 w_k1_MAT_s005
                                 + 0.841784 w_k1_MSFT_s005
                                 + 0.733113 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.750087 w_k1_ADM_s005
                                 + 0.898194 w_k1_AXP_s005
                                 + 0.830687 w_k1_DIS_s005
                                 + 0.867938 w_k1_FDX_s005
                                 + 1.11745 w_k1_IBM_s005
                                 + 1.04237 w_k1_INTC_s005
                                 + 0.995028 w_k1_KO_s005
                                 + 1.01237 w_k1_MAT_s005
                                 + 1.01375 w_k1_MSFT_s005
                                 + 0.881417 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.3809 w_k1_ADM_s005
                                 + 1.34795 w_k1_AXP_s005
                                 + 1.15434 w_k1_DIS_s005
                                 + 1.27332 w_k1_FDX_s005
                                 + 1.03376 w_k1_IBM_s005
                                 + 1.07768 w_k1_INTC_s005
                                 + 1.02883 w_k1_KO_s005 + 1.20159 w_k1_MAT_s005
                                 + 1.11418 w_k1_MSFT_s005
                                 + 1.02136 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 0.996641 w_k1_ADM_s005
                                 + 0.789977 w_k1_AXP_s005
                                 + 1.05808 w_k1_DIS_s005
                                 + 0.993317 w_k1_FDX_s005
                                 + 1.0237 w_k1_IBM_s005
                                 + 0.953357 w_k1_INTC_s005
                                 + 0.975265 w_k1_KO_s005
                                 + 1.24857 w_k1_MAT_s005
                                 + 1.01193 w_k1_MSFT_s005
                                 + 0.904015 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.06057 w_k1_ADM_s005
                                 + 1.05342 w_k1_AXP_s005
                                 + 1.02669 w_k1_DIS_s005
                                 + 1.07571 w_k1_FDX_s005
                                 + 0.880981 w_k1_IBM_s005
                                 + 1.03894 w_k1_INTC_s005
                                 + 0.954772 w_k1_KO_s005
                                 + 1.24224 w_k1_MAT_s005
                                 + 0.99156 w_k1_MSFT_s005
                                 + 0.950941 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.23973 w_k1_ADM_s005
                                 + 0.926702 w_k1_AXP_s005
                                 + 0.991933 w_k1_DIS_s005
                                 + 1.13728 w_k1_FDX_s005
                                 + 0.736411 w_k1_IBM_s005
                                 + 0.837866 w_k1_INTC_s005
                                 + 0.856893 w_k1_KO_s005
                                 + 0.846744 w_k1_MAT_s005
                                 + 1.02098 w_k1_MSFT_s005
                                 + 0.884922 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.830868 w_k1_ADM_s005
                                 + 0.657752 w_k1_AXP_s005
                                 + 0.700339 w_k1_DIS_s005
                                 + 0.831146 w_k1_FDX_s005
                                 + 0.782575 w_k1_IBM_s005
                                 + 0.682771 w_k1_INTC_s005
                                 + 0.742208 w_k1_KO_s005
                                 + 0.737495 w_k1_MAT_s005
                                 + 0.780065 w_k1_MSFT_s005
                                 + 0.870011 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 0.986359 w_k1_ADM_s005
                                 + 1.10118 w_k1_AXP_s005
                                 + 1.44078 w_k1_DIS_s005
                                 + 1.16467 w_k1_FDX_s005
                                 + 1.62588 w_k1_IBM_s005
                                 + 1.20122 w_k1_INTC_s005
                                 + 1.42533 w_k1_KO_s005 + 1.60952 w_k1_MAT_s005
                                 + 1.10992 w_k1_MSFT_s005
                                 + 1.14214 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 0.951579 w_k1_ADM_s006
                                 + 1.04867 w_k1_AXP_s006
                                 + 0.97855 w_k1_DIS_s006
                                 + 0.913032 w_k1_FDX_s006
                                 + 1.14802 w_k1_IBM_s006
                                 + 0.955065 w_k1_INTC_s006
                                 + 1.11604 w_k1_KO_s006 + 1.14999 w_k1_MAT_s006
                                 + 0.858672 w_k1_MSFT_s006
                                 + 0.894716 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 0.905456 w_k1_ADM_s006
                                 + 0.997459 w_k1_AXP_s006
                                 + 0.920713 w_k1_DIS_s006
                                 + 0.966241 w_k1_FDX_s006
                                 + 0.829484 w_k1_IBM_s006
                                 + 1.13655 w_k1_INTC_s006
                                 + 0.823657 w_k1_KO_s006
                                 + 0.900886 w_k1_MAT_s006
                                 + 0.789486 w_k1_MSFT_s006
                                 + 0.860176 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 0.951297 w_k1_ADM_s006
                                 + 1.17582 w_k1_AXP_s006
                                 + 1.00233 w_k1_DIS_s006 + 1.0863 w_k1_FDX_s006
                                 + 1.07218 w_k1_IBM_s006
                                 + 1.10668 w_k1_INTC_s006
                                 + 0.981856 w_k1_KO_s006
                                 + 1.45051 w_k1_MAT_s006
                                 + 1.00631 w_k1_MSFT_s006
                                 + 0.958305 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 0.856315 w_k1_ADM_s006
                                 + 1.03359 w_k1_AXP_s006
                                 + 1.27174 w_k1_DIS_s006 + 1.2396 w_k1_FDX_s006
                                 + 0.78296 w_k1_IBM_s006
                                 + 1.07494 w_k1_INTC_s006
                                 + 0.888249 w_k1_KO_s006
                                 + 0.625739 w_k1_MAT_s006
                                 + 1.03573 w_k1_MSFT_s006
                                 + 0.953025 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 0.535664 w_k1_ADM_s006
                                 + 0.668842 w_k1_AXP_s006
                                 + 0.637176 w_k1_DIS_s006
                                 + 0.765127 w_k1_FDX_s006
                                 + 0.719049 w_k1_IBM_s006
                                 + 0.88246 w_k1_INTC_s006
                                 + 0.618568 w_k1_KO_s006
                                 + 0.741538 w_k1_MAT_s006
                                 + 0.742522 w_k1_MSFT_s006
                                 + 0.85614 w_k1_PG_s006 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 0.700073 w_k1_ADM_s006
                                 + 0.906218 w_k1_AXP_s006
                                 + 1.09569 w_k1_DIS_s006
                                 + 0.905449 w_k1_FDX_s006
                                 + 1.65592 w_k1_IBM_s006
                                 + 1.16014 w_k1_INTC_s006
                                 + 1.28277 w_k1_KO_s006 + 1.75789 w_k1_MAT_s006
                                 + 0.903743 w_k1_MSFT_s006
                                 + 0.968848 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 1.01218 w_k1_ADM_s006
                                 + 1.39173 w_k1_AXP_s006
                                 + 1.79793 w_k1_DIS_s006 + 1.232 w_k1_FDX_s006
                                 + 1.68681 w_k1_IBM_s006
                                 + 1.10182 w_k1_INTC_s006
                                 + 1.48409 w_k1_KO_s006 + 2.15484 w_k1_MAT_s006
                                 + 1.07683 w_k1_MSFT_s006
                                 + 1.11288 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.24597 w_k1_ADM_s006
                                 + 1.2795 w_k1_AXP_s006 + 1.40995 w_k1_DIS_s006
                                 + 1.25044 w_k1_FDX_s006
                                 + 1.19659 w_k1_IBM_s006
                                 + 1.17525 w_k1_INTC_s006
                                 + 1.19615 w_k1_KO_s006 + 1.19864 w_k1_MAT_s006
                                 + 1.22591 w_k1_MSFT_s006
                                 + 1.08422 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 1.21077 w_k1_ADM_s007
                                 + 1.19195 w_k1_AXP_s007
                                 + 1.73842 w_k1_DIS_s007
                                 + 1.37866 w_k1_FDX_s007
                                 + 1.07202 w_k1_IBM_s007
                                 + 1.03957 w_k1_INTC_s007
                                 + 1.30019 w_k1_KO_s007 + 1.0135 w_k1_MAT_s007
                                 + 1.35172 w_k1_MSFT_s007
                                 + 1.14383 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 1.08631 w_k1_ADM_s007
                                 + 1.12279 w_k1_AXP_s007
                                 + 1.25503 w_k1_DIS_s007 + 1.127 w_k1_FDX_s007
                                 + 1.01658 w_k1_IBM_s007
                                 + 0.986411 w_k1_INTC_s007
                                 + 1.01663 w_k1_KO_s007 + 1.1518 w_k1_MAT_s007
                                 + 1.08004 w_k1_MSFT_s007 + 1.0155 w_k1_PG_s007
                                 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 0.906241 w_k1_ADM_s007
                                 + 0.809107 w_k1_AXP_s007
                                 + 0.911401 w_k1_DIS_s007
                                 + 0.920109 w_k1_FDX_s007
                                 + 0.828515 w_k1_IBM_s007
                                 + 0.826647 w_k1_INTC_s007
                                 + 0.948564 w_k1_KO_s007
                                 + 0.933285 w_k1_MAT_s007
                                 + 1.0053 w_k1_MSFT_s007 + 1.01351 w_k1_PG_s007
                                 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 1.15234 w_k1_ADM_s007
                                 + 1.31856 w_k1_AXP_s007
                                 + 1.68004 w_k1_DIS_s007
                                 + 1.06859 w_k1_FDX_s007
                                 + 1.37895 w_k1_IBM_s007
                                 + 0.989195 w_k1_INTC_s007
                                 + 1.58556 w_k1_KO_s007 + 1.4146 w_k1_MAT_s007
                                 + 1.10259 w_k1_MSFT_s007
                                 + 1.27379 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 1.07259 w_k1_ADM_s007
                                 + 0.921382 w_k1_AXP_s007
                                 + 1.29075 w_k1_DIS_s007 + 0.9013 w_k1_FDX_s007
                                 + 1.3724 w_k1_IBM_s007
                                 + 1.00596 w_k1_INTC_s007
                                 + 1.15865 w_k1_KO_s007
                                 + 0.978211 w_k1_MAT_s007
                                 + 1.0095 w_k1_MSFT_s007 + 1.00503 w_k1_PG_s007
                                 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.783234 w_k1_ADM_s007
                                 + 0.626586 w_k1_AXP_s007
                                 + 0.757291 w_k1_DIS_s007
                                 + 0.714657 w_k1_FDX_s007
                                 + 1.05482 w_k1_IBM_s007
                                 + 0.641193 w_k1_INTC_s007
                                 + 0.999056 w_k1_KO_s007
                                 + 1.15222 w_k1_MAT_s007
                                 + 0.818937 w_k1_MSFT_s007
                                 + 0.977912 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 0.865427 w_k1_ADM_s007
                                 + 1.0784 w_k1_AXP_s007
                                 + 0.793688 w_k1_DIS_s007
                                 + 0.917546 w_k1_FDX_s007
                                 + 1.00741 w_k1_IBM_s007
                                 + 0.857248 w_k1_INTC_s007
                                 + 0.856138 w_k1_KO_s007
                                 + 1.06566 w_k1_MAT_s007
                                 + 0.789291 w_k1_MSFT_s007
                                 + 0.986208 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.897868 w_k1_ADM_s007
                                 + 0.584385 w_k1_AXP_s007
                                 + 0.604502 w_k1_DIS_s007
                                 + 0.741687 w_k1_FDX_s007
                                 + 0.617354 w_k1_IBM_s007
                                 + 0.761194 w_k1_INTC_s007
                                 + 0.67947 w_k1_KO_s007
                                 + 0.368404 w_k1_MAT_s007
                                 + 0.750639 w_k1_MSFT_s007
                                 + 0.865598 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 0.904501 w_k1_ADM_s010
                                 + 0.954367 w_k1_AXP_s010
                                 + 1.05394 w_k1_DIS_s010
                                 + 0.928972 w_k1_FDX_s010
                                 + 1.15321 w_k1_IBM_s010
                                 + 1.05428 w_k1_INTC_s010 + 1.1578 w_k1_KO_s010
                                 + 1.1327 w_k1_MAT_s010
                                 + 0.960437 w_k1_MSFT_s010
                                 + 1.01571 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.19554 w_k1_ADM_s010
                                 + 1.24811 w_k1_AXP_s010
                                 + 0.803388 w_k1_DIS_s010
                                 + 1.04552 w_k1_FDX_s010 + 1.0753 w_k1_IBM_s010
                                 + 0.919565 w_k1_INTC_s010
                                 + 0.881693 w_k1_KO_s010
                                 + 1.44068 w_k1_MAT_s010
                                 + 0.858376 w_k1_MSFT_s010
                                 + 1.03951 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 0.919009 w_k1_ADM_s010
                                 + 0.983962 w_k1_AXP_s010
                                 + 0.737624 w_k1_DIS_s010
                                 + 0.892078 w_k1_FDX_s010
                                 + 1.59661 w_k1_IBM_s010
                                 + 1.15032 w_k1_INTC_s010
                                 + 1.15072 w_k1_KO_s010 + 1.98982 w_k1_MAT_s010
                                 + 1.04247 w_k1_MSFT_s010 + 1.1018 w_k1_PG_s010
                                 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.918577 w_k1_ADM_s010
                                 + 0.799707 w_k1_AXP_s010
                                 + 0.736429 w_k1_DIS_s010
                                 + 0.813587 w_k1_FDX_s010
                                 + 1.05887 w_k1_IBM_s010
                                 + 1.01381 w_k1_INTC_s010
                                 + 0.808588 w_k1_KO_s010
                                 + 0.988333 w_k1_MAT_s010
                                 + 0.880026 w_k1_MSFT_s010
                                 + 0.890915 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.851032 w_k1_ADM_s010
                                 + 0.715166 w_k1_AXP_s010
                                 + 0.788196 w_k1_DIS_s010
                                 + 0.89301 w_k1_FDX_s010
                                 + 0.643106 w_k1_IBM_s010
                                 + 0.900853 w_k1_INTC_s010
                                 + 0.770853 w_k1_KO_s010
                                 + 0.705161 w_k1_MAT_s010
                                 + 0.902424 w_k1_MSFT_s010
                                 + 0.824559 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.54598 w_k1_ADM_s010
                                 + 1.1326 w_k1_AXP_s010 + 1.43608 w_k1_DIS_s010
                                 + 1.27131 w_k1_FDX_s010
                                 + 1.04448 w_k1_IBM_s010
                                 + 1.16815 w_k1_INTC_s010
                                 + 1.25059 w_k1_KO_s010 + 1.28188 w_k1_MAT_s010
                                 + 1.31365 w_k1_MSFT_s010
                                 + 1.15014 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.04101 w_k1_ADM_s010
                                 + 0.833869 w_k1_AXP_s010
                                 + 0.810498 w_k1_DIS_s010
                                 + 0.979273 w_k1_FDX_s010
                                 + 1.17413 w_k1_IBM_s010
                                 + 0.957725 w_k1_INTC_s010
                                 + 1.08488 w_k1_KO_s010 + 1.69598 w_k1_MAT_s010
                                 + 0.965774 w_k1_MSFT_s010
                                 + 1.05644 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 1.463 w_k1_ADM_s010
                                 + 1.4141 w_k1_AXP_s010 + 1.50884 w_k1_DIS_s010
                                 + 1.17115 w_k1_FDX_s010
                                 + 1.83823 w_k1_IBM_s010
                                 + 1.21099 w_k1_INTC_s010
                                 + 1.76079 w_k1_KO_s010 + 2.29425 w_k1_MAT_s010
                                 + 1.25073 w_k1_MSFT_s010
                                 + 1.31038 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.37576 w_k1_ADM_s011
                                 + 1.36913 w_k1_AXP_s011
                                 + 1.29794 w_k1_DIS_s011
                                 + 1.17322 w_k1_FDX_s011
                                 + 1.22602 w_k1_IBM_s011
                                 + 0.96334 w_k1_INTC_s011
                                 + 1.17207 w_k1_KO_s011 + 1.51931 w_k1_MAT_s011
                                 + 1.10586 w_k1_MSFT_s011
                                 + 1.06067 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.978635 w_k1_ADM_s011
                                 + 0.766538 w_k1_AXP_s011
                                 + 0.836342 w_k1_DIS_s011
                                 + 0.808652 w_k1_FDX_s011
                                 + 1.00011 w_k1_IBM_s011
                                 + 0.753116 w_k1_INTC_s011
                                 + 0.786317 w_k1_KO_s011
                                 + 0.874179 w_k1_MAT_s011
                                 + 0.849862 w_k1_MSFT_s011
                                 + 0.90452 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.14834 w_k1_ADM_s011
                                 + 1.13347 w_k1_AXP_s011
                                 + 0.91649 w_k1_DIS_s011
                                 + 0.908262 w_k1_FDX_s011
                                 + 1.09886 w_k1_IBM_s011
                                 + 0.807461 w_k1_INTC_s011
                                 + 1.06859 w_k1_KO_s011 + 1.40163 w_k1_MAT_s011
                                 + 1.04468 w_k1_MSFT_s011
                                 + 0.909214 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 1.09187 w_k1_ADM_s011
                                 + 0.964414 w_k1_AXP_s011
                                 + 0.777755 w_k1_DIS_s011
                                 + 1.01343 w_k1_FDX_s011
                                 + 0.761332 w_k1_IBM_s011
                                 + 0.890033 w_k1_INTC_s011
                                 + 0.785966 w_k1_KO_s011
                                 + 0.934783 w_k1_MAT_s011
                                 + 0.839939 w_k1_MSFT_s011
                                 + 0.913748 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.37312 w_k1_ADM_s011
                                 + 1.15101 w_k1_AXP_s011
                                 + 1.07509 w_k1_DIS_s011
                                 + 1.07107 w_k1_FDX_s011
                                 + 1.07214 w_k1_IBM_s011
                                 + 0.799079 w_k1_INTC_s011
                                 + 1.01551 w_k1_KO_s011
                                 + 0.873529 w_k1_MAT_s011
                                 + 1.03621 w_k1_MSFT_s011
                                 + 0.961026 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.915878 w_k1_ADM_s011
                                 + 0.706287 w_k1_AXP_s011
                                 + 0.946951 w_k1_DIS_s011
                                 + 0.982011 w_k1_FDX_s011
                                 + 0.713317 w_k1_IBM_s011
                                 + 0.892622 w_k1_INTC_s011
                                 + 0.817295 w_k1_KO_s011
                                 + 0.415984 w_k1_MAT_s011
                                 + 0.967525 w_k1_MSFT_s011
                                 + 0.890401 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 1.24618 w_k1_ADM_s011
                                 + 1.25648 w_k1_AXP_s011
                                 + 1.77088 w_k1_DIS_s011
                                 + 1.35634 w_k1_FDX_s011
                                 + 0.884316 w_k1_IBM_s011
                                 + 0.97991 w_k1_INTC_s011 + 1.3023 w_k1_KO_s011
                                 + 1.44074 w_k1_MAT_s011
                                 + 1.29863 w_k1_MSFT_s011
                                 + 1.10311 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.901277 w_k1_ADM_s011
                                 + 0.691442 w_k1_AXP_s011
                                 + 0.640337 w_k1_DIS_s011
                                 + 0.700561 w_k1_FDX_s011
                                 + 0.712134 w_k1_IBM_s011
                                 + 0.599839 w_k1_INTC_s011
                                 + 0.735547 w_k1_KO_s011
                                 + 0.723381 w_k1_MAT_s011
                                 + 0.73126 w_k1_MSFT_s011
                                 + 0.726543 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.31096 w_k1_ADM_s012
                                 + 1.11419 w_k1_AXP_s012
                                 + 1.07308 w_k1_DIS_s012
                                 + 1.15349 w_k1_FDX_s012
                                 + 0.822982 w_k1_IBM_s012
                                 + 0.939655 w_k1_INTC_s012
                                 + 0.887875 w_k1_KO_s012
                                 + 0.811828 w_k1_MAT_s012
                                 + 0.955801 w_k1_MSFT_s012
                                 + 1.05204 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 1.09885 w_k1_ADM_s012
                                 + 1.34436 w_k1_AXP_s012
                                 + 1.09289 w_k1_DIS_s012
                                 + 1.15525 w_k1_FDX_s012
                                 + 1.39645 w_k1_IBM_s012
                                 + 1.28434 w_k1_INTC_s012 + 1.1571 w_k1_KO_s012
                                 + 1.7328 w_k1_MAT_s012
                                 + 1.27372 w_k1_MSFT_s012
                                 + 1.12131 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 1.00652 w_k1_ADM_s012
                                 + 0.78645 w_k1_AXP_s012
                                 + 0.732217 w_k1_DIS_s012
                                 + 0.853983 w_k1_FDX_s012
                                 + 0.82413 w_k1_IBM_s012
                                 + 0.824482 w_k1_INTC_s012
                                 + 0.833462 w_k1_KO_s012
                                 + 0.85183 w_k1_MAT_s012
                                 + 0.742211 w_k1_MSFT_s012
                                 + 0.91286 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.17198 w_k1_ADM_s012
                                 + 1.2314 w_k1_AXP_s012 + 1.98183 w_k1_DIS_s012
                                 + 1.40117 w_k1_FDX_s012
                                 + 1.32822 w_k1_IBM_s012
                                 + 1.33206 w_k1_INTC_s012
                                 + 1.35486 w_k1_KO_s012 + 1.2203 w_k1_MAT_s012
                                 + 1.31099 w_k1_MSFT_s012 + 1.2312 w_k1_PG_s012
                                 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 0.89002 w_k1_ADM_s012
                                 + 0.895668 w_k1_AXP_s012
                                 + 0.79886 w_k1_DIS_s012
                                 + 0.858031 w_k1_FDX_s012
                                 + 1.46146 w_k1_IBM_s012
                                 + 0.754772 w_k1_INTC_s012
                                 + 1.02385 w_k1_KO_s012 + 1.7092 w_k1_MAT_s012
                                 + 0.746453 w_k1_MSFT_s012
                                 + 0.98413 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.744002 w_k1_ADM_s012
                                 + 0.765577 w_k1_AXP_s012
                                 + 0.815392 w_k1_DIS_s012
                                 + 0.789593 w_k1_FDX_s012
                                 + 1.09911 w_k1_IBM_s012
                                 + 0.849336 w_k1_INTC_s012
                                 + 0.885484 w_k1_KO_s012
                                 + 1.03687 w_k1_MAT_s012
                                 + 0.806044 w_k1_MSFT_s012
                                 + 0.955409 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 1.24813 w_k1_ADM_s012
                                 + 1.4353 w_k1_AXP_s012 + 1.31232 w_k1_DIS_s012
                                 + 1.29895 w_k1_FDX_s012 + 1.0557 w_k1_IBM_s012
                                 + 1.15026 w_k1_INTC_s012
                                 + 1.17763 w_k1_KO_s012 + 1.01076 w_k1_MAT_s012
                                 + 1.07379 w_k1_MSFT_s012
                                 + 1.07894 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 1.16391 w_k1_ADM_s012
                                 + 0.987067 w_k1_AXP_s012
                                 + 1.03863 w_k1_DIS_s012
                                 + 1.05221 w_k1_FDX_s012
                                 + 1.13323 w_k1_IBM_s012
                                 + 1.05656 w_k1_INTC_s012
                                 + 1.04406 w_k1_KO_s012 + 1.14678 w_k1_MAT_s012
                                 + 1.04485 w_k1_MSFT_s012
                                 + 1.12745 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.01429 w_k1_ADM_s013
                                 + 1.04063 w_k1_AXP_s013
                                 + 1.28479 w_k1_DIS_s013
                                 + 1.34818 w_k1_FDX_s013
                                 + 0.729656 w_k1_IBM_s013
                                 + 1.19718 w_k1_INTC_s013
                                 + 0.95641 w_k1_KO_s013
                                 + 0.779376 w_k1_MAT_s013
                                 + 1.32035 w_k1_MSFT_s013
                                 + 1.10025 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.34635 w_k1_ADM_s013
                                 + 1.40852 w_k1_AXP_s013
                                 + 1.91199 w_k1_DIS_s013
                                 + 1.50732 w_k1_FDX_s013
                                 + 1.06225 w_k1_IBM_s013
                                 + 1.45853 w_k1_INTC_s013
                                 + 1.38861 w_k1_KO_s013 + 1.04064 w_k1_MAT_s013
                                 + 1.40337 w_k1_MSFT_s013
                                 + 1.27182 w_k1_PG_s013 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 0.853234 w_k1_ADM_s013
                                 + 1.08276 w_k1_AXP_s013
                                 + 1.13943 w_k1_DIS_s013
                                 + 1.11453 w_k1_FDX_s013
                                 + 0.781075 w_k1_IBM_s013
                                 + 1.05972 w_k1_INTC_s013
                                 + 0.924924 w_k1_KO_s013
                                 + 0.805035 w_k1_MAT_s013
                                 + 1.11827 w_k1_MSFT_s013
                                 + 0.979767 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 0.897868 w_k1_ADM_s013
                                 + 0.95737 w_k1_AXP_s013
                                 + 0.823982 w_k1_DIS_s013
                                 + 0.964349 w_k1_FDX_s013
                                 + 0.859141 w_k1_IBM_s013
                                 + 0.912591 w_k1_INTC_s013
                                 + 0.895421 w_k1_KO_s013
                                 + 0.849336 w_k1_MAT_s013
                                 + 0.837738 w_k1_MSFT_s013
                                 + 0.87339 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.37282 w_k1_ADM_s013
                                 + 1.44726 w_k1_AXP_s013
                                 + 1.40841 w_k1_DIS_s013
                                 + 1.39094 w_k1_FDX_s013 + 1.2939 w_k1_IBM_s013
                                 + 1.11468 w_k1_INTC_s013
                                 + 0.967612 w_k1_KO_s013
                                 + 1.09787 w_k1_MAT_s013
                                 + 1.05505 w_k1_MSFT_s013 + 1.0167 w_k1_PG_s013
                                 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.26031 w_k1_ADM_s013
                                 + 1.25601 w_k1_AXP_s013
                                 + 1.06825 w_k1_DIS_s013
                                 + 1.12632 w_k1_FDX_s013
                                 + 1.50513 w_k1_IBM_s013
                                 + 1.0941 w_k1_INTC_s013 + 1.24492 w_k1_KO_s013
                                 + 1.16087 w_k1_MAT_s013
                                 + 0.931023 w_k1_MSFT_s013
                                 + 1.13611 w_k1_PG_s013 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 0.611537 w_k1_ADM_s013
                                 + 0.582303 w_k1_AXP_s013
                                 + 0.603062 w_k1_DIS_s013
                                 + 0.694961 w_k1_FDX_s013
                                 + 0.66219 w_k1_IBM_s013
                                 + 0.699448 w_k1_INTC_s013
                                 + 0.654167 w_k1_KO_s013
                                 + 0.479784 w_k1_MAT_s013
                                 + 0.762532 w_k1_MSFT_s013
                                 + 0.728554 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 0.971124 w_k1_ADM_s013
                                 + 0.999227 w_k1_AXP_s013
                                 + 0.851177 w_k1_DIS_s013
                                 + 0.85886 w_k1_FDX_s013 + 1.1469 w_k1_IBM_s013
                                 + 0.929346 w_k1_INTC_s013
                                 + 0.932907 w_k1_KO_s013
                                 + 0.705327 w_k1_MAT_s013
                                 + 0.81741 w_k1_MSFT_s013
                                 + 0.872342 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.860398 w_k1_ADM_s014
                                 + 0.987382 w_k1_AXP_s014
                                 + 0.936686 w_k1_DIS_s014
                                 + 1.11126 w_k1_FDX_s014
                                 + 0.799661 w_k1_IBM_s014
                                 + 0.863807 w_k1_INTC_s014
                                 + 0.929408 w_k1_KO_s014
                                 + 0.693644 w_k1_MAT_s014
                                 + 1.04839 w_k1_MSFT_s014
                                 + 1.06672 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 0.715913 w_k1_ADM_s014
                                 + 0.564957 w_k1_AXP_s014
                                 + 0.726116 w_k1_DIS_s014
                                 + 0.753876 w_k1_FDX_s014
                                 + 0.794738 w_k1_IBM_s014
                                 + 0.649699 w_k1_INTC_s014
                                 + 0.806782 w_k1_KO_s014
                                 + 0.40966 w_k1_MAT_s014
                                 + 0.886102 w_k1_MSFT_s014
                                 + 0.883336 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 0.791 w_k1_ADM_s014
                                 + 0.653974 w_k1_AXP_s014
                                 + 0.634181 w_k1_DIS_s014
                                 + 0.859724 w_k1_FDX_s014
                                 + 0.570359 w_k1_IBM_s014
                                 + 0.977256 w_k1_INTC_s014
                                 + 0.583053 w_k1_KO_s014
                                 + 0.323492 w_k1_MAT_s014
                                 + 0.906844 w_k1_MSFT_s014
                                 + 0.822325 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 1.0529 w_k1_ADM_s014
                                 + 1.30944 w_k1_AXP_s014
                                 + 1.03421 w_k1_DIS_s014
                                 + 1.18247 w_k1_FDX_s014
                                 + 1.02979 w_k1_IBM_s014
                                 + 1.10839 w_k1_INTC_s014
                                 + 0.911418 w_k1_KO_s014
                                 + 0.996013 w_k1_MAT_s014
                                 + 1.01755 w_k1_MSFT_s014
                                 + 1.15032 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.733745 w_k1_ADM_s014
                                 + 0.916324 w_k1_AXP_s014
                                 + 0.986363 w_k1_DIS_s014
                                 + 0.893434 w_k1_FDX_s014
                                 + 1.44375 w_k1_IBM_s014
                                 + 0.914389 w_k1_INTC_s014
                                 + 1.12939 w_k1_KO_s014 + 1.72315 w_k1_MAT_s014
                                 + 1.01538 w_k1_MSFT_s014
                                 + 1.12549 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 0.92304 w_k1_ADM_s014
                                 + 0.909881 w_k1_AXP_s014
                                 + 1.30119 w_k1_DIS_s014
                                 + 1.18257 w_k1_FDX_s014
                                 + 1.01914 w_k1_IBM_s014
                                 + 1.19109 w_k1_INTC_s014
                                 + 1.30062 w_k1_KO_s014
                                 + 0.881285 w_k1_MAT_s014
                                 + 1.4104 w_k1_MSFT_s014 + 1.24764 w_k1_PG_s014
                                 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 0.812593 w_k1_ADM_s014
                                 + 0.873221 w_k1_AXP_s014
                                 + 0.816481 w_k1_DIS_s014
                                 + 0.847161 w_k1_FDX_s014
                                 + 0.956893 w_k1_IBM_s014
                                 + 0.93279 w_k1_INTC_s014
                                 + 0.815695 w_k1_KO_s014
                                 + 0.677709 w_k1_MAT_s014
                                 + 0.879395 w_k1_MSFT_s014
                                 + 0.890289 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 0.958522 w_k1_ADM_s014
                                 + 1.31518 w_k1_AXP_s014
                                 + 1.43805 w_k1_DIS_s014
                                 + 1.19966 w_k1_FDX_s014
                                 + 1.45535 w_k1_IBM_s014
                                 + 1.16697 w_k1_INTC_s014
                                 + 1.43052 w_k1_KO_s014 + 1.00014 w_k1_MAT_s014
                                 + 1.27759 w_k1_MSFT_s014
                                 + 1.17881 w_k1_PG_s014 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 0.775062 w_k1_ADM_s015
                                 + 0.733805 w_k1_AXP_s015
                                 + 0.526586 w_k1_DIS_s015
                                 + 0.710513 w_k1_FDX_s015
                                 + 0.908428 w_k1_IBM_s015
                                 + 0.940601 w_k1_INTC_s015
                                 + 0.696374 w_k1_KO_s015
                                 + 0.866671 w_k1_MAT_s015
                                 + 0.729178 w_k1_MSFT_s015
                                 + 0.770001 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.14499 w_k1_ADM_s015
                                 + 0.715319 w_k1_AXP_s015
                                 + 1.25996 w_k1_DIS_s015
                                 + 1.06844 w_k1_FDX_s015
                                 + 0.668311 w_k1_IBM_s015
                                 + 1.07316 w_k1_INTC_s015
                                 + 1.10526 w_k1_KO_s015
                                 + 0.606448 w_k1_MAT_s015
                                 + 1.18133 w_k1_MSFT_s015
                                 + 1.12464 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.78053 w_k1_ADM_s015
                                 + 0.783893 w_k1_AXP_s015
                                 + 0.867596 w_k1_DIS_s015
                                 + 0.8377 w_k1_FDX_s015 + 1.05493 w_k1_IBM_s015
                                 + 0.926857 w_k1_INTC_s015
                                 + 1.10199 w_k1_KO_s015
                                 + 0.899168 w_k1_MAT_s015
                                 + 0.930132 w_k1_MSFT_s015
                                 + 1.01835 w_k1_PG_s015 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 1.08692 w_k1_ADM_s015
                                 + 0.981893 w_k1_AXP_s015
                                 + 0.924206 w_k1_DIS_s015
                                 + 0.909627 w_k1_FDX_s015
                                 + 1.43173 w_k1_IBM_s015
                                 + 0.805006 w_k1_INTC_s015
                                 + 1.02147 w_k1_KO_s015 + 1.30416 w_k1_MAT_s015
                                 + 0.811539 w_k1_MSFT_s015
                                 + 1.07317 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 1.35944 w_k1_ADM_s015
                                 + 1.06107 w_k1_AXP_s015
                                 + 1.26199 w_k1_DIS_s015
                                 + 1.14608 w_k1_FDX_s015 + 1.1734 w_k1_IBM_s015
                                 + 1.12719 w_k1_INTC_s015
                                 + 1.25562 w_k1_KO_s015
                                 + 0.805852 w_k1_MAT_s015
                                 + 1.23943 w_k1_MSFT_s015
                                 + 1.17131 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 1.03444 w_k1_ADM_s015
                                 + 1.09053 w_k1_AXP_s015
                                 + 0.798591 w_k1_DIS_s015
                                 + 1.04486 w_k1_FDX_s015
                                 + 0.748868 w_k1_IBM_s015
                                 + 0.913455 w_k1_INTC_s015
                                 + 0.829088 w_k1_KO_s015
                                 + 0.591104 w_k1_MAT_s015
                                 + 0.88702 w_k1_MSFT_s015
                                 + 0.924853 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.889586 w_k1_ADM_s015
                                 + 1.59425 w_k1_AXP_s015
                                 + 1.56191 w_k1_DIS_s015
                                 + 1.12325 w_k1_FDX_s015 + 1.1932 w_k1_IBM_s015
                                 + 1.21317 w_k1_INTC_s015
                                 + 1.07992 w_k1_KO_s015 + 1.07059 w_k1_MAT_s015
                                 + 1.06811 w_k1_MSFT_s015
                                 + 1.26323 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 0.926961 w_k1_ADM_s015
                                 + 0.922948 w_k1_AXP_s015
                                 + 0.811545 w_k1_DIS_s015
                                 + 0.845607 w_k1_FDX_s015
                                 + 0.740308 w_k1_IBM_s015
                                 + 1.03072 w_k1_INTC_s015
                                 + 0.673227 w_k1_KO_s015
                                 + 0.478751 w_k1_MAT_s015
                                 + 0.849956 w_k1_MSFT_s015
                                 + 0.803856 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.13986 w_k1_ADM_s016
                                 + 1.01346 w_k1_AXP_s016
                                 + 1.09817 w_k1_DIS_s016
                                 + 1.18998 w_k1_FDX_s016
                                 + 0.713191 w_k1_IBM_s016
                                 + 1.10804 w_k1_INTC_s016
                                 + 0.894445 w_k1_KO_s016
                                 + 0.551527 w_k1_MAT_s016
                                 + 1.07099 w_k1_MSFT_s016
                                 + 0.970804 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 0.911442 w_k1_ADM_s016
                                 + 0.62383 w_k1_AXP_s016
                                 + 0.662019 w_k1_DIS_s016
                                 + 0.888316 w_k1_FDX_s016
                                 + 0.568233 w_k1_IBM_s016
                                 + 0.809126 w_k1_INTC_s016
                                 + 0.693796 w_k1_KO_s016
                                 + 0.538978 w_k1_MAT_s016
                                 + 0.812125 w_k1_MSFT_s016
                                 + 0.807779 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 0.658047 w_k1_ADM_s016
                                 + 0.712646 w_k1_AXP_s016
                                 + 0.54957 w_k1_DIS_s016
                                 + 0.608607 w_k1_FDX_s016
                                 + 1.25439 w_k1_IBM_s016
                                 + 0.716428 w_k1_INTC_s016
                                 + 0.827779 w_k1_KO_s016
                                 + 1.07504 w_k1_MAT_s016
                                 + 0.618499 w_k1_MSFT_s016
                                 + 0.861813 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.07017 w_k1_ADM_s016
                                 + 1.29378 w_k1_AXP_s016
                                 + 1.34844 w_k1_DIS_s016 + 1.1587 w_k1_FDX_s016
                                 + 1.27458 w_k1_IBM_s016
                                 + 0.998551 w_k1_INTC_s016
                                 + 1.26337 w_k1_KO_s016 + 1.5925 w_k1_MAT_s016
                                 + 1.14635 w_k1_MSFT_s016
                                 + 1.17079 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.72782 w_k1_ADM_s016
                                 + 1.72828 w_k1_AXP_s016
                                 + 1.86296 w_k1_DIS_s016
                                 + 1.61142 w_k1_FDX_s016
                                 + 0.993093 w_k1_IBM_s016
                                 + 1.12163 w_k1_INTC_s016
                                 + 1.38276 w_k1_KO_s016
                                 + 0.816663 w_k1_MAT_s016
                                 + 1.22368 w_k1_MSFT_s016
                                 + 1.30712 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.08 w_k1_ADM_s016
                                 + 1.57111 w_k1_AXP_s016
                                 + 1.01375 w_k1_DIS_s016
                                 + 1.22362 w_k1_FDX_s016
                                 + 1.20258 w_k1_IBM_s016
                                 + 0.9746 w_k1_INTC_s016 + 1.03863 w_k1_KO_s016
                                 + 1.69902 w_k1_MAT_s016
                                 + 1.07474 w_k1_MSFT_s016
                                 + 0.998136 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.802468 w_k1_ADM_s016
                                 + 0.933799 w_k1_AXP_s016
                                 + 0.829966 w_k1_DIS_s016
                                 + 0.899618 w_k1_FDX_s016
                                 + 0.756403 w_k1_IBM_s016
                                 + 1.04248 w_k1_INTC_s016
                                 + 0.834352 w_k1_KO_s016
                                 + 0.638025 w_k1_MAT_s016
                                 + 0.929227 w_k1_MSFT_s016
                                 + 0.879242 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 0.8784 w_k1_ADM_s016
                                 + 1.11912 w_k1_AXP_s016
                                 + 0.766242 w_k1_DIS_s016
                                 + 0.809114 w_k1_FDX_s016
                                 + 1.21481 w_k1_IBM_s016
                                 + 0.73043 w_k1_INTC_s016
                                 + 0.868297 w_k1_KO_s016
                                 + 1.16723 w_k1_MAT_s016
                                 + 0.690514 w_k1_MSFT_s016
                                 + 0.758523 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.01994 w_k1_ADM_s017
                                 + 0.980308 w_k1_AXP_s017
                                 + 0.932855 w_k1_DIS_s017
                                 + 1.06887 w_k1_FDX_s017
                                 + 0.682676 w_k1_IBM_s017
                                 + 1.01492 w_k1_INTC_s017
                                 + 0.921071 w_k1_KO_s017
                                 + 0.68394 w_k1_MAT_s017
                                 + 1.03249 w_k1_MSFT_s017
                                 + 0.889002 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 0.927715 w_k1_ADM_s017
                                 + 1.00706 w_k1_AXP_s017
                                 + 1.05892 w_k1_DIS_s017
                                 + 0.97004 w_k1_FDX_s017
                                 + 1.21378 w_k1_IBM_s017
                                 + 1.07276 w_k1_INTC_s017
                                 + 1.10696 w_k1_KO_s017 + 1.0144 w_k1_MAT_s017
                                 + 0.945076 w_k1_MSFT_s017
                                 + 0.855622 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 0.932596 w_k1_ADM_s017
                                 + 1.10911 w_k1_AXP_s017
                                 + 1.06902 w_k1_DIS_s017 + 1.071 w_k1_FDX_s017
                                 + 0.894867 w_k1_IBM_s017
                                 + 1.00616 w_k1_INTC_s017
                                 + 0.880015 w_k1_KO_s017
                                 + 0.970975 w_k1_MAT_s017
                                 + 1.09143 w_k1_MSFT_s017
                                 + 0.912577 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 1.24252 w_k1_ADM_s017
                                 + 0.950871 w_k1_AXP_s017
                                 + 0.897627 w_k1_DIS_s017
                                 + 0.818857 w_k1_FDX_s017
                                 + 1.42812 w_k1_IBM_s017
                                 + 0.929658 w_k1_INTC_s017
                                 + 1.08669 w_k1_KO_s017 + 1.23803 w_k1_MAT_s017
                                 + 0.905973 w_k1_MSFT_s017
                                 + 0.948528 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 0.893138 w_k1_ADM_s017
                                 + 0.938558 w_k1_AXP_s017
                                 + 0.731875 w_k1_DIS_s017
                                 + 0.971252 w_k1_FDX_s017
                                 + 0.610135 w_k1_IBM_s017
                                 + 0.862628 w_k1_INTC_s017
                                 + 0.770902 w_k1_KO_s017
                                 + 0.770563 w_k1_MAT_s017
                                 + 0.877618 w_k1_MSFT_s017
                                 + 0.868632 w_k1_PG_s017 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 1.11863 w_k1_ADM_s017
                                 + 1.37107 w_k1_AXP_s017
                                 + 1.22109 w_k1_DIS_s017
                                 + 1.32209 w_k1_FDX_s017
                                 + 0.782491 w_k1_IBM_s017
                                 + 1.41747 w_k1_INTC_s017
                                 + 1.17751 w_k1_KO_s017
                                 + 0.793029 w_k1_MAT_s017
                                 + 1.12583 w_k1_MSFT_s017
                                 + 1.08732 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.940327 w_k1_ADM_s017
                                 + 0.659493 w_k1_AXP_s017
                                 + 0.635098 w_k1_DIS_s017
                                 + 0.748263 w_k1_FDX_s017
                                 + 0.797934 w_k1_IBM_s017
                                 + 0.848245 w_k1_INTC_s017
                                 + 0.748251 w_k1_KO_s017
                                 + 0.563545 w_k1_MAT_s017
                                 + 0.743281 w_k1_MSFT_s017
                                 + 0.762323 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.04663 w_k1_ADM_s017
                                 + 1.38475 w_k1_AXP_s017
                                 + 1.32606 w_k1_DIS_s017
                                 + 1.20179 w_k1_FDX_s017
                                 + 1.39801 w_k1_IBM_s017
                                 + 1.12002 w_k1_INTC_s017
                                 + 1.35951 w_k1_KO_s017 + 1.51551 w_k1_MAT_s017
                                 + 1.21026 w_k1_MSFT_s017
                                 + 0.998288 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.789791 w_k1_ADM_s100
                                 + 0.897123 w_k1_AXP_s100
                                 + 0.904351 w_k1_DIS_s100
                                 + 0.870437 w_k1_FDX_s100
                                 + 0.997328 w_k1_IBM_s100
                                 + 0.842503 w_k1_INTC_s100
                                 + 1.05396 w_k1_KO_s100
                                 + 0.881597 w_k1_MAT_s100
                                 + 1.00662 w_k1_MSFT_s100
                                 + 0.921756 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.17734 w_k1_ADM_s100
                                 + 1.37445 w_k1_AXP_s100
                                 + 1.13557 w_k1_DIS_s100
                                 + 1.14375 w_k1_FDX_s100
                                 + 0.972009 w_k1_IBM_s100
                                 + 0.913504 w_k1_INTC_s100
                                 + 1.14588 w_k1_KO_s100 + 1.23172 w_k1_MAT_s100
                                 + 1.15067 w_k1_MSFT_s100
                                 + 1.02703 w_k1_PG_s100 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.13499 w_k1_ADM_s100
                                 + 1.4106 w_k1_AXP_s100 + 1.47277 w_k1_DIS_s100
                                 + 1.17758 w_k1_FDX_s100
                                 + 1.26378 w_k1_IBM_s100
                                 + 1.23335 w_k1_INTC_s100
                                 + 1.55581 w_k1_KO_s100 + 1.52785 w_k1_MAT_s100
                                 + 1.47762 w_k1_MSFT_s100
                                 + 1.14496 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.772098 w_k1_ADM_s100
                                 + 1.07933 w_k1_AXP_s100
                                 + 0.78492 w_k1_DIS_s100
                                 + 0.931341 w_k1_FDX_s100
                                 + 0.806883 w_k1_IBM_s100
                                 + 0.797719 w_k1_INTC_s100
                                 + 0.800377 w_k1_KO_s100
                                 + 0.964056 w_k1_MAT_s100
                                 + 0.867014 w_k1_MSFT_s100
                                 + 0.898632 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.819382 w_k1_ADM_s100
                                 + 0.976367 w_k1_AXP_s100
                                 + 1.00942 w_k1_DIS_s100
                                 + 0.997871 w_k1_FDX_s100
                                 + 0.727166 w_k1_IBM_s100
                                 + 0.981299 w_k1_INTC_s100
                                 + 1.03233 w_k1_KO_s100
                                 + 0.443973 w_k1_MAT_s100
                                 + 1.06653 w_k1_MSFT_s100
                                 + 1.00953 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 1.04729 w_k1_ADM_s100
                                 + 1.18001 w_k1_AXP_s100
                                 + 1.17518 w_k1_DIS_s100
                                 + 0.99043 w_k1_FDX_s100
                                 + 1.62817 w_k1_IBM_s100
                                 + 0.871841 w_k1_INTC_s100
                                 + 1.37935 w_k1_KO_s100 + 1.78747 w_k1_MAT_s100
                                 + 1.20326 w_k1_MSFT_s100
                                 + 1.19105 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.706475 w_k1_ADM_s100
                                 + 0.761102 w_k1_AXP_s100
                                 + 0.733482 w_k1_DIS_s100
                                 + 0.711318 w_k1_FDX_s100
                                 + 1.25914 w_k1_IBM_s100
                                 + 0.960967 w_k1_INTC_s100
                                 + 1.14941 w_k1_KO_s100
                                 + 0.997475 w_k1_MAT_s100
                                 + 0.96121 w_k1_MSFT_s100
                                 + 0.95296 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.13294 w_k1_ADM_s100
                                 + 1.36029 w_k1_AXP_s100
                                 + 1.45603 w_k1_DIS_s100
                                 + 1.38382 w_k1_FDX_s100
                                 + 1.02607 w_k1_IBM_s100
                                 + 1.22245 w_k1_INTC_s100
                                 + 1.33289 w_k1_KO_s100 + 1.06959 w_k1_MAT_s100
                                 + 1.38361 w_k1_MSFT_s100 + 1.064 w_k1_PG_s100
                                 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.763709 w_k1_ADM_s101
                                 + 0.78425 w_k1_AXP_s101
                                 + 0.76489 w_k1_DIS_s101
                                 + 0.794925 w_k1_FDX_s101
                                 + 0.981423 w_k1_IBM_s101
                                 + 0.815321 w_k1_INTC_s101
                                 + 0.816108 w_k1_KO_s101
                                 + 0.872582 w_k1_MAT_s101
                                 + 0.826189 w_k1_MSFT_s101
                                 + 0.935066 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.22035 w_k1_ADM_s101
                                 + 1.29944 w_k1_AXP_s101
                                 + 1.15985 w_k1_DIS_s101
                                 + 1.07569 w_k1_FDX_s101
                                 + 1.15216 w_k1_IBM_s101
                                 + 1.07818 w_k1_INTC_s101
                                 + 1.09305 w_k1_KO_s101 + 1.20932 w_k1_MAT_s101
                                 + 1.08872 w_k1_MSFT_s101
                                 + 1.15277 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 0.888716 w_k1_ADM_s101
                                 + 0.736247 w_k1_AXP_s101
                                 + 0.79489 w_k1_DIS_s101
                                 + 0.81731 w_k1_FDX_s101
                                 + 0.598728 w_k1_IBM_s101
                                 + 0.796873 w_k1_INTC_s101
                                 + 0.746219 w_k1_KO_s101
                                 + 0.591914 w_k1_MAT_s101
                                 + 0.820697 w_k1_MSFT_s101
                                 + 0.866164 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.952867 w_k1_ADM_s101
                                 + 1.32956 w_k1_AXP_s101
                                 + 0.910628 w_k1_DIS_s101
                                 + 0.87109 w_k1_FDX_s101
                                 + 1.38879 w_k1_IBM_s101
                                 + 0.879923 w_k1_INTC_s101
                                 + 1.07293 w_k1_KO_s101 + 1.51942 w_k1_MAT_s101
                                 + 0.895474 w_k1_MSFT_s101
                                 + 1.05139 w_k1_PG_s101 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.998524 w_k1_ADM_s101
                                 + 1.07943 w_k1_AXP_s101
                                 + 0.933453 w_k1_DIS_s101
                                 + 1.01337 w_k1_FDX_s101
                                 + 0.924748 w_k1_IBM_s101
                                 + 1.06194 w_k1_INTC_s101
                                 + 0.788841 w_k1_KO_s101
                                 + 0.890532 w_k1_MAT_s101
                                 + 0.877872 w_k1_MSFT_s101
                                 + 1.04208 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 1.05218 w_k1_ADM_s101
                                 + 1.12396 w_k1_AXP_s101
                                 + 1.10358 w_k1_DIS_s101
                                 + 1.22366 w_k1_FDX_s101
                                 + 0.819629 w_k1_IBM_s101
                                 + 1.21674 w_k1_INTC_s101
                                 + 0.89244 w_k1_KO_s101 + 1.04204 w_k1_MAT_s101
                                 + 1.20861 w_k1_MSFT_s101 + 1.0359 w_k1_PG_s101
                                 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.14574 w_k1_ADM_s101
                                 + 1.3399 w_k1_AXP_s101 + 1.66061 w_k1_DIS_s101
                                 + 1.24121 w_k1_FDX_s101 + 1.29 w_k1_IBM_s101
                                 + 1.35946 w_k1_INTC_s101
                                 + 1.34484 w_k1_KO_s101 + 1.3656 w_k1_MAT_s101
                                 + 1.32985 w_k1_MSFT_s101 + 1.273 w_k1_PG_s101
                                 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.960971 w_k1_ADM_s101
                                 + 1.16021 w_k1_AXP_s101
                                 + 1.24491 w_k1_DIS_s101
                                 + 0.935867 w_k1_FDX_s101
                                 + 1.61813 w_k1_IBM_s101
                                 + 1.00038 w_k1_INTC_s101
                                 + 1.23105 w_k1_KO_s101 + 1.74003 w_k1_MAT_s101
                                 + 0.964344 w_k1_MSFT_s101
                                 + 1.09961 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 0.938018 w_k1_ADM_s102
                                 + 0.597095 w_k1_AXP_s102
                                 + 0.806346 w_k1_DIS_s102
                                 + 0.848226 w_k1_FDX_s102
                                 + 0.745918 w_k1_IBM_s102
                                 + 0.840058 w_k1_INTC_s102
                                 + 0.915704 w_k1_KO_s102
                                 + 0.466752 w_k1_MAT_s102
                                 + 0.879376 w_k1_MSFT_s102
                                 + 0.835575 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 0.818947 w_k1_ADM_s102
                                 + 1.00253 w_k1_AXP_s102
                                 + 1.05789 w_k1_DIS_s102
                                 + 0.942216 w_k1_FDX_s102
                                 + 1.33823 w_k1_IBM_s102
                                 + 1.05458 w_k1_INTC_s102
                                 + 1.30264 w_k1_KO_s102 + 1.41787 w_k1_MAT_s102
                                 + 1.10747 w_k1_MSFT_s102
                                 + 1.05357 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.02119 w_k1_ADM_s102
                                 + 1.18422 w_k1_AXP_s102
                                 + 0.869254 w_k1_DIS_s102
                                 + 0.945173 w_k1_FDX_s102
                                 + 0.928827 w_k1_IBM_s102
                                 + 0.895839 w_k1_INTC_s102
                                 + 0.93775 w_k1_KO_s102 + 1.31996 w_k1_MAT_s102
                                 + 0.90566 w_k1_MSFT_s102
                                 + 0.948116 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 1.11676 w_k1_ADM_s102
                                 + 1.4573 w_k1_AXP_s102 + 1.03719 w_k1_DIS_s102
                                 + 1.03376 w_k1_FDX_s102
                                 + 1.47811 w_k1_IBM_s102
                                 + 0.953196 w_k1_INTC_s102
                                 + 1.21351 w_k1_KO_s102 + 1.78908 w_k1_MAT_s102
                                 + 0.95912 w_k1_MSFT_s102
                                 + 1.19793 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 1.18726 w_k1_ADM_s102
                                 + 1.42509 w_k1_AXP_s102
                                 + 1.06938 w_k1_DIS_s102
                                 + 1.22858 w_k1_FDX_s102
                                 + 0.967739 w_k1_IBM_s102
                                 + 0.97292 w_k1_INTC_s102
                                 + 1.07015 w_k1_KO_s102 + 1.34621 w_k1_MAT_s102
                                 + 1.09593 w_k1_MSFT_s102
                                 + 1.23089 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 0.819044 w_k1_ADM_s102
                                 + 0.839274 w_k1_AXP_s102
                                 + 0.817151 w_k1_DIS_s102
                                 + 0.882689 w_k1_FDX_s102
                                 + 1.00349 w_k1_IBM_s102
                                 + 0.766515 w_k1_INTC_s102
                                 + 0.957806 w_k1_KO_s102
                                 + 1.00528 w_k1_MAT_s102
                                 + 0.872298 w_k1_MSFT_s102
                                 + 1.03079 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.868068 w_k1_ADM_s102
                                 + 0.967496 w_k1_AXP_s102
                                 + 0.707425 w_k1_DIS_s102
                                 + 0.894915 w_k1_FDX_s102
                                 + 0.711191 w_k1_IBM_s102
                                 + 0.785367 w_k1_INTC_s102
                                 + 0.87513 w_k1_KO_s102
                                 + 0.731008 w_k1_MAT_s102
                                 + 0.816652 w_k1_MSFT_s102
                                 + 0.870654 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 1.1791 w_k1_ADM_s102
                                 + 1.14444 w_k1_AXP_s102
                                 + 1.36873 w_k1_DIS_s102
                                 + 1.24028 w_k1_FDX_s102
                                 + 0.972692 w_k1_IBM_s102
                                 + 1.10746 w_k1_INTC_s102
                                 + 1.29206 w_k1_KO_s102 + 1.34028 w_k1_MAT_s102
                                 + 1.22302 w_k1_MSFT_s102
                                 + 1.19019 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.10695 w_k1_ADM_s103
                                 + 1.17714 w_k1_AXP_s103
                                 + 1.28763 w_k1_DIS_s103
                                 + 1.05627 w_k1_FDX_s103
                                 + 1.12347 w_k1_IBM_s103
                                 + 0.964634 w_k1_INTC_s103
                                 + 1.38997 w_k1_KO_s103 + 1.40848 w_k1_MAT_s103
                                 + 1.03453 w_k1_MSFT_s103
                                 + 1.12584 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 1.29537 w_k1_ADM_s103
                                 + 1.75265 w_k1_AXP_s103 + 1.4449 w_k1_DIS_s103
                                 + 1.31232 w_k1_FDX_s103
                                 + 1.32759 w_k1_IBM_s103
                                 + 1.18169 w_k1_INTC_s103
                                 + 1.47121 w_k1_KO_s103 + 1.86579 w_k1_MAT_s103
                                 + 1.41549 w_k1_MSFT_s103
                                 + 1.21645 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 0.765081 w_k1_ADM_s103
                                 + 0.738877 w_k1_AXP_s103
                                 + 0.54964 w_k1_DIS_s103
                                 + 0.665612 w_k1_FDX_s103
                                 + 0.837938 w_k1_IBM_s103
                                 + 0.757045 w_k1_INTC_s103
                                 + 0.869766 w_k1_KO_s103
                                 + 0.916239 w_k1_MAT_s103
                                 + 0.697589 w_k1_MSFT_s103
                                 + 0.782206 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.72325 w_k1_ADM_s103
                                 + 1.42258 w_k1_AXP_s103 + 1.7982 w_k1_DIS_s103
                                 + 1.43617 w_k1_FDX_s103
                                 + 0.895868 w_k1_IBM_s103
                                 + 1.15508 w_k1_INTC_s103
                                 + 1.46072 w_k1_KO_s103 + 1.03849 w_k1_MAT_s103
                                 + 1.4651 w_k1_MSFT_s103 + 1.21861 w_k1_PG_s103
                                 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 1.02758 w_k1_ADM_s103
                                 + 0.805846 w_k1_AXP_s103
                                 + 0.787808 w_k1_DIS_s103
                                 + 0.877283 w_k1_FDX_s103
                                 + 0.652171 w_k1_IBM_s103
                                 + 0.827841 w_k1_INTC_s103
                                 + 0.896452 w_k1_KO_s103
                                 + 0.807004 w_k1_MAT_s103
                                 + 0.853877 w_k1_MSFT_s103
                                 + 0.835905 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.787928 w_k1_ADM_s103
                                 + 0.830417 w_k1_AXP_s103
                                 + 0.760502 w_k1_DIS_s103
                                 + 0.715444 w_k1_FDX_s103
                                 + 1.3038 w_k1_IBM_s103
                                 + 0.759828 w_k1_INTC_s103
                                 + 1.19002 w_k1_KO_s103 + 1.37015 w_k1_MAT_s103
                                 + 0.770945 w_k1_MSFT_s103
                                 + 1.01089 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.938307 w_k1_ADM_s103
                                 + 0.912676 w_k1_AXP_s103
                                 + 1.00984 w_k1_DIS_s103
                                 + 0.863304 w_k1_FDX_s103
                                 + 1.10772 w_k1_IBM_s103
                                 + 1.01554 w_k1_INTC_s103
                                 + 1.10134 w_k1_KO_s103
                                 + 0.718874 w_k1_MAT_s103
                                 + 0.869922 w_k1_MSFT_s103
                                 + 0.915737 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 1.15199 w_k1_ADM_s103
                                 + 0.918306 w_k1_AXP_s103
                                 + 1.15538 w_k1_DIS_s103
                                 + 1.13805 w_k1_FDX_s103
                                 + 0.858463 w_k1_IBM_s103
                                 + 0.887838 w_k1_INTC_s103
                                 + 1.17404 w_k1_KO_s103 + 1.12933 w_k1_MAT_s103
                                 + 1.03124 w_k1_MSFT_s103
                                 + 1.01438 w_k1_PG_s103 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.27371 w_k1_ADM_s104
                                 + 1.37947 w_k1_AXP_s104
                                 + 0.975616 w_k1_DIS_s104
                                 + 1.20251 w_k1_FDX_s104
                                 + 0.98835 w_k1_IBM_s104
                                 + 0.921151 w_k1_INTC_s104
                                 + 0.988189 w_k1_KO_s104
                                 + 1.20282 w_k1_MAT_s104
                                 + 0.939719 w_k1_MSFT_s104
                                 + 1.13042 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.30183 w_k1_ADM_s104
                                 + 1.29576 w_k1_AXP_s104
                                 + 1.80647 w_k1_DIS_s104
                                 + 1.40053 w_k1_FDX_s104 + 1.301 w_k1_IBM_s104
                                 + 0.934044 w_k1_INTC_s104
                                 + 1.31567 w_k1_KO_s104 + 2.00982 w_k1_MAT_s104
                                 + 1.2884 w_k1_MSFT_s104 + 1.24932 w_k1_PG_s104
                                 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.769371 w_k1_ADM_s104
                                 + 0.833962 w_k1_AXP_s104
                                 + 0.655785 w_k1_DIS_s104
                                 + 0.809008 w_k1_FDX_s104
                                 + 1.08854 w_k1_IBM_s104
                                 + 0.742664 w_k1_INTC_s104
                                 + 0.862712 w_k1_KO_s104
                                 + 0.941986 w_k1_MAT_s104
                                 + 0.700305 w_k1_MSFT_s104
                                 + 0.903215 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 1.10397 w_k1_ADM_s104
                                 + 1.12608 w_k1_AXP_s104 + 1.2368 w_k1_DIS_s104
                                 + 1.16024 w_k1_FDX_s104
                                 + 1.12603 w_k1_IBM_s104
                                 + 0.947408 w_k1_INTC_s104
                                 + 1.2339 w_k1_KO_s104 + 1.13626 w_k1_MAT_s104
                                 + 1.05206 w_k1_MSFT_s104
                                 + 1.04715 w_k1_PG_s104 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.81238 w_k1_ADM_s104
                                 + 0.671466 w_k1_AXP_s104
                                 + 0.986428 w_k1_DIS_s104
                                 + 0.925447 w_k1_FDX_s104
                                 + 0.807406 w_k1_IBM_s104
                                 + 0.717112 w_k1_INTC_s104
                                 + 0.965051 w_k1_KO_s104
                                 + 1.11203 w_k1_MAT_s104
                                 + 0.943611 w_k1_MSFT_s104
                                 + 1.06534 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 0.835793 w_k1_ADM_s104
                                 + 0.863759 w_k1_AXP_s104
                                 + 1.15728 w_k1_DIS_s104
                                 + 0.937936 w_k1_FDX_s104
                                 + 1.47021 w_k1_IBM_s104
                                 + 0.706498 w_k1_INTC_s104
                                 + 1.23487 w_k1_KO_s104 + 1.51707 w_k1_MAT_s104
                                 + 0.893 w_k1_MSFT_s104 + 1.09768 w_k1_PG_s104
                                 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 1.08778 w_k1_ADM_s104
                                 + 0.847607 w_k1_AXP_s104
                                 + 0.747724 w_k1_DIS_s104
                                 + 0.966136 w_k1_FDX_s104
                                 + 0.66597 w_k1_IBM_s104
                                 + 0.757185 w_k1_INTC_s104
                                 + 0.723129 w_k1_KO_s104
                                 + 0.830824 w_k1_MAT_s104
                                 + 0.830539 w_k1_MSFT_s104
                                 + 0.86433 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 1.04884 w_k1_ADM_s104
                                 + 1.04139 w_k1_AXP_s104
                                 + 1.07464 w_k1_DIS_s104
                                 + 1.16274 w_k1_FDX_s104
                                 + 0.728299 w_k1_IBM_s104
                                 + 0.895163 w_k1_INTC_s104
                                 + 0.973504 w_k1_KO_s104
                                 + 1.26226 w_k1_MAT_s104
                                 + 0.934683 w_k1_MSFT_s104
                                 + 0.97285 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.29557 w_k1_ADM_s105
                                 + 1.10684 w_k1_AXP_s105
                                 + 1.36422 w_k1_DIS_s105
                                 + 1.32158 w_k1_FDX_s105
                                 + 0.894567 w_k1_IBM_s105
                                 + 1.19123 w_k1_INTC_s105
                                 + 1.20061 w_k1_KO_s105 + 1.15121 w_k1_MAT_s105
                                 + 1.38441 w_k1_MSFT_s105
                                 + 1.21646 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.26723 w_k1_ADM_s105
                                 + 1.39955 w_k1_AXP_s105
                                 + 1.43618 w_k1_DIS_s105
                                 + 1.27775 w_k1_FDX_s105
                                 + 1.52289 w_k1_IBM_s105
                                 + 1.11328 w_k1_INTC_s105
                                 + 1.47573 w_k1_KO_s105 + 1.9808 w_k1_MAT_s105
                                 + 1.26913 w_k1_MSFT_s105
                                 + 1.24912 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 0.877267 w_k1_ADM_s105
                                 + 0.832761 w_k1_AXP_s105
                                 + 1.04414 w_k1_DIS_s105
                                 + 0.991733 w_k1_FDX_s105
                                 + 0.980364 w_k1_IBM_s105
                                 + 1.24631 w_k1_INTC_s105
                                 + 1.11768 w_k1_KO_s105
                                 + 0.963555 w_k1_MAT_s105
                                 + 1.13161 w_k1_MSFT_s105
                                 + 0.996848 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 1.19818 w_k1_ADM_s105
                                 + 1.02183 w_k1_AXP_s105
                                 + 1.03385 w_k1_DIS_s105
                                 + 0.99247 w_k1_FDX_s105 + 1.282 w_k1_IBM_s105
                                 + 0.983786 w_k1_INTC_s105
                                 + 1.28038 w_k1_KO_s105 + 1.40758 w_k1_MAT_s105
                                 + 1.03158 w_k1_MSFT_s105
                                 + 1.05947 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 0.988098 w_k1_ADM_s105
                                 + 0.671983 w_k1_AXP_s105
                                 + 0.619064 w_k1_DIS_s105
                                 + 0.783138 w_k1_FDX_s105
                                 + 0.914824 w_k1_IBM_s105
                                 + 0.744145 w_k1_INTC_s105
                                 + 0.780757 w_k1_KO_s105
                                 + 0.900638 w_k1_MAT_s105
                                 + 0.789669 w_k1_MSFT_s105
                                 + 0.886441 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 1.02417 w_k1_ADM_s105
                                 + 1.10048 w_k1_AXP_s105
                                 + 0.764142 w_k1_DIS_s105
                                 + 1.00745 w_k1_FDX_s105
                                 + 0.951391 w_k1_IBM_s105
                                 + 1.1975 w_k1_INTC_s105
                                 + 0.929255 w_k1_KO_s105
                                 + 1.16432 w_k1_MAT_s105
                                 + 1.03017 w_k1_MSFT_s105
                                 + 0.939001 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.722901 w_k1_ADM_s105
                                 + 0.914043 w_k1_AXP_s105
                                 + 0.7285 w_k1_DIS_s105
                                 + 0.757339 w_k1_FDX_s105
                                 + 1.35253 w_k1_IBM_s105
                                 + 0.898571 w_k1_INTC_s105
                                 + 1.03278 w_k1_KO_s105 + 1.31002 w_k1_MAT_s105
                                 + 0.832755 w_k1_MSFT_s105
                                 + 0.884125 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 0.934052 w_k1_ADM_s105
                                 + 0.739418 w_k1_AXP_s105
                                 + 0.719117 w_k1_DIS_s105
                                 + 0.917246 w_k1_FDX_s105
                                 + 0.633334 w_k1_IBM_s105
                                 + 0.863391 w_k1_INTC_s105
                                 + 0.69037 w_k1_KO_s105
                                 + 0.790456 w_k1_MAT_s105
                                 + 0.895795 w_k1_MSFT_s105
                                 + 0.885419 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 0.719636 w_k1_ADM_s106
                                 + 0.739546 w_k1_AXP_s106
                                 + 0.693993 w_k1_DIS_s106
                                 + 0.743231 w_k1_FDX_s106
                                 + 0.892864 w_k1_IBM_s106
                                 + 0.758155 w_k1_INTC_s106
                                 + 0.833495 w_k1_KO_s106
                                 + 0.852098 w_k1_MAT_s106
                                 + 0.691668 w_k1_MSFT_s106
                                 + 0.88295 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 0.955522 w_k1_ADM_s106
                                 + 1.0639 w_k1_AXP_s106 + 1.25975 w_k1_DIS_s106
                                 + 1.08136 w_k1_FDX_s106
                                 + 1.06509 w_k1_IBM_s106
                                 + 1.16261 w_k1_INTC_s106
                                 + 1.19209 w_k1_KO_s106
                                 + 0.945321 w_k1_MAT_s106
                                 + 1.1234 w_k1_MSFT_s106 + 1.14833 w_k1_PG_s106
                                 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 0.738259 w_k1_ADM_s106
                                 + 0.60382 w_k1_AXP_s106
                                 + 0.632209 w_k1_DIS_s106
                                 + 0.705764 w_k1_FDX_s106
                                 + 0.700616 w_k1_IBM_s106
                                 + 0.765572 w_k1_INTC_s106
                                 + 0.718638 w_k1_KO_s106
                                 + 0.502061 w_k1_MAT_s106
                                 + 0.708561 w_k1_MSFT_s106
                                 + 0.799438 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 1.01161 w_k1_ADM_s106
                                 + 1.25074 w_k1_AXP_s106
                                 + 0.928064 w_k1_DIS_s106
                                 + 1.05074 w_k1_FDX_s106
                                 + 0.949763 w_k1_IBM_s106
                                 + 1.06676 w_k1_INTC_s106
                                 + 1.04558 w_k1_KO_s106 + 1.25868 w_k1_MAT_s106
                                 + 1.04685 w_k1_MSFT_s106
                                 + 1.10574 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 0.882436 w_k1_ADM_s106
                                 + 0.912471 w_k1_AXP_s106
                                 + 0.875481 w_k1_DIS_s106
                                 + 1.01259 w_k1_FDX_s106
                                 + 0.694438 w_k1_IBM_s106
                                 + 0.984979 w_k1_INTC_s106
                                 + 0.761002 w_k1_KO_s106
                                 + 0.502196 w_k1_MAT_s106
                                 + 0.997451 w_k1_MSFT_s106
                                 + 0.874978 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.849029 w_k1_ADM_s106
                                 + 0.740551 w_k1_AXP_s106
                                 + 0.827225 w_k1_DIS_s106
                                 + 0.826402 w_k1_FDX_s106
                                 + 1.03187 w_k1_IBM_s106
                                 + 0.782448 w_k1_INTC_s106
                                 + 0.996763 w_k1_KO_s106
                                 + 0.888346 w_k1_MAT_s106
                                 + 0.917663 w_k1_MSFT_s106
                                 + 0.927494 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.27839 w_k1_ADM_s106
                                 + 1.22081 w_k1_AXP_s106
                                 + 1.46259 w_k1_DIS_s106
                                 + 1.14727 w_k1_FDX_s106
                                 + 1.44843 w_k1_IBM_s106
                                 + 1.1017 w_k1_INTC_s106 + 1.41915 w_k1_KO_s106
                                 + 1.11738 w_k1_MAT_s106
                                 + 1.08401 w_k1_MSFT_s106 + 1.1832 w_k1_PG_s106
                                 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 0.840016 w_k1_ADM_s106
                                 + 0.690885 w_k1_AXP_s106
                                 + 0.806148 w_k1_DIS_s106
                                 + 0.781274 w_k1_FDX_s106
                                 + 1.43208 w_k1_IBM_s106
                                 + 0.837858 w_k1_INTC_s106
                                 + 1.14456 w_k1_KO_s106 + 1.3509 w_k1_MAT_s106
                                 + 0.845259 w_k1_MSFT_s106
                                 + 0.94089 w_k1_PG_s106 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 1.27946 w_k1_ADM_s107
                                 + 1.23147 w_k1_AXP_s107
                                 + 1.19089 w_k1_DIS_s107
                                 + 1.18596 w_k1_FDX_s107
                                 + 0.852877 w_k1_IBM_s107
                                 + 1.11604 w_k1_INTC_s107
                                 + 1.07937 w_k1_KO_s107 + 1.20903 w_k1_MAT_s107
                                 + 1.11205 w_k1_MSFT_s107
                                 + 1.08061 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.972641 w_k1_ADM_s107
                                 + 0.726138 w_k1_AXP_s107
                                 + 0.715653 w_k1_DIS_s107
                                 + 0.891482 w_k1_FDX_s107
                                 + 0.618732 w_k1_IBM_s107
                                 + 0.87824 w_k1_INTC_s107
                                 + 0.691083 w_k1_KO_s107
                                 + 0.649438 w_k1_MAT_s107
                                 + 0.910373 w_k1_MSFT_s107
                                 + 0.779018 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 0.919585 w_k1_ADM_s107
                                 + 0.83765 w_k1_AXP_s107
                                 + 0.856693 w_k1_DIS_s107
                                 + 0.790217 w_k1_FDX_s107
                                 + 1.13436 w_k1_IBM_s107
                                 + 0.80846 w_k1_INTC_s107
                                 + 1.00198 w_k1_KO_s107 + 1.37854 w_k1_MAT_s107
                                 + 0.904548 w_k1_MSFT_s107
                                 + 0.832429 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.905852 w_k1_ADM_s107
                                 + 1.04856 w_k1_AXP_s107
                                 + 1.10912 w_k1_DIS_s107
                                 + 1.05026 w_k1_FDX_s107
                                 + 1.14967 w_k1_IBM_s107
                                 + 1.08507 w_k1_INTC_s107
                                 + 1.11261 w_k1_KO_s107 + 1.54998 w_k1_MAT_s107
                                 + 1.12426 w_k1_MSFT_s107
                                 + 0.942816 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.1568 w_k1_ADM_s107
                                 + 1.01346 w_k1_AXP_s107
                                 + 0.893561 w_k1_DIS_s107
                                 + 1.03057 w_k1_FDX_s107
                                 + 0.72593 w_k1_IBM_s107
                                 + 0.989382 w_k1_INTC_s107
                                 + 0.899904 w_k1_KO_s107
                                 + 0.680773 w_k1_MAT_s107
                                 + 1.09811 w_k1_MSFT_s107
                                 + 0.973121 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 0.725703 w_k1_ADM_s107
                                 + 0.730694 w_k1_AXP_s107
                                 + 0.633663 w_k1_DIS_s107
                                 + 0.737848 w_k1_FDX_s107
                                 + 0.752482 w_k1_IBM_s107
                                 + 0.841074 w_k1_INTC_s107
                                 + 0.821477 w_k1_KO_s107
                                 + 0.984231 w_k1_MAT_s107
                                 + 0.705444 w_k1_MSFT_s107
                                 + 0.715698 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 0.795033 w_k1_ADM_s107
                                 + 0.802088 w_k1_AXP_s107
                                 + 1.04475 w_k1_DIS_s107
                                 + 0.878389 w_k1_FDX_s107
                                 + 0.822227 w_k1_IBM_s107
                                 + 1.13599 w_k1_INTC_s107
                                 + 1.05993 w_k1_KO_s107
                                 + 0.807423 w_k1_MAT_s107
                                 + 0.997404 w_k1_MSFT_s107
                                 + 0.880193 w_k1_PG_s107 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 1.26651 w_k1_ADM_s107
                                 + 1.50963 w_k1_AXP_s107
                                 + 1.63491 w_k1_DIS_s107 + 1.3997 w_k1_FDX_s107
                                 + 1.19985 w_k1_IBM_s107
                                 + 1.24739 w_k1_INTC_s107
                                 + 1.29466 w_k1_KO_s107 + 1.29514 w_k1_MAT_s107
                                 + 1.49976 w_k1_MSFT_s107
                                 + 1.12392 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 1.13181 w_k1_ADM_s110
                                 + 1.26319 w_k1_AXP_s110
                                 + 1.13818 w_k1_DIS_s110
                                 + 1.13336 w_k1_FDX_s110
                                 + 1.04019 w_k1_IBM_s110
                                 + 1.28328 w_k1_INTC_s110
                                 + 1.08177 w_k1_KO_s110
                                 + 0.878268 w_k1_MAT_s110
                                 + 1.17818 w_k1_MSFT_s110
                                 + 1.05398 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.15991 w_k1_ADM_s110
                                 + 1.15659 w_k1_AXP_s110
                                 + 1.21172 w_k1_DIS_s110
                                 + 1.26916 w_k1_FDX_s110
                                 + 0.888318 w_k1_IBM_s110
                                 + 1.17952 w_k1_INTC_s110
                                 + 0.984546 w_k1_KO_s110 + 1.052 w_k1_MAT_s110
                                 + 1.2387 w_k1_MSFT_s110 + 1.07594 w_k1_PG_s110
                                 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 1.09495 w_k1_ADM_s110
                                 + 0.936911 w_k1_AXP_s110
                                 + 0.868195 w_k1_DIS_s110
                                 + 0.930415 w_k1_FDX_s110
                                 + 1.2905 w_k1_IBM_s110
                                 + 1.12631 w_k1_INTC_s110
                                 + 1.23047 w_k1_KO_s110
                                 + 0.986204 w_k1_MAT_s110
                                 + 1.05407 w_k1_MSFT_s110 + 1.0117 w_k1_PG_s110
                                 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 1.14485 w_k1_ADM_s110
                                 + 1.35402 w_k1_AXP_s110
                                 + 1.46516 w_k1_DIS_s110
                                 + 1.41627 w_k1_FDX_s110 + 1.5035 w_k1_IBM_s110
                                 + 1.13564 w_k1_INTC_s110
                                 + 1.43638 w_k1_KO_s110 + 1.71545 w_k1_MAT_s110
                                 + 1.25007 w_k1_MSFT_s110
                                 + 1.17488 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.992803 w_k1_ADM_s110
                                 + 0.938948 w_k1_AXP_s110
                                 + 0.960635 w_k1_DIS_s110
                                 + 1.0227 w_k1_FDX_s110 + 1.05161 w_k1_IBM_s110
                                 + 1.04103 w_k1_INTC_s110
                                 + 1.06798 w_k1_KO_s110 + 0.96232 w_k1_MAT_s110
                                 + 1.14246 w_k1_MSFT_s110
                                 + 1.06264 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 1.06029 w_k1_ADM_s110
                                 + 1.02054 w_k1_AXP_s110
                                 + 0.89413 w_k1_DIS_s110
                                 + 0.910656 w_k1_FDX_s110
                                 + 1.65467 w_k1_IBM_s110
                                 + 1.04353 w_k1_INTC_s110
                                 + 1.22327 w_k1_KO_s110 + 1.28483 w_k1_MAT_s110
                                 + 1.04089 w_k1_MSFT_s110
                                 + 1.04259 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.03748 w_k1_ADM_s110
                                 + 0.723799 w_k1_AXP_s110
                                 + 0.747849 w_k1_DIS_s110
                                 + 0.877326 w_k1_FDX_s110
                                 + 0.973591 w_k1_IBM_s110
                                 + 1.02773 w_k1_INTC_s110
                                 + 0.959619 w_k1_KO_s110
                                 + 0.913416 w_k1_MAT_s110
                                 + 0.93874 w_k1_MSFT_s110
                                 + 0.965649 w_k1_PG_s110 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.951978 w_k1_ADM_s110
                                 + 0.751711 w_k1_AXP_s110
                                 + 0.636574 w_k1_DIS_s110
                                 + 0.851361 w_k1_FDX_s110
                                 + 0.71124 w_k1_IBM_s110
                                 + 0.902757 w_k1_INTC_s110
                                 + 0.793977 w_k1_KO_s110
                                 + 0.809335 w_k1_MAT_s110
                                 + 0.86483 w_k1_MSFT_s110
                                 + 0.831536 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.804919 w_k1_ADM_s111
                                 + 0.711869 w_k1_AXP_s111
                                 + 0.674822 w_k1_DIS_s111
                                 + 0.839857 w_k1_FDX_s111
                                 + 0.664146 w_k1_IBM_s111
                                 + 1.12866 w_k1_INTC_s111
                                 + 0.674308 w_k1_KO_s111
                                 + 0.475214 w_k1_MAT_s111
                                 + 0.840348 w_k1_MSFT_s111
                                 + 0.848543 w_k1_PG_s111 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 0.893337 w_k1_ADM_s111
                                 + 1.106 w_k1_AXP_s111 + 0.946046 w_k1_DIS_s111
                                 + 1.04003 w_k1_FDX_s111
                                 + 0.97907 w_k1_IBM_s111
                                 + 0.990243 w_k1_INTC_s111
                                 + 0.796904 w_k1_KO_s111
                                 + 0.911762 w_k1_MAT_s111
                                 + 0.913038 w_k1_MSFT_s111
                                 + 1.00416 w_k1_PG_s111 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.775062 w_k1_ADM_s111
                                 + 0.898781 w_k1_AXP_s111
                                 + 1.20631 w_k1_DIS_s111
                                 + 1.07592 w_k1_FDX_s111
                                 + 0.911405 w_k1_IBM_s111
                                 + 1.14027 w_k1_INTC_s111
                                 + 0.977728 w_k1_KO_s111
                                 + 0.568351 w_k1_MAT_s111
                                 + 1.09919 w_k1_MSFT_s111
                                 + 1.06211 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 0.938418 w_k1_ADM_s111
                                 + 0.910804 w_k1_AXP_s111
                                 + 0.852187 w_k1_DIS_s111
                                 + 0.842522 w_k1_FDX_s111
                                 + 1.18538 w_k1_IBM_s111
                                 + 1.09571 w_k1_INTC_s111
                                 + 0.889344 w_k1_KO_s111
                                 + 1.05922 w_k1_MAT_s111
                                 + 0.970823 w_k1_MSFT_s111
                                 + 0.950182 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 1.44878 w_k1_ADM_s111
                                 + 1.26577 w_k1_AXP_s111
                                 + 1.76798 w_k1_DIS_s111
                                 + 1.28716 w_k1_FDX_s111
                                 + 1.47372 w_k1_IBM_s111
                                 + 1.2703 w_k1_INTC_s111 + 1.39185 w_k1_KO_s111
                                 + 0.808747 w_k1_MAT_s111
                                 + 1.33376 w_k1_MSFT_s111
                                 + 1.19262 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 1.29957 w_k1_ADM_s111
                                 + 1.22905 w_k1_AXP_s111
                                 + 1.52416 w_k1_DIS_s111
                                 + 1.48199 w_k1_FDX_s111
                                 + 1.04803 w_k1_IBM_s111
                                 + 1.27934 w_k1_INTC_s111
                                 + 1.06995 w_k1_KO_s111 + 1.05707 w_k1_MAT_s111
                                 + 1.32176 w_k1_MSFT_s111
                                 + 1.13202 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.689824 w_k1_ADM_s111
                                 + 0.633502 w_k1_AXP_s111
                                 + 0.949347 w_k1_DIS_s111
                                 + 0.861997 w_k1_FDX_s111
                                 + 0.805319 w_k1_IBM_s111
                                 + 1.03827 w_k1_INTC_s111 + 0.7946 w_k1_KO_s111
                                 + 0.585993 w_k1_MAT_s111
                                 + 1.03912 w_k1_MSFT_s111
                                 + 0.843365 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 0.984544 w_k1_ADM_s111
                                 + 1.21566 w_k1_AXP_s111
                                 + 1.21846 w_k1_DIS_s111
                                 + 1.07724 w_k1_FDX_s111
                                 + 1.38506 w_k1_IBM_s111
                                 + 1.21125 w_k1_INTC_s111
                                 + 1.13715 w_k1_KO_s111 + 1.40089 w_k1_MAT_s111
                                 + 1.0525 w_k1_MSFT_s111 + 1.09657 w_k1_PG_s111
                                 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 1.05919 w_k1_ADM_s112
                                 + 0.690842 w_k1_AXP_s112
                                 + 0.701345 w_k1_DIS_s112
                                 + 0.69056 w_k1_FDX_s112
                                 + 0.999626 w_k1_IBM_s112
                                 + 0.848937 w_k1_INTC_s112
                                 + 0.91547 w_k1_KO_s112
                                 + 0.815083 w_k1_MAT_s112
                                 + 0.847145 w_k1_MSFT_s112
                                 + 0.783271 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.09389 w_k1_ADM_s112
                                 + 1.27188 w_k1_AXP_s112
                                 + 0.915264 w_k1_DIS_s112
                                 + 0.995785 w_k1_FDX_s112
                                 + 1.06614 w_k1_IBM_s112
                                 + 1.01737 w_k1_INTC_s112
                                 + 0.913644 w_k1_KO_s112
                                 + 1.26667 w_k1_MAT_s112
                                 + 0.975608 w_k1_MSFT_s112
                                 + 0.985436 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.17271 w_k1_ADM_s112
                                 + 1.22941 w_k1_AXP_s112
                                 + 1.49832 w_k1_DIS_s112
                                 + 1.13868 w_k1_FDX_s112
                                 + 1.61465 w_k1_IBM_s112
                                 + 1.06039 w_k1_INTC_s112 + 1.2875 w_k1_KO_s112
                                 + 1.64229 w_k1_MAT_s112
                                 + 1.28736 w_k1_MSFT_s112
                                 + 1.08657 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 1.36024 w_k1_ADM_s112
                                 + 1.01356 w_k1_AXP_s112
                                 + 1.22126 w_k1_DIS_s112
                                 + 1.18956 w_k1_FDX_s112
                                 + 0.961251 w_k1_IBM_s112
                                 + 0.974211 w_k1_INTC_s112
                                 + 1.04537 w_k1_KO_s112 + 1.10601 w_k1_MAT_s112
                                 + 1.20131 w_k1_MSFT_s112
                                 + 1.07259 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.876403 w_k1_ADM_s112
                                 + 0.544886 w_k1_AXP_s112
                                 + 0.529904 w_k1_DIS_s112
                                 + 0.687951 w_k1_FDX_s112
                                 + 0.73222 w_k1_IBM_s112
                                 + 0.703916 w_k1_INTC_s112
                                 + 0.734021 w_k1_KO_s112
                                 + 0.625749 w_k1_MAT_s112
                                 + 0.664804 w_k1_MSFT_s112
                                 + 0.745555 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 1.01112 w_k1_ADM_s112
                                 + 0.760795 w_k1_AXP_s112
                                 + 0.869291 w_k1_DIS_s112
                                 + 0.857705 w_k1_FDX_s112
                                 + 1.19027 w_k1_IBM_s112
                                 + 0.851511 w_k1_INTC_s112
                                 + 1.01199 w_k1_KO_s112 + 1.08095 w_k1_MAT_s112
                                 + 0.992491 w_k1_MSFT_s112
                                 + 0.927319 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.58001 w_k1_ADM_s112
                                 + 1.7101 w_k1_AXP_s112 + 1.27182 w_k1_DIS_s112
                                 + 1.21231 w_k1_FDX_s112
                                 + 1.50467 w_k1_IBM_s112
                                 + 1.01172 w_k1_INTC_s112
                                 + 1.37357 w_k1_KO_s112 + 2.12407 w_k1_MAT_s112
                                 + 1.24346 w_k1_MSFT_s112
                                 + 1.15588 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 1.19728 w_k1_ADM_s112
                                 + 0.803553 w_k1_AXP_s112
                                 + 0.767966 w_k1_DIS_s112
                                 + 0.901855 w_k1_FDX_s112
                                 + 0.682463 w_k1_IBM_s112
                                 + 0.853444 w_k1_INTC_s112
                                 + 0.742561 w_k1_KO_s112
                                 + 0.647655 w_k1_MAT_s112
                                 + 0.853919 w_k1_MSFT_s112
                                 + 0.808618 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 0.93213 w_k1_ADM_s113
                                 + 0.742809 w_k1_AXP_s113
                                 + 1.14821 w_k1_DIS_s113
                                 + 0.926727 w_k1_FDX_s113
                                 + 0.89874 w_k1_IBM_s113
                                 + 0.957487 w_k1_INTC_s113
                                 + 1.00901 w_k1_KO_s113
                                 + 0.676407 w_k1_MAT_s113
                                 + 1.07634 w_k1_MSFT_s113
                                 + 0.863969 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.944812 w_k1_ADM_s113
                                 + 1.11297 w_k1_AXP_s113 + 1.0335 w_k1_DIS_s113
                                 + 0.960497 w_k1_FDX_s113
                                 + 1.18386 w_k1_IBM_s113
                                 + 0.866238 w_k1_INTC_s113
                                 + 0.984922 w_k1_KO_s113
                                 + 1.29547 w_k1_MAT_s113
                                 + 0.999885 w_k1_MSFT_s113
                                 + 1.03035 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.09799 w_k1_ADM_s113
                                 + 1.03357 w_k1_AXP_s113
                                 + 1.27197 w_k1_DIS_s113
                                 + 1.15897 w_k1_FDX_s113
                                 + 0.738871 w_k1_IBM_s113
                                 + 1.10367 w_k1_INTC_s113
                                 + 1.04113 w_k1_KO_s113
                                 + 0.888761 w_k1_MAT_s113
                                 + 1.1877 w_k1_MSFT_s113
                                 + 0.933134 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 1.20814 w_k1_ADM_s113
                                 + 1.34095 w_k1_AXP_s113
                                 + 1.78273 w_k1_DIS_s113
                                 + 1.29146 w_k1_FDX_s113
                                 + 1.45539 w_k1_IBM_s113
                                 + 1.3873 w_k1_INTC_s113 + 1.63213 w_k1_KO_s113
                                 + 1.59585 w_k1_MAT_s113
                                 + 1.35927 w_k1_MSFT_s113
                                 + 1.14866 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.04413 w_k1_ADM_s113
                                 + 1.12179 w_k1_AXP_s113
                                 + 1.28735 w_k1_DIS_s113
                                 + 1.21831 w_k1_FDX_s113
                                 + 0.962809 w_k1_IBM_s113
                                 + 1.08503 w_k1_INTC_s113
                                 + 0.947625 w_k1_KO_s113
                                 + 1.20813 w_k1_MAT_s113
                                 + 1.08481 w_k1_MSFT_s113
                                 + 1.01091 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.841105 w_k1_ADM_s113
                                 + 0.94385 w_k1_AXP_s113
                                 + 0.785017 w_k1_DIS_s113
                                 + 0.861065 w_k1_FDX_s113
                                 + 0.756818 w_k1_IBM_s113
                                 + 0.954599 w_k1_INTC_s113
                                 + 0.733004 w_k1_KO_s113
                                 + 0.749934 w_k1_MAT_s113
                                 + 0.85675 w_k1_MSFT_s113
                                 + 0.759062 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 0.607014 w_k1_ADM_s113
                                 + 0.690785 w_k1_AXP_s113
                                 + 0.681462 w_k1_DIS_s113
                                 + 0.694778 w_k1_FDX_s113
                                 + 0.794665 w_k1_IBM_s113
                                 + 0.855575 w_k1_INTC_s113
                                 + 0.70525 w_k1_KO_s113
                                 + 0.726224 w_k1_MAT_s113
                                 + 0.787791 w_k1_MSFT_s113
                                 + 0.76037 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.878146 w_k1_ADM_s113
                                 + 0.837995 w_k1_AXP_s113
                                 + 0.899438 w_k1_DIS_s113
                                 + 0.797665 w_k1_FDX_s113
                                 + 1.34821 w_k1_IBM_s113
                                 + 0.909675 w_k1_INTC_s113
                                 + 1.08119 w_k1_KO_s113 + 1.213 w_k1_MAT_s113
                                 + 0.970996 w_k1_MSFT_s113
                                 + 0.95892 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 1.19513 w_k1_ADM_s114
                                 + 1.43035 w_k1_AXP_s114
                                 + 1.22268 w_k1_DIS_s114
                                 + 1.32756 w_k1_FDX_s114
                                 + 1.13092 w_k1_IBM_s114
                                 + 1.39327 w_k1_INTC_s114 + 1.3142 w_k1_KO_s114
                                 + 1.23211 w_k1_MAT_s114
                                 + 1.20192 w_k1_MSFT_s114
                                 + 1.17819 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.978363 w_k1_ADM_s114
                                 + 0.818786 w_k1_AXP_s114
                                 + 0.965271 w_k1_DIS_s114
                                 + 0.93799 w_k1_FDX_s114
                                 + 0.859148 w_k1_IBM_s114
                                 + 1.2007 w_k1_INTC_s114
                                 + 0.990723 w_k1_KO_s114
                                 + 0.719686 w_k1_MAT_s114
                                 + 1.13024 w_k1_MSFT_s114
                                 + 0.907911 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 1.13573 w_k1_ADM_s114
                                 + 1.03521 w_k1_AXP_s114
                                 + 1.11609 w_k1_DIS_s114
                                 + 1.11378 w_k1_FDX_s114
                                 + 1.05624 w_k1_IBM_s114
                                 + 1.21919 w_k1_INTC_s114
                                 + 1.10483 w_k1_KO_s114 + 0.81341 w_k1_MAT_s114
                                 + 1.02335 w_k1_MSFT_s114
                                 + 1.02854 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 0.713299 w_k1_ADM_s114
                                 + 0.697523 w_k1_AXP_s114
                                 + 0.52613 w_k1_DIS_s114
                                 + 0.77525 w_k1_FDX_s114
                                 + 0.736459 w_k1_IBM_s114
                                 + 1.05268 w_k1_INTC_s114
                                 + 0.723963 w_k1_KO_s114
                                 + 0.581619 w_k1_MAT_s114
                                 + 0.860977 w_k1_MSFT_s114
                                 + 0.705654 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 1.08232 w_k1_ADM_s114
                                 + 0.954947 w_k1_AXP_s114
                                 + 0.785179 w_k1_DIS_s114
                                 + 0.864364 w_k1_FDX_s114
                                 + 1.10257 w_k1_IBM_s114
                                 + 1.05864 w_k1_INTC_s114
                                 + 1.06796 w_k1_KO_s114 + 1.12448 w_k1_MAT_s114
                                 + 0.96322 w_k1_MSFT_s114
                                 + 0.977151 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.26247 w_k1_ADM_s114
                                 + 1.55937 w_k1_AXP_s114
                                 + 1.49716 w_k1_DIS_s114
                                 + 1.21518 w_k1_FDX_s114
                                 + 1.53008 w_k1_IBM_s114
                                 + 1.54699 w_k1_INTC_s114
                                 + 1.58762 w_k1_KO_s114 + 1.41577 w_k1_MAT_s114
                                 + 1.23092 w_k1_MSFT_s114
                                 + 1.10225 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.932166 w_k1_ADM_s114
                                 + 1.07359 w_k1_AXP_s114
                                 + 1.03907 w_k1_DIS_s114
                                 + 1.20314 w_k1_FDX_s114
                                 + 0.649288 w_k1_IBM_s114
                                 + 1.05795 w_k1_INTC_s114
                                 + 1.02859 w_k1_KO_s114
                                 + 0.837678 w_k1_MAT_s114
                                 + 1.16098 w_k1_MSFT_s114
                                 + 0.957642 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.821037 w_k1_ADM_s114
                                 + 0.708817 w_k1_AXP_s114
                                 + 0.761521 w_k1_DIS_s114
                                 + 0.748232 w_k1_FDX_s114
                                 + 1.0767 w_k1_IBM_s114
                                 + 0.893978 w_k1_INTC_s114
                                 + 1.02237 w_k1_KO_s114
                                 + 0.936804 w_k1_MAT_s114
                                 + 0.872657 w_k1_MSFT_s114
                                 + 0.880504 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.03992 w_k1_ADM_s115
                                 + 1.04105 w_k1_AXP_s115
                                 + 1.25414 w_k1_DIS_s115
                                 + 1.20002 w_k1_FDX_s115
                                 + 0.721293 w_k1_IBM_s115
                                 + 1.18063 w_k1_INTC_s115
                                 + 0.918336 w_k1_KO_s115
                                 + 0.71803 w_k1_MAT_s115
                                 + 1.07753 w_k1_MSFT_s115
                                 + 0.933638 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 0.930616 w_k1_ADM_s115
                                 + 0.794478 w_k1_AXP_s115
                                 + 0.968957 w_k1_DIS_s115
                                 + 0.941292 w_k1_FDX_s115
                                 + 1.05597 w_k1_IBM_s115
                                 + 0.861638 w_k1_INTC_s115
                                 + 0.968968 w_k1_KO_s115
                                 + 0.969638 w_k1_MAT_s115
                                 + 0.925644 w_k1_MSFT_s115
                                 + 0.924991 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 1.06763 w_k1_ADM_s115
                                 + 1.12826 w_k1_AXP_s115
                                 + 0.709854 w_k1_DIS_s115
                                 + 0.968433 w_k1_FDX_s115
                                 + 0.887926 w_k1_IBM_s115
                                 + 1.01162 w_k1_INTC_s115
                                 + 0.866443 w_k1_KO_s115 + 1.0133 w_k1_MAT_s115
                                 + 0.890684 w_k1_MSFT_s115
                                 + 1.0054 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 0.656287 w_k1_ADM_s115
                                 + 0.696819 w_k1_AXP_s115
                                 + 0.605907 w_k1_DIS_s115
                                 + 0.748714 w_k1_FDX_s115
                                 + 0.761316 w_k1_IBM_s115
                                 + 0.875088 w_k1_INTC_s115
                                 + 0.74221 w_k1_KO_s115 + 0.9427 w_k1_MAT_s115
                                 + 0.784752 w_k1_MSFT_s115
                                 + 0.867954 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 1.27415 w_k1_ADM_s115
                                 + 1.20049 w_k1_AXP_s115
                                 + 1.48463 w_k1_DIS_s115
                                 + 1.28885 w_k1_FDX_s115
                                 + 1.23082 w_k1_IBM_s115
                                 + 1.06973 w_k1_INTC_s115
                                 + 1.25825 w_k1_KO_s115 + 1.34107 w_k1_MAT_s115
                                 + 1.34009 w_k1_MSFT_s115
                                 + 1.07033 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.21565 w_k1_ADM_s115
                                 + 1.37632 w_k1_AXP_s115
                                 + 1.11116 w_k1_DIS_s115
                                 + 1.29343 w_k1_FDX_s115
                                 + 0.964141 w_k1_IBM_s115
                                 + 0.969097 w_k1_INTC_s115
                                 + 1.09631 w_k1_KO_s115 + 1.4438 w_k1_MAT_s115
                                 + 1.15765 w_k1_MSFT_s115
                                 + 1.07262 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.01651 w_k1_ADM_s115
                                 + 0.871779 w_k1_AXP_s115
                                 + 1.16455 w_k1_DIS_s115
                                 + 1.09575 w_k1_FDX_s115
                                 + 1.06678 w_k1_IBM_s115
                                 + 0.809475 w_k1_INTC_s115
                                 + 1.20581 w_k1_KO_s115 + 1.31332 w_k1_MAT_s115
                                 + 1.12457 w_k1_MSFT_s115
                                 + 1.04965 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.31427 w_k1_ADM_s115
                                 + 1.12506 w_k1_AXP_s115
                                 + 1.76265 w_k1_DIS_s115
                                 + 1.38928 w_k1_FDX_s115 + 1.4177 w_k1_IBM_s115
                                 + 1.0612 w_k1_INTC_s115 + 1.44285 w_k1_KO_s115
                                 + 1.51432 w_k1_MAT_s115
                                 + 1.38623 w_k1_MSFT_s115
                                 + 1.21964 w_k1_PG_s115 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 0.706751 w_k1_ADM_s116
                                 + 0.816909 w_k1_AXP_s116
                                 + 0.634932 w_k1_DIS_s116
                                 + 0.704135 w_k1_FDX_s116
                                 + 0.797102 w_k1_IBM_s116
                                 + 0.659036 w_k1_INTC_s116
                                 + 0.728056 w_k1_KO_s116
                                 + 1.24063 w_k1_MAT_s116
                                 + 0.759814 w_k1_MSFT_s116
                                 + 0.798086 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 0.846957 w_k1_ADM_s116
                                 + 0.715186 w_k1_AXP_s116
                                 + 0.762003 w_k1_DIS_s116
                                 + 0.938788 w_k1_FDX_s116
                                 + 0.702696 w_k1_IBM_s116
                                 + 0.730971 w_k1_INTC_s116
                                 + 0.780423 w_k1_KO_s116
                                 + 0.733702 w_k1_MAT_s116
                                 + 0.932389 w_k1_MSFT_s116
                                 + 0.853932 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 1.02955 w_k1_ADM_s116
                                 + 0.906538 w_k1_AXP_s116
                                 + 1.35177 w_k1_DIS_s116
                                 + 1.01221 w_k1_FDX_s116
                                 + 1.13352 w_k1_IBM_s116
                                 + 0.94923 w_k1_INTC_s116
                                 + 1.11424 w_k1_KO_s116 + 1.47011 w_k1_MAT_s116
                                 + 1.28238 w_k1_MSFT_s116
                                 + 1.01363 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 0.940794 w_k1_ADM_s116
                                 + 0.973759 w_k1_AXP_s116
                                 + 0.833646 w_k1_DIS_s116
                                 + 0.884639 w_k1_FDX_s116
                                 + 0.959521 w_k1_IBM_s116
                                 + 0.706594 w_k1_INTC_s116
                                 + 0.771777 w_k1_KO_s116 + 1.22 w_k1_MAT_s116
                                 + 0.778161 w_k1_MSFT_s116
                                 + 0.953517 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 1.50186 w_k1_ADM_s116
                                 + 1.41784 w_k1_AXP_s116
                                 + 1.45364 w_k1_DIS_s116
                                 + 1.34927 w_k1_FDX_s116
                                 + 1.17628 w_k1_IBM_s116
                                 + 1.04626 w_k1_INTC_s116
                                 + 1.16789 w_k1_KO_s116 + 2.05247 w_k1_MAT_s116
                                 + 1.36784 w_k1_MSFT_s116
                                 + 1.26571 w_k1_PG_s116 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.824639 w_k1_ADM_s116
                                 + 0.762036 w_k1_AXP_s116
                                 + 0.957555 w_k1_DIS_s116
                                 + 0.985196 w_k1_FDX_s116
                                 + 0.89879 w_k1_IBM_s116
                                 + 0.889967 w_k1_INTC_s116
                                 + 0.853939 w_k1_KO_s116
                                 + 1.23025 w_k1_MAT_s116
                                 + 1.09391 w_k1_MSFT_s116
                                 + 1.01039 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.067 w_k1_ADM_s116
                                 + 1.09234 w_k1_AXP_s116
                                 + 1.03275 w_k1_DIS_s116
                                 + 1.11766 w_k1_FDX_s116
                                 + 0.908455 w_k1_IBM_s116
                                 + 0.955903 w_k1_INTC_s116
                                 + 0.907314 w_k1_KO_s116
                                 + 0.964749 w_k1_MAT_s116
                                 + 1.11852 w_k1_MSFT_s116
                                 + 0.969754 w_k1_PG_s116 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 0.947131 w_k1_ADM_s116
                                 + 1.20939 w_k1_AXP_s116
                                 + 1.00526 w_k1_DIS_s116
                                 + 0.941424 w_k1_FDX_s116
                                 + 1.45244 w_k1_IBM_s116
                                 + 0.996592 w_k1_INTC_s116
                                 + 1.15988 w_k1_KO_s116 + 1.63872 w_k1_MAT_s116
                                 + 1.01658 w_k1_MSFT_s116
                                 + 1.12349 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.693959 w_k1_ADM_s117
                                 + 0.590081 w_k1_AXP_s117
                                 + 0.661066 w_k1_DIS_s117
                                 + 0.769177 w_k1_FDX_s117
                                 + 0.681365 w_k1_IBM_s117
                                 + 0.848319 w_k1_INTC_s117
                                 + 0.713835 w_k1_KO_s117
                                 + 0.680535 w_k1_MAT_s117
                                 + 0.798769 w_k1_MSFT_s117
                                 + 0.84603 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 1.05878 w_k1_ADM_s117
                                 + 1.15763 w_k1_AXP_s117
                                 + 0.950589 w_k1_DIS_s117
                                 + 1.01394 w_k1_FDX_s117
                                 + 1.11644 w_k1_IBM_s117
                                 + 0.985535 w_k1_INTC_s117
                                 + 1.10535 w_k1_KO_s117 + 1.23941 w_k1_MAT_s117
                                 + 0.94708 w_k1_MSFT_s117
                                 + 1.03703 w_k1_PG_s117 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.12687 w_k1_ADM_s117
                                 + 1.05148 w_k1_AXP_s117
                                 + 1.36998 w_k1_DIS_s117
                                 + 1.12001 w_k1_FDX_s117
                                 + 1.33233 w_k1_IBM_s117
                                 + 1.22431 w_k1_INTC_s117
                                 + 1.28524 w_k1_KO_s117 + 1.22766 w_k1_MAT_s117
                                 + 1.08858 w_k1_MSFT_s117
                                 + 1.06993 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 1.11225 w_k1_ADM_s117
                                 + 1.42708 w_k1_AXP_s117
                                 + 1.63697 w_k1_DIS_s117
                                 + 1.39373 w_k1_FDX_s117
                                 + 1.53908 w_k1_IBM_s117
                                 + 1.26432 w_k1_INTC_s117
                                 + 1.31077 w_k1_KO_s117 + 1.52238 w_k1_MAT_s117
                                 + 1.24075 w_k1_MSFT_s117
                                 + 1.20988 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 1.01585 w_k1_ADM_s117
                                 + 1.49344 w_k1_AXP_s117
                                 + 1.14695 w_k1_DIS_s117
                                 + 1.00153 w_k1_FDX_s117
                                 + 1.62936 w_k1_IBM_s117
                                 + 1.29934 w_k1_INTC_s117
                                 + 1.18996 w_k1_KO_s117 + 1.63926 w_k1_MAT_s117
                                 + 0.951914 w_k1_MSFT_s117
                                 + 1.10981 w_k1_PG_s117 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.870445 w_k1_ADM_s117
                                 + 0.840849 w_k1_AXP_s117
                                 + 0.844414 w_k1_DIS_s117
                                 + 0.884472 w_k1_FDX_s117
                                 + 1.01061 w_k1_IBM_s117
                                 + 1.08581 w_k1_INTC_s117
                                 + 1.00242 w_k1_KO_s117
                                 + 0.862769 w_k1_MAT_s117
                                 + 0.851926 w_k1_MSFT_s117
                                 + 0.934423 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.01534 w_k1_ADM_s117
                                 + 1.0626 w_k1_AXP_s117 + 1.38113 w_k1_DIS_s117
                                 + 1.17817 w_k1_FDX_s117
                                 + 0.906602 w_k1_IBM_s117
                                 + 1.21172 w_k1_INTC_s117
                                 + 1.03695 w_k1_KO_s117 + 0.77069 w_k1_MAT_s117
                                 + 1.16698 w_k1_MSFT_s117
                                 + 1.07557 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.8005 w_k1_ADM_s117
                                 + 0.868541 w_k1_AXP_s117
                                 + 0.959432 w_k1_DIS_s117
                                 + 0.898857 w_k1_FDX_s117
                                 + 1.36153 w_k1_IBM_s117
                                 + 1.03526 w_k1_INTC_s117
                                 + 1.24853 w_k1_KO_s117 + 1.25695 w_k1_MAT_s117
                                 + 1.06066 w_k1_MSFT_s117
                                 + 1.04872 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0369175661183973 w_k1_ADM_s000
                                 + 0.0177962483947762 w_k1_ADM_s001
                                 + 0.0107392634852787 w_k1_ADM_s002
                                 + 0.0389343638583354 w_k1_ADM_s003
                                 + 0.0132904988323322 w_k1_ADM_s004
                                 + 0.0171557282403976 w_k1_ADM_s005
                                 + 0.0155565222106427 w_k1_ADM_s006
                                 + 0.0221137303771126 w_k1_ADM_s007
                                 + 0.0105577641779303 w_k1_ADM_s010
                                 + 0.00733117742119985 w_k1_ADM_s011
                                 + 0.0161382490939659 w_k1_ADM_s012
                                 + 0.0100243153747723 w_k1_ADM_s013
                                 + 0.0141500720162015 w_k1_ADM_s014
                                 + 0.00776873317731447 w_k1_ADM_s015
                                 + 0.00586586690376743 w_k1_ADM_s016
                                 + 0.0167713790608249 w_k1_ADM_s017
                                 + 0.0334753815741529 w_k1_ADM_s100
                                 + 0.0626225808705528 w_k1_ADM_s101
                                 + 0.0292578359304944 w_k1_ADM_s102
                                 + 0.109197356145899 w_k1_ADM_s103
                                 + 0.0264533637260228 w_k1_ADM_s104
                                 + 0.0360664907430581 w_k1_ADM_s105
                                 + 0.0516885438662177 w_k1_ADM_s106
                                 + 0.0231298870421095 w_k1_ADM_s107
                                 + 0.0662547963350966 w_k1_ADM_s110
                                 + 0.0162751809130715 w_k1_ADM_s111
                                 + 0.0835045095783203 w_k1_ADM_s112
                                 + 0.0367896995188743 w_k1_ADM_s113
                                 + 0.0505386880675851 w_k1_ADM_s114
                                 + 0.0538083628150595 w_k1_ADM_s115
                                 + 0.0277030355861171 w_k1_ADM_s116
                                 + 0.0473788070864167 w_k1_ADM_s117
                                 + 0.0334918492596149 w_k1_AXP_s000
                                 + 0.0199667359282357 w_k1_AXP_s001
                                 + 0.0118868596142262 w_k1_AXP_s002
                                 + 0.0447358027167767 w_k1_AXP_s003
                                 + 0.0126425243186518 w_k1_AXP_s004
                                 + 0.0163385320376348 w_k1_AXP_s005
                                 + 0.0181737132117853 w_k1_AXP_s006
                                 + 0.0216128671913627 w_k1_AXP_s007
                                 + 0.00916134756092872 w_k1_AXP_s010
                                 + 0.0066333981312964 w_k1_AXP_s011
                                 + 0.0152685196923846 w_k1_AXP_s012
                                 + 0.0105264896109182 w_k1_AXP_s013
                                 + 0.0163361577349639 w_k1_AXP_s014
                                 + 0.00777960567338645 w_k1_AXP_s015
                                 + 0.00678000349195709 w_k1_AXP_s016
                                 + 0.0174233520670574 w_k1_AXP_s017
                                 + 0.0399860603142441 w_k1_AXP_s100
                                 + 0.0692547874529912 w_k1_AXP_s101
                                 + 0.031957493826566 w_k1_AXP_s102
                                 + 0.104114095877527 w_k1_AXP_s103
                                 + 0.0254753243743534 w_k1_AXP_s104
                                 + 0.0341345662759551 w_k1_AXP_s105
                                 + 0.0531897832824826 w_k1_AXP_s106
                                 + 0.0234282293755952 w_k1_AXP_s107
                                 + 0.0624809415165973 w_k1_AXP_s110
                                 + 0.0170515800418206 w_k1_AXP_s111
                                 + 0.0672254842228215 w_k1_AXP_s112
                                 + 0.0389262265973645 w_k1_AXP_s113
                                 + 0.050403167301542 w_k1_AXP_s114
                                 + 0.050640201729423 w_k1_AXP_s115
                                 + 0.0276564842920284 w_k1_AXP_s116
                                 + 0.0488887562493302 w_k1_AXP_s117
                                 + 0.0330235459505448 w_k1_DIS_s000
                                 + 0.0189413324963584 w_k1_DIS_s001
                                 + 0.012399649787681 w_k1_DIS_s002
                                 + 0.0435548869317059 w_k1_DIS_s003
                                 + 0.0122749763293509 w_k1_DIS_s004
                                 + 0.0166601248905754 w_k1_DIS_s005
                                 + 0.0189046967749193 w_k1_DIS_s006
                                 + 0.0246178074246422 w_k1_DIS_s007
                                 + 0.00952722462327566 w_k1_DIS_s010
                                 + 0.00680017004633924 w_k1_DIS_s011
                                 + 0.0155046582852282 w_k1_DIS_s012
                                 + 0.0102020527938694 w_k1_DIS_s013
                                 + 0.0166759078945312 w_k1_DIS_s014
                                 + 0.00812253169592694 w_k1_DIS_s015
                                 + 0.00598756448926976 w_k1_DIS_s016
                                 + 0.0165528684992375 w_k1_DIS_s017
                                 + 0.0388407758819566 w_k1_DIS_s100
                                 + 0.0671116106452229 w_k1_DIS_s101
                                 + 0.0292358645516244 w_k1_DIS_s102
                                 + 0.10790409861948 w_k1_DIS_s103
                                 + 0.0263255016232721 w_k1_DIS_s104
                                 + 0.0348155318036837 w_k1_DIS_s105
                                 + 0.055722274968148 w_k1_DIS_s106
                                 + 0.0239523472400784 w_k1_DIS_s107
                                 + 0.059990602489667 w_k1_DIS_s110
                                 + 0.0189068290030834 w_k1_DIS_s111
                                 + 0.0695250065543045 w_k1_DIS_s112
                                 + 0.043685239264519 w_k1_DIS_s113
                                 + 0.0479345320258334 w_k1_DIS_s114
                                 + 0.056677675622093 w_k1_DIS_s115
                                 + 0.0279694853482335 w_k1_DIS_s116
                                 + 0.0533863812420315 w_k1_DIS_s117
                                 + 0.0330370360016075 w_k1_FDX_s000
                                 + 0.0189481693160998 w_k1_FDX_s001
                                 + 0.0123832572365113 w_k1_FDX_s002
                                 + 0.0421127349784224 w_k1_FDX_s003
                                 + 0.0122883069102668 w_k1_FDX_s004
                                 + 0.0174815497283884 w_k1_FDX_s005
                                 + 0.0178034793753642 w_k1_FDX_s006
                                 + 0.0220419192969925 w_k1_FDX_s007
                                 + 0.00963335410914859 w_k1_FDX_s010
                                 + 0.00644391063413493 w_k1_FDX_s011
                                 + 0.0155215978041226 w_k1_FDX_s012
                                 + 0.0103730925004851 w_k1_FDX_s013
                                 + 0.016806694133571 w_k1_FDX_s014
                                 + 0.00746511195599607 w_k1_FDX_s015
                                 + 0.00599575596253508 w_k1_FDX_s016
                                 + 0.0169916100241823 w_k1_FDX_s017
                                 + 0.0368128103106394 w_k1_FDX_s100
                                 + 0.0630476895499698 w_k1_FDX_s101
                                 + 0.0300697447878151 w_k1_FDX_s102
                                 + 0.100008954461718 w_k1_FDX_s103
                                 + 0.0271019814632708 w_k1_FDX_s104
                                 + 0.0358928460525837 w_k1_FDX_s105
                                 + 0.0528775505957485 w_k1_FDX_s106
                                 + 0.023443870459438 w_k1_FDX_s107
                                 + 0.0652026794681345 w_k1_FDX_s110
                                 + 0.0181451740343783 w_k1_FDX_s111
                                 + 0.0689184676722462 w_k1_FDX_s112
                                 + 0.0389152126447256 w_k1_FDX_s113
                                 + 0.0500486207172725 w_k1_FDX_s114
                                 + 0.0561867539092982 w_k1_FDX_s115
                                 + 0.0275527562926692 w_k1_FDX_s116
                                 + 0.0510941051331614 w_k1_FDX_s117
                                 + 0.0333041162145141 w_k1_IBM_s000
                                 + 0.0187013085999602 w_k1_IBM_s001
                                 + 0.0125576398046368 w_k1_IBM_s002
                                 + 0.044974291439176 w_k1_IBM_s003
                                 + 0.01244458583534 w_k1_IBM_s004
                                 + 0.0169498995719614 w_k1_IBM_s005
                                 + 0.0185695530642876 w_k1_IBM_s006
                                 + 0.0225411431803834 w_k1_IBM_s007
                                 + 0.0100976401148154 w_k1_IBM_s010
                                 + 0.00610748054602655 w_k1_IBM_s011
                                 + 0.0165311718729072 w_k1_IBM_s012
                                 + 0.0103065455598118 w_k1_IBM_s013
                                 + 0.0170150986372783 w_k1_IBM_s014
                                 + 0.00757078942604171 w_k1_IBM_s015
                                 + 0.00617875723647886 w_k1_IBM_s016
                                 + 0.01660647181625 w_k1_IBM_s017
                                 + 0.0369648201305346 w_k1_IBM_s100
                                 + 0.0677333591014161 w_k1_IBM_s101
                                 + 0.0311436911382898 w_k1_IBM_s102
                                 + 0.0968856099656823 w_k1_IBM_s103
                                 + 0.0260129238656886 w_k1_IBM_s104
                                 + 0.0354439384991449 w_k1_IBM_s105
                                 + 0.0571418882875657 w_k1_IBM_s106
                                 + 0.0221250385628616 w_k1_IBM_s107
                                 + 0.0650129179487478 w_k1_IBM_s110
                                 + 0.0182358639444452 w_k1_IBM_s111
                                 + 0.0735951521522263 w_k1_IBM_s112
                                 + 0.0393894783638036 w_k1_IBM_s113
                                 + 0.0494982461375726 w_k1_IBM_s114
                                 + 0.0537527012681778 w_k1_IBM_s115
                                 + 0.0275182429562517 w_k1_IBM_s116
                                 + 0.0550660263687225 w_k1_IBM_s117
                                 + 0.0325227289172528 w_k1_INTC_s000
                                 + 0.0199270980333205 w_k1_INTC_s001
                                 + 0.0133001396691281 w_k1_INTC_s002
                                 + 0.0401397829183883 w_k1_INTC_s003
                                 + 0.013162063855614 w_k1_INTC_s004
                                 + 0.0169650677878039 w_k1_INTC_s005
                                 + 0.0184092907400649 w_k1_INTC_s006
                                 + 0.0199722041847264 w_k1_INTC_s007
                                 + 0.00980836257679673 w_k1_INTC_s010
                                 + 0.00528597314261715 w_k1_INTC_s011
                                 + 0.015016094714563 w_k1_INTC_s012
                                 + 0.0100324095729469 w_k1_INTC_s013
                                 + 0.0160628919451744 w_k1_INTC_s014
                                 + 0.00785947245270184 w_k1_INTC_s015
                                 + 0.00533330466199345 w_k1_INTC_s016
                                 + 0.0172106234629146 w_k1_INTC_s017
                                 + 0.0350575844312753 w_k1_INTC_s100
                                 + 0.0650388140155465 w_k1_INTC_s101
                                 + 0.027841998347737 w_k1_INTC_s102
                                 + 0.0933271709456659 w_k1_INTC_s103
                                 + 0.0214316093641727 w_k1_INTC_s104
                                 + 0.0366573501771958 w_k1_INTC_s105
                                 + 0.0535924713248294 w_k1_INTC_s106
                                 + 0.0235765402576032 w_k1_INTC_s107
                                 + 0.0671315722903421 w_k1_INTC_s110
                                 + 0.0203527959901273 w_k1_INTC_s111
                                 + 0.0647700125345171 w_k1_INTC_s112
                                 + 0.0381988860336874 w_k1_INTC_s113
                                 + 0.0576899943518945 w_k1_INTC_s114
                                 + 0.0500945169908336 w_k1_INTC_s115
                                 + 0.0237538786210035 w_k1_INTC_s116
                                 + 0.0562514643249091 w_k1_INTC_s117
                                 + 0.0318257585173516 w_k1_KO_s000
                                 + 0.0179690970914279 w_k1_KO_s001
                                 + 0.0119235705393711 w_k1_KO_s002
                                 + 0.0475369934220072 w_k1_KO_s003
                                 + 0.0125578248967797 w_k1_KO_s004
                                 + 0.0164546081587862 w_k1_KO_s005
                                 + 0.0175184788815427 w_k1_KO_s006
                                 + 0.0230265368537214 w_k1_KO_s007
                                 + 0.0100514077314589 w_k1_KO_s010
                                 + 0.00625116514639055 w_k1_KO_s011
                                 + 0.01522592062163 w_k1_KO_s012
                                 + 0.00968969551206576 w_k1_KO_s013
                                 + 0.0166491933269025 w_k1_KO_s014
                                 + 0.00751561127820051 w_k1_KO_s015
                                 + 0.00579661400789335 w_k1_KO_s016
                                 + 0.0165908328946546 w_k1_KO_s017
                                 + 0.0418020951601369 w_k1_KO_s100
                                 + 0.0619447627177474 w_k1_KO_s101
                                 + 0.032661092386941 w_k1_KO_s102
                                 + 0.116994838980983 w_k1_KO_s103
                                 + 0.0260511221281465 w_k1_KO_s104
                                 + 0.0368966456888169 w_k1_KO_s105
                                 + 0.0583085522136959 w_k1_KO_s106
                                 + 0.0235017340933914 w_k1_KO_s107
                                 + 0.0648787670611105 w_k1_KO_s110
                                 + 0.0165544489661728 w_k1_KO_s111
                                 + 0.069539735097638 w_k1_KO_s112
                                 + 0.0387517161198573 w_k1_KO_s113
                                 + 0.0531120239537789 w_k1_KO_s114
                                 + 0.0551306805472641 w_k1_KO_s115
                                 + 0.0257600054465172 w_k1_KO_s116
                                 + 0.0534604389423049 w_k1_KO_s117
                                 + 0.036051160322179 w_k1_MAT_s000
                                 + 0.0268758621486863 w_k1_MAT_s001
                                 + 0.00934269540268972 w_k1_MAT_s002
                                 + 0.0536139851028768 w_k1_MAT_s003
                                 + 0.0112172937458284 w_k1_MAT_s004
                                 + 0.0182231029415343 w_k1_MAT_s005
                                 + 0.0200323644561574 w_k1_MAT_s006
                                 + 0.0227429569377269 w_k1_MAT_s007
                                 + 0.0120893258907888 w_k1_MAT_s010
                                 + 0.00667841432890743 w_k1_MAT_s011
                                 + 0.0172170895658009 w_k1_MAT_s012
                                 + 0.00831202451918834 w_k1_MAT_s013
                                 + 0.014219653633634 w_k1_MAT_s014
                                 + 0.00613890052250471 w_k1_MAT_s015
                                 + 0.00655067136630051 w_k1_MAT_s016
                                 + 0.0158915964288542 w_k1_MAT_s017
                                 + 0.0366374120473587 w_k1_MAT_s100
                                 + 0.0706415282468798 w_k1_MAT_s101
                                 + 0.0356509028395462 w_k1_MAT_s102
                                 + 0.112247374847341 w_k1_MAT_s103
                                 + 0.03037915935731 w_k1_MAT_s104
                                 + 0.0413804235836162 w_k1_MAT_s105
                                 + 0.0523158894673225 w_k1_MAT_s106
                                 + 0.0268509381641211 w_k1_MAT_s107
                                 + 0.0642361939520643 w_k1_MAT_s110
                                 + 0.0153351268758477 w_k1_MAT_s111
                                 + 0.0762487702849634 w_k1_MAT_s112
                                 + 0.041987256546036 w_k1_MAT_s113
                                 + 0.0458374221183249 w_k1_MAT_s114
                                 + 0.0594640963137625 w_k1_MAT_s115
                                 + 0.0361567715434544 w_k1_MAT_s116
                                 + 0.0515573940394181 w_k1_MAT_s117
                                 + 0.0305595262921984 w_k1_MSFT_s000
                                 + 0.0177434911390835 w_k1_MSFT_s001
                                 + 0.012717445257629 w_k1_MSFT_s002
                                 + 0.043933138685616 w_k1_MSFT_s003
                                 + 0.0132539790808439 w_k1_MSFT_s004
                                 + 0.0168727138423491 w_k1_MSFT_s005
                                 + 0.0161515667707789 w_k1_MSFT_s006
                                 + 0.0222401165432345 w_k1_MSFT_s007
                                 + 0.00985106373823179 w_k1_MSFT_s010
                                 + 0.00635055444774264 w_k1_MSFT_s011
                                 + 0.0144407685072417 w_k1_MSFT_s012
                                 + 0.00952443826087036 w_k1_MSFT_s013
                                 + 0.0173173057210283 w_k1_MSFT_s014
                                 + 0.00755986875093724 w_k1_MSFT_s015
                                 + 0.00546467512598757 w_k1_MSFT_s016
                                 + 0.0166848075189894 w_k1_MSFT_s017
                                 + 0.0408237032668666 w_k1_MSFT_s100
                                 + 0.0630777389613094 w_k1_MSFT_s101
                                 + 0.0298219218570217 w_k1_MSFT_s102
                                 + 0.099732433018203 w_k1_MSFT_s103
                                 + 0.0240221954340188 w_k1_MSFT_s104
                                 + 0.0372964065606484 w_k1_MSFT_s105
                                 + 0.05352718728173 w_k1_MSFT_s106
                                 + 0.0246951239681424 w_k1_MSFT_s107
                                 + 0.0665088931407291 w_k1_MSFT_s110
                                 + 0.0184613703925042 w_k1_MSFT_s111
                                 + 0.0718956530689711 w_k1_MSFT_s112
                                 + 0.0399469088198267 w_k1_MSFT_s113
                                 + 0.0507556309594308 w_k1_MSFT_s114
                                 + 0.0553678407325263 w_k1_MSFT_s115
                                 + 0.0286922368422667 w_k1_MSFT_s116
                                 + 0.0504669095706738 w_k1_MSFT_s117
                                 + 0.0345925234591442 w_k1_PG_s000
                                 + 0.018636856929479 w_k1_PG_s001
                                 + 0.0116458721020407 w_k1_PG_s002
                                 + 0.0441264380122033 w_k1_PG_s003
                                 + 0.0124229115301557 w_k1_PG_s004
                                 + 0.0157462732887675 w_k1_PG_s005
                                 + 0.0164534230262346 w_k1_PG_s006
                                 + 0.0228910942242436 w_k1_PG_s007
                                 + 0.00965220709301224 w_k1_PG_s010
                                 + 0.00592931340722393 w_k1_PG_s011
                                 + 0.0157706005711026 w_k1_PG_s012
                                 + 0.00954858545946901 w_k1_PG_s013
                                 + 0.0172636242592732 w_k1_PG_s014
                                 + 0.00789684477589625 w_k1_PG_s015
                                 + 0.00556896383683116 w_k1_PG_s016
                                 + 0.0152343350893926 w_k1_PG_s017
                                 + 0.0368029644208547 w_k1_PG_s100
                                 + 0.0665579940858547 w_k1_PG_s101
                                 + 0.0316330994939794 w_k1_PG_s102
                                 + 0.0999425507751306 w_k1_PG_s103
                                 + 0.026470004981715 w_k1_PG_s104
                                 + 0.0353634504602242 w_k1_PG_s105
                                 + 0.0561049889319252 w_k1_PG_s106
                                 + 0.0213927027879188 w_k1_PG_s107
                                 + 0.0623307339648737 w_k1_PG_s110
                                 + 0.0177942881979173 w_k1_PG_s111
                                 + 0.0669052307744298 w_k1_PG_s112
                                 + 0.0364966253261456 w_k1_PG_s113
                                 + 0.0474498198766219 w_k1_PG_s114
                                 + 0.0529720504942024 w_k1_PG_s115
                                 + 0.0273947144611755 w_k1_PG_s116
                                 + 0.051927213442527 w_k1_PG_s117 >= 1.2
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
