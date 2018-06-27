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
 w_k1_ADM_s0_return_constraint:  1.14255 w_0_ADM - w_k1_ADM_s0
                                 + 0.99 b_k1_ADM_s0 - 1.01 s_k1_ADM_s0  = 0
 w_k1_ADM_s1_return_constraint:  0.926826 w_0_ADM - w_k1_ADM_s1
                                 + 0.99 b_k1_ADM_s1 - 1.01 s_k1_ADM_s1  = 0
 w_k1_ADM_s00_return_constraint: 1.0653 w_k1_ADM_s0 - w_k1_ADM_s00
                                 + 0.99 b_k1_ADM_s00 - 1.01 s_k1_ADM_s00  = 0
 w_k1_ADM_s01_return_constraint: 0.84879 w_k1_ADM_s0 - w_k1_ADM_s01
                                 + 0.99 b_k1_ADM_s01 - 1.01 s_k1_ADM_s01  = 0
 w_k1_ADM_s10_return_constraint: 0.941489 w_k1_ADM_s1 - w_k1_ADM_s10
                                 + 0.99 b_k1_ADM_s10 - 1.01 s_k1_ADM_s10  = 0
 w_k1_ADM_s11_return_constraint: 1.1933 w_k1_ADM_s1 - w_k1_ADM_s11
                                 + 0.99 b_k1_ADM_s11 - 1.01 s_k1_ADM_s11  = 0
 w_k1_ADM_s000_return_constraint: 1.11702 w_k1_ADM_s00 - w_k1_ADM_s000
                                 + 0.99 b_k1_ADM_s000 - 1.01 s_k1_ADM_s000  = 0
 w_k1_ADM_s001_return_constraint: 1.64777 w_k1_ADM_s00 - w_k1_ADM_s001
                                 + 0.99 b_k1_ADM_s001 - 1.01 s_k1_ADM_s001  = 0
 w_k1_ADM_s002_return_constraint: 0.99839 w_k1_ADM_s00 - w_k1_ADM_s002
                                 + 0.99 b_k1_ADM_s002 - 1.01 s_k1_ADM_s002  = 0
 w_k1_ADM_s003_return_constraint: 1.20896 w_k1_ADM_s00 - w_k1_ADM_s003
                                 + 0.99 b_k1_ADM_s003 - 1.01 s_k1_ADM_s003  = 0
 w_k1_ADM_s004_return_constraint: 1.22972 w_k1_ADM_s00 - w_k1_ADM_s004
                                 + 0.99 b_k1_ADM_s004 - 1.01 s_k1_ADM_s004  = 0
 w_k1_ADM_s005_return_constraint: 0.936249 w_k1_ADM_s00 - w_k1_ADM_s005
                                 + 0.99 b_k1_ADM_s005 - 1.01 s_k1_ADM_s005  = 0
 w_k1_ADM_s006_return_constraint: 1.15531 w_k1_ADM_s00 - w_k1_ADM_s006
                                 + 0.99 b_k1_ADM_s006 - 1.01 s_k1_ADM_s006  = 0
 w_k1_ADM_s007_return_constraint: 0.909242 w_k1_ADM_s00 - w_k1_ADM_s007
                                 + 0.99 b_k1_ADM_s007 - 1.01 s_k1_ADM_s007  = 0
 w_k1_ADM_s010_return_constraint: 0.734808 w_k1_ADM_s01 - w_k1_ADM_s010
                                 + 0.99 b_k1_ADM_s010 - 1.01 s_k1_ADM_s010  = 0
 w_k1_ADM_s011_return_constraint: 1.1438 w_k1_ADM_s01 - w_k1_ADM_s011
                                 + 0.99 b_k1_ADM_s011 - 1.01 s_k1_ADM_s011  = 0
 w_k1_ADM_s012_return_constraint: 0.934232 w_k1_ADM_s01 - w_k1_ADM_s012
                                 + 0.99 b_k1_ADM_s012 - 1.01 s_k1_ADM_s012  = 0
 w_k1_ADM_s013_return_constraint: 0.960541 w_k1_ADM_s01 - w_k1_ADM_s013
                                 + 0.99 b_k1_ADM_s013 - 1.01 s_k1_ADM_s013  = 0
 w_k1_ADM_s014_return_constraint: 1.24307 w_k1_ADM_s01 - w_k1_ADM_s014
                                 + 0.99 b_k1_ADM_s014 - 1.01 s_k1_ADM_s014  = 0
 w_k1_ADM_s015_return_constraint: 1.02585 w_k1_ADM_s01 - w_k1_ADM_s015
                                 + 0.99 b_k1_ADM_s015 - 1.01 s_k1_ADM_s015  = 0
 w_k1_ADM_s016_return_constraint: 1.15977 w_k1_ADM_s01 - w_k1_ADM_s016
                                 + 0.99 b_k1_ADM_s016 - 1.01 s_k1_ADM_s016  = 0
 w_k1_ADM_s017_return_constraint: 0.985173 w_k1_ADM_s01 - w_k1_ADM_s017
                                 + 0.99 b_k1_ADM_s017 - 1.01 s_k1_ADM_s017  = 0
 w_k1_ADM_s100_return_constraint: 0.838236 w_k1_ADM_s10 - w_k1_ADM_s100
                                 + 0.99 b_k1_ADM_s100 - 1.01 s_k1_ADM_s100  = 0
 w_k1_ADM_s101_return_constraint: 0.833413 w_k1_ADM_s10 - w_k1_ADM_s101
                                 + 0.99 b_k1_ADM_s101 - 1.01 s_k1_ADM_s101  = 0
 w_k1_ADM_s102_return_constraint: 1.10451 w_k1_ADM_s10 - w_k1_ADM_s102
                                 + 0.99 b_k1_ADM_s102 - 1.01 s_k1_ADM_s102  = 0
 w_k1_ADM_s103_return_constraint: 1.36492 w_k1_ADM_s10 - w_k1_ADM_s103
                                 + 0.99 b_k1_ADM_s103 - 1.01 s_k1_ADM_s103  = 0
 w_k1_ADM_s104_return_constraint: 1.20106 w_k1_ADM_s10 - w_k1_ADM_s104
                                 + 0.99 b_k1_ADM_s104 - 1.01 s_k1_ADM_s104  = 0
 w_k1_ADM_s105_return_constraint: 0.740069 w_k1_ADM_s10 - w_k1_ADM_s105
                                 + 0.99 b_k1_ADM_s105 - 1.01 s_k1_ADM_s105  = 0
 w_k1_ADM_s106_return_constraint: 0.936632 w_k1_ADM_s10 - w_k1_ADM_s106
                                 + 0.99 b_k1_ADM_s106 - 1.01 s_k1_ADM_s106  = 0
 w_k1_ADM_s107_return_constraint: 1.23667 w_k1_ADM_s10 - w_k1_ADM_s107
                                 + 0.99 b_k1_ADM_s107 - 1.01 s_k1_ADM_s107  = 0
 w_k1_ADM_s110_return_constraint: 1.04102 w_k1_ADM_s11 - w_k1_ADM_s110
                                 + 0.99 b_k1_ADM_s110 - 1.01 s_k1_ADM_s110  = 0
 w_k1_ADM_s111_return_constraint: 0.933947 w_k1_ADM_s11 - w_k1_ADM_s111
                                 + 0.99 b_k1_ADM_s111 - 1.01 s_k1_ADM_s111  = 0
 w_k1_ADM_s112_return_constraint: 1.04433 w_k1_ADM_s11 - w_k1_ADM_s112
                                 + 0.99 b_k1_ADM_s112 - 1.01 s_k1_ADM_s112  = 0
 w_k1_ADM_s113_return_constraint: 1.10096 w_k1_ADM_s11 - w_k1_ADM_s113
                                 + 0.99 b_k1_ADM_s113 - 1.01 s_k1_ADM_s113  = 0
 w_k1_ADM_s114_return_constraint: 1.12281 w_k1_ADM_s11 - w_k1_ADM_s114
                                 + 0.99 b_k1_ADM_s114 - 1.01 s_k1_ADM_s114  = 0
 w_k1_ADM_s115_return_constraint: 1.35688 w_k1_ADM_s11 - w_k1_ADM_s115
                                 + 0.99 b_k1_ADM_s115 - 1.01 s_k1_ADM_s115  = 0
 w_k1_ADM_s116_return_constraint: 1.00287 w_k1_ADM_s11 - w_k1_ADM_s116
                                 + 0.99 b_k1_ADM_s116 - 1.01 s_k1_ADM_s116  = 0
 w_k1_ADM_s117_return_constraint: 0.948886 w_k1_ADM_s11 - w_k1_ADM_s117
                                 + 0.99 b_k1_ADM_s117 - 1.01 s_k1_ADM_s117  = 0
 w_k1_AXP_s0_return_constraint:  1.23991 w_0_AXP - w_k1_AXP_s0
                                 + 0.99 b_k1_AXP_s0 - 1.01 s_k1_AXP_s0  = 0
 w_k1_AXP_s1_return_constraint:  0.86111 w_0_AXP - w_k1_AXP_s1
                                 + 0.99 b_k1_AXP_s1 - 1.01 s_k1_AXP_s1  = 0
 w_k1_AXP_s00_return_constraint: 1.24167 w_k1_AXP_s0 - w_k1_AXP_s00
                                 + 0.99 b_k1_AXP_s00 - 1.01 s_k1_AXP_s00  = 0
 w_k1_AXP_s01_return_constraint: 0.771984 w_k1_AXP_s0 - w_k1_AXP_s01
                                 + 0.99 b_k1_AXP_s01 - 1.01 s_k1_AXP_s01  = 0
 w_k1_AXP_s10_return_constraint: 0.936541 w_k1_AXP_s1 - w_k1_AXP_s10
                                 + 0.99 b_k1_AXP_s10 - 1.01 s_k1_AXP_s10  = 0
 w_k1_AXP_s11_return_constraint: 1.3164 w_k1_AXP_s1 - w_k1_AXP_s11
                                 + 0.99 b_k1_AXP_s11 - 1.01 s_k1_AXP_s11  = 0
 w_k1_AXP_s000_return_constraint: 1.0184 w_k1_AXP_s00 - w_k1_AXP_s000
                                 + 0.99 b_k1_AXP_s000 - 1.01 s_k1_AXP_s000  = 0
 w_k1_AXP_s001_return_constraint: 1.43057 w_k1_AXP_s00 - w_k1_AXP_s001
                                 + 0.99 b_k1_AXP_s001 - 1.01 s_k1_AXP_s001  = 0
 w_k1_AXP_s002_return_constraint: 0.989302 w_k1_AXP_s00 - w_k1_AXP_s002
                                 + 0.99 b_k1_AXP_s002 - 1.01 s_k1_AXP_s002  = 0
 w_k1_AXP_s003_return_constraint: 1.52567 w_k1_AXP_s00 - w_k1_AXP_s003
                                 + 0.99 b_k1_AXP_s003 - 1.01 s_k1_AXP_s003  = 0
 w_k1_AXP_s004_return_constraint: 1.05281 w_k1_AXP_s00 - w_k1_AXP_s004
                                 + 0.99 b_k1_AXP_s004 - 1.01 s_k1_AXP_s004  = 0
 w_k1_AXP_s005_return_constraint: 0.707436 w_k1_AXP_s00 - w_k1_AXP_s005
                                 + 0.99 b_k1_AXP_s005 - 1.01 s_k1_AXP_s005  = 0
 w_k1_AXP_s006_return_constraint: 1.3536 w_k1_AXP_s00 - w_k1_AXP_s006
                                 + 0.99 b_k1_AXP_s006 - 1.01 s_k1_AXP_s006  = 0
 w_k1_AXP_s007_return_constraint: 0.771937 w_k1_AXP_s00 - w_k1_AXP_s007
                                 + 0.99 b_k1_AXP_s007 - 1.01 s_k1_AXP_s007  = 0
 w_k1_AXP_s010_return_constraint: 0.780082 w_k1_AXP_s01 - w_k1_AXP_s010
                                 + 0.99 b_k1_AXP_s010 - 1.01 s_k1_AXP_s010  = 0
 w_k1_AXP_s011_return_constraint: 1.34816 w_k1_AXP_s01 - w_k1_AXP_s011
                                 + 0.99 b_k1_AXP_s011 - 1.01 s_k1_AXP_s011  = 0
 w_k1_AXP_s012_return_constraint: 0.788081 w_k1_AXP_s01 - w_k1_AXP_s012
                                 + 0.99 b_k1_AXP_s012 - 1.01 s_k1_AXP_s012  = 0
 w_k1_AXP_s013_return_constraint: 1.3302 w_k1_AXP_s01 - w_k1_AXP_s013
                                 + 0.99 b_k1_AXP_s013 - 1.01 s_k1_AXP_s013  = 0
 w_k1_AXP_s014_return_constraint: 1.17979 w_k1_AXP_s01 - w_k1_AXP_s014
                                 + 0.99 b_k1_AXP_s014 - 1.01 s_k1_AXP_s014  = 0
 w_k1_AXP_s015_return_constraint: 0.948176 w_k1_AXP_s01 - w_k1_AXP_s015
                                 + 0.99 b_k1_AXP_s015 - 1.01 s_k1_AXP_s015  = 0
 w_k1_AXP_s016_return_constraint: 1.33233 w_k1_AXP_s01 - w_k1_AXP_s016
                                 + 0.99 b_k1_AXP_s016 - 1.01 s_k1_AXP_s016  = 0
 w_k1_AXP_s017_return_constraint: 0.905877 w_k1_AXP_s01 - w_k1_AXP_s017
                                 + 0.99 b_k1_AXP_s017 - 1.01 s_k1_AXP_s017  = 0
 w_k1_AXP_s100_return_constraint: 0.807597 w_k1_AXP_s10 - w_k1_AXP_s100
                                 + 0.99 b_k1_AXP_s100 - 1.01 s_k1_AXP_s100  = 0
 w_k1_AXP_s101_return_constraint: 0.684012 w_k1_AXP_s10 - w_k1_AXP_s101
                                 + 0.99 b_k1_AXP_s101 - 1.01 s_k1_AXP_s101  = 0
 w_k1_AXP_s102_return_constraint: 1.11661 w_k1_AXP_s10 - w_k1_AXP_s102
                                 + 0.99 b_k1_AXP_s102 - 1.01 s_k1_AXP_s102  = 0
 w_k1_AXP_s103_return_constraint: 1.48777 w_k1_AXP_s10 - w_k1_AXP_s103
                                 + 0.99 b_k1_AXP_s103 - 1.01 s_k1_AXP_s103  = 0
 w_k1_AXP_s104_return_constraint: 1.04577 w_k1_AXP_s10 - w_k1_AXP_s104
                                 + 0.99 b_k1_AXP_s104 - 1.01 s_k1_AXP_s104  = 0
 w_k1_AXP_s105_return_constraint: 0.671065 w_k1_AXP_s10 - w_k1_AXP_s105
                                 + 0.99 b_k1_AXP_s105 - 1.01 s_k1_AXP_s105  = 0
 w_k1_AXP_s106_return_constraint: 0.955175 w_k1_AXP_s10 - w_k1_AXP_s106
                                 + 0.99 b_k1_AXP_s106 - 1.01 s_k1_AXP_s106  = 0
 w_k1_AXP_s107_return_constraint: 1.2488 w_k1_AXP_s10 - w_k1_AXP_s107
                                 + 0.99 b_k1_AXP_s107 - 1.01 s_k1_AXP_s107  = 0
 w_k1_AXP_s110_return_constraint: 0.819417 w_k1_AXP_s11 - w_k1_AXP_s110
                                 + 0.99 b_k1_AXP_s110 - 1.01 s_k1_AXP_s110  = 0
 w_k1_AXP_s111_return_constraint: 0.859005 w_k1_AXP_s11 - w_k1_AXP_s111
                                 + 0.99 b_k1_AXP_s111 - 1.01 s_k1_AXP_s111  = 0
 w_k1_AXP_s112_return_constraint: 0.737529 w_k1_AXP_s11 - w_k1_AXP_s112
                                 + 0.99 b_k1_AXP_s112 - 1.01 s_k1_AXP_s112  = 0
 w_k1_AXP_s113_return_constraint: 1.36988 w_k1_AXP_s11 - w_k1_AXP_s113
                                 + 0.99 b_k1_AXP_s113 - 1.01 s_k1_AXP_s113  = 0
 w_k1_AXP_s114_return_constraint: 1.33226 w_k1_AXP_s11 - w_k1_AXP_s114
                                 + 0.99 b_k1_AXP_s114 - 1.01 s_k1_AXP_s114  = 0
 w_k1_AXP_s115_return_constraint: 1.5604 w_k1_AXP_s11 - w_k1_AXP_s115
                                 + 0.99 b_k1_AXP_s115 - 1.01 s_k1_AXP_s115  = 0
 w_k1_AXP_s116_return_constraint: 1.10375 w_k1_AXP_s11 - w_k1_AXP_s116
                                 + 0.99 b_k1_AXP_s116 - 1.01 s_k1_AXP_s116  = 0
 w_k1_AXP_s117_return_constraint: 0.938992 w_k1_AXP_s11 - w_k1_AXP_s117
                                 + 0.99 b_k1_AXP_s117 - 1.01 s_k1_AXP_s117  = 0
 w_k1_DIS_s0_return_constraint:  1.29751 w_0_DIS - w_k1_DIS_s0
                                 + 0.99 b_k1_DIS_s0 - 1.01 s_k1_DIS_s0  = 0
 w_k1_DIS_s1_return_constraint:  0.859872 w_0_DIS - w_k1_DIS_s1
                                 + 0.99 b_k1_DIS_s1 - 1.01 s_k1_DIS_s1  = 0
 w_k1_DIS_s00_return_constraint: 1.28878 w_k1_DIS_s0 - w_k1_DIS_s00
                                 + 0.99 b_k1_DIS_s00 - 1.01 s_k1_DIS_s00  = 0
 w_k1_DIS_s01_return_constraint: 0.80224 w_k1_DIS_s0 - w_k1_DIS_s01
                                 + 0.99 b_k1_DIS_s01 - 1.01 s_k1_DIS_s01  = 0
 w_k1_DIS_s10_return_constraint: 0.899353 w_k1_DIS_s1 - w_k1_DIS_s10
                                 + 0.99 b_k1_DIS_s10 - 1.01 s_k1_DIS_s10  = 0
 w_k1_DIS_s11_return_constraint: 1.41947 w_k1_DIS_s1 - w_k1_DIS_s11
                                 + 0.99 b_k1_DIS_s11 - 1.01 s_k1_DIS_s11  = 0
 w_k1_DIS_s000_return_constraint: 1.29917 w_k1_DIS_s00 - w_k1_DIS_s000
                                 + 0.99 b_k1_DIS_s000 - 1.01 s_k1_DIS_s000  = 0
 w_k1_DIS_s001_return_constraint: 1.95232 w_k1_DIS_s00 - w_k1_DIS_s001
                                 + 0.99 b_k1_DIS_s001 - 1.01 s_k1_DIS_s001  = 0
 w_k1_DIS_s002_return_constraint: 1.04253 w_k1_DIS_s00 - w_k1_DIS_s002
                                 + 0.99 b_k1_DIS_s002 - 1.01 s_k1_DIS_s002  = 0
 w_k1_DIS_s003_return_constraint: 1.45879 w_k1_DIS_s00 - w_k1_DIS_s003
                                 + 0.99 b_k1_DIS_s003 - 1.01 s_k1_DIS_s003  = 0
 w_k1_DIS_s004_return_constraint: 0.929586 w_k1_DIS_s00 - w_k1_DIS_s004
                                 + 0.99 b_k1_DIS_s004 - 1.01 s_k1_DIS_s004  = 0
 w_k1_DIS_s005_return_constraint: 0.675483 w_k1_DIS_s00 - w_k1_DIS_s005
                                 + 0.99 b_k1_DIS_s005 - 1.01 s_k1_DIS_s005  = 0
 w_k1_DIS_s006_return_constraint: 1.27175 w_k1_DIS_s00 - w_k1_DIS_s006
                                 + 0.99 b_k1_DIS_s006 - 1.01 s_k1_DIS_s006  = 0
 w_k1_DIS_s007_return_constraint: 0.702521 w_k1_DIS_s00 - w_k1_DIS_s007
                                 + 0.99 b_k1_DIS_s007 - 1.01 s_k1_DIS_s007  = 0
 w_k1_DIS_s010_return_constraint: 0.578399 w_k1_DIS_s01 - w_k1_DIS_s010
                                 + 0.99 b_k1_DIS_s010 - 1.01 s_k1_DIS_s010  = 0
 w_k1_DIS_s011_return_constraint: 1.26433 w_k1_DIS_s01 - w_k1_DIS_s011
                                 + 0.99 b_k1_DIS_s011 - 1.01 s_k1_DIS_s011  = 0
 w_k1_DIS_s012_return_constraint: 0.723475 w_k1_DIS_s01 - w_k1_DIS_s012
                                 + 0.99 b_k1_DIS_s012 - 1.01 s_k1_DIS_s012  = 0
 w_k1_DIS_s013_return_constraint: 0.854573 w_k1_DIS_s01 - w_k1_DIS_s013
                                 + 0.99 b_k1_DIS_s013 - 1.01 s_k1_DIS_s013  = 0
 w_k1_DIS_s014_return_constraint: 0.924157 w_k1_DIS_s01 - w_k1_DIS_s014
                                 + 0.99 b_k1_DIS_s014 - 1.01 s_k1_DIS_s014  = 0
 w_k1_DIS_s015_return_constraint: 0.965295 w_k1_DIS_s01 - w_k1_DIS_s015
                                 + 0.99 b_k1_DIS_s015 - 1.01 s_k1_DIS_s015  = 0
 w_k1_DIS_s016_return_constraint: 1.40667 w_k1_DIS_s01 - w_k1_DIS_s016
                                 + 0.99 b_k1_DIS_s016 - 1.01 s_k1_DIS_s016  = 0
 w_k1_DIS_s017_return_constraint: 0.883641 w_k1_DIS_s01 - w_k1_DIS_s017
                                 + 0.99 b_k1_DIS_s017 - 1.01 s_k1_DIS_s017  = 0
 w_k1_DIS_s100_return_constraint: 0.87641 w_k1_DIS_s10 - w_k1_DIS_s100
                                 + 0.99 b_k1_DIS_s100 - 1.01 s_k1_DIS_s100  = 0
 w_k1_DIS_s101_return_constraint: 0.639848 w_k1_DIS_s10 - w_k1_DIS_s101
                                 + 0.99 b_k1_DIS_s101 - 1.01 s_k1_DIS_s101  = 0
 w_k1_DIS_s102_return_constraint: 1.11431 w_k1_DIS_s10 - w_k1_DIS_s102
                                 + 0.99 b_k1_DIS_s102 - 1.01 s_k1_DIS_s102  = 0
 w_k1_DIS_s103_return_constraint: 1.57402 w_k1_DIS_s10 - w_k1_DIS_s103
                                 + 0.99 b_k1_DIS_s103 - 1.01 s_k1_DIS_s103  = 0
 w_k1_DIS_s104_return_constraint: 1.01327 w_k1_DIS_s10 - w_k1_DIS_s104
                                 + 0.99 b_k1_DIS_s104 - 1.01 s_k1_DIS_s104  = 0
 w_k1_DIS_s105_return_constraint: 0.522574 w_k1_DIS_s10 - w_k1_DIS_s105
                                 + 0.99 b_k1_DIS_s105 - 1.01 s_k1_DIS_s105  = 0
 w_k1_DIS_s106_return_constraint: 0.965715 w_k1_DIS_s10 - w_k1_DIS_s106
                                 + 0.99 b_k1_DIS_s106 - 1.01 s_k1_DIS_s106  = 0
 w_k1_DIS_s107_return_constraint: 1.55336 w_k1_DIS_s10 - w_k1_DIS_s107
                                 + 0.99 b_k1_DIS_s107 - 1.01 s_k1_DIS_s107  = 0
 w_k1_DIS_s110_return_constraint: 1.02945 w_k1_DIS_s11 - w_k1_DIS_s110
                                 + 0.99 b_k1_DIS_s110 - 1.01 s_k1_DIS_s110  = 0
 w_k1_DIS_s111_return_constraint: 0.760992 w_k1_DIS_s11 - w_k1_DIS_s111
                                 + 0.99 b_k1_DIS_s111 - 1.01 s_k1_DIS_s111  = 0
 w_k1_DIS_s112_return_constraint: 0.758106 w_k1_DIS_s11 - w_k1_DIS_s112
                                 + 0.99 b_k1_DIS_s112 - 1.01 s_k1_DIS_s112  = 0
 w_k1_DIS_s113_return_constraint: 1.2472 w_k1_DIS_s11 - w_k1_DIS_s113
                                 + 0.99 b_k1_DIS_s113 - 1.01 s_k1_DIS_s113  = 0
 w_k1_DIS_s114_return_constraint: 0.99639 w_k1_DIS_s11 - w_k1_DIS_s114
                                 + 0.99 b_k1_DIS_s114 - 1.01 s_k1_DIS_s114  = 0
 w_k1_DIS_s115_return_constraint: 1.5659 w_k1_DIS_s11 - w_k1_DIS_s115
                                 + 0.99 b_k1_DIS_s115 - 1.01 s_k1_DIS_s115  = 0
 w_k1_DIS_s116_return_constraint: 1.40028 w_k1_DIS_s11 - w_k1_DIS_s116
                                 + 0.99 b_k1_DIS_s116 - 1.01 s_k1_DIS_s116  = 0
 w_k1_DIS_s117_return_constraint: 0.857903 w_k1_DIS_s11 - w_k1_DIS_s117
                                 + 0.99 b_k1_DIS_s117 - 1.01 s_k1_DIS_s117  = 0
 w_k1_FDX_s0_return_constraint:  1.19276 w_0_FDX - w_k1_FDX_s0
                                 + 0.99 b_k1_FDX_s0 - 1.01 s_k1_FDX_s0  = 0
 w_k1_FDX_s1_return_constraint:  0.890739 w_0_FDX - w_k1_FDX_s1
                                 + 0.99 b_k1_FDX_s1 - 1.01 s_k1_FDX_s1  = 0
 w_k1_FDX_s00_return_constraint: 1.15218 w_k1_FDX_s0 - w_k1_FDX_s00
                                 + 0.99 b_k1_FDX_s00 - 1.01 s_k1_FDX_s00  = 0
 w_k1_FDX_s01_return_constraint: 0.84368 w_k1_FDX_s0 - w_k1_FDX_s01
                                 + 0.99 b_k1_FDX_s01 - 1.01 s_k1_FDX_s01  = 0
 w_k1_FDX_s10_return_constraint: 0.950976 w_k1_FDX_s1 - w_k1_FDX_s10
                                 + 0.99 b_k1_FDX_s10 - 1.01 s_k1_FDX_s10  = 0
 w_k1_FDX_s11_return_constraint: 1.24275 w_k1_FDX_s1 - w_k1_FDX_s11
                                 + 0.99 b_k1_FDX_s11 - 1.01 s_k1_FDX_s11  = 0
 w_k1_FDX_s000_return_constraint: 1.13895 w_k1_FDX_s00 - w_k1_FDX_s000
                                 + 0.99 b_k1_FDX_s000 - 1.01 s_k1_FDX_s000  = 0
 w_k1_FDX_s001_return_constraint: 1.57232 w_k1_FDX_s00 - w_k1_FDX_s001
                                 + 0.99 b_k1_FDX_s001 - 1.01 s_k1_FDX_s001  = 0
 w_k1_FDX_s002_return_constraint: 0.964649 w_k1_FDX_s00 - w_k1_FDX_s002
                                 + 0.99 b_k1_FDX_s002 - 1.01 s_k1_FDX_s002  = 0
 w_k1_FDX_s003_return_constraint: 1.31356 w_k1_FDX_s00 - w_k1_FDX_s003
                                 + 0.99 b_k1_FDX_s003 - 1.01 s_k1_FDX_s003  = 0
 w_k1_FDX_s004_return_constraint: 1.10808 w_k1_FDX_s00 - w_k1_FDX_s004
                                 + 0.99 b_k1_FDX_s004 - 1.01 s_k1_FDX_s004  = 0
 w_k1_FDX_s005_return_constraint: 0.883135 w_k1_FDX_s00 - w_k1_FDX_s005
                                 + 0.99 b_k1_FDX_s005 - 1.01 s_k1_FDX_s005  = 0
 w_k1_FDX_s006_return_constraint: 1.06135 w_k1_FDX_s00 - w_k1_FDX_s006
                                 + 0.99 b_k1_FDX_s006 - 1.01 s_k1_FDX_s006  = 0
 w_k1_FDX_s007_return_constraint: 0.799469 w_k1_FDX_s00 - w_k1_FDX_s007
                                 + 0.99 b_k1_FDX_s007 - 1.01 s_k1_FDX_s007  = 0
 w_k1_FDX_s010_return_constraint: 0.739647 w_k1_FDX_s01 - w_k1_FDX_s010
                                 + 0.99 b_k1_FDX_s010 - 1.01 s_k1_FDX_s010  = 0
 w_k1_FDX_s011_return_constraint: 1.15278 w_k1_FDX_s01 - w_k1_FDX_s011
                                 + 0.99 b_k1_FDX_s011 - 1.01 s_k1_FDX_s011  = 0
 w_k1_FDX_s012_return_constraint: 0.839659 w_k1_FDX_s01 - w_k1_FDX_s012
                                 + 0.99 b_k1_FDX_s012 - 1.01 s_k1_FDX_s012  = 0
 w_k1_FDX_s013_return_constraint: 1.05346 w_k1_FDX_s01 - w_k1_FDX_s013
                                 + 0.99 b_k1_FDX_s013 - 1.01 s_k1_FDX_s013  = 0
 w_k1_FDX_s014_return_constraint: 1.1706 w_k1_FDX_s01 - w_k1_FDX_s014
                                 + 0.99 b_k1_FDX_s014 - 1.01 s_k1_FDX_s014  = 0
 w_k1_FDX_s015_return_constraint: 0.887577 w_k1_FDX_s01 - w_k1_FDX_s015
                                 + 0.99 b_k1_FDX_s015 - 1.01 s_k1_FDX_s015  = 0
 w_k1_FDX_s016_return_constraint: 1.35808 w_k1_FDX_s01 - w_k1_FDX_s016
                                 + 0.99 b_k1_FDX_s016 - 1.01 s_k1_FDX_s016  = 0
 w_k1_FDX_s017_return_constraint: 0.993161 w_k1_FDX_s01 - w_k1_FDX_s017
                                 + 0.99 b_k1_FDX_s017 - 1.01 s_k1_FDX_s017  = 0
 w_k1_FDX_s100_return_constraint: 0.832264 w_k1_FDX_s10 - w_k1_FDX_s100
                                 + 0.99 b_k1_FDX_s100 - 1.01 s_k1_FDX_s100  = 0
 w_k1_FDX_s101_return_constraint: 0.871798 w_k1_FDX_s10 - w_k1_FDX_s101
                                 + 0.99 b_k1_FDX_s101 - 1.01 s_k1_FDX_s101  = 0
 w_k1_FDX_s102_return_constraint: 1.04446 w_k1_FDX_s10 - w_k1_FDX_s102
                                 + 0.99 b_k1_FDX_s102 - 1.01 s_k1_FDX_s102  = 0
 w_k1_FDX_s103_return_constraint: 1.23667 w_k1_FDX_s10 - w_k1_FDX_s103
                                 + 0.99 b_k1_FDX_s103 - 1.01 s_k1_FDX_s103  = 0
 w_k1_FDX_s104_return_constraint: 1.18733 w_k1_FDX_s10 - w_k1_FDX_s104
                                 + 0.99 b_k1_FDX_s104 - 1.01 s_k1_FDX_s104  = 0
 w_k1_FDX_s105_return_constraint: 0.702264 w_k1_FDX_s10 - w_k1_FDX_s105
                                 + 0.99 b_k1_FDX_s105 - 1.01 s_k1_FDX_s105  = 0
 w_k1_FDX_s106_return_constraint: 1.06356 w_k1_FDX_s10 - w_k1_FDX_s106
                                 + 0.99 b_k1_FDX_s106 - 1.01 s_k1_FDX_s106  = 0
 w_k1_FDX_s107_return_constraint: 1.40922 w_k1_FDX_s10 - w_k1_FDX_s107
                                 + 0.99 b_k1_FDX_s107 - 1.01 s_k1_FDX_s107  = 0
 w_k1_FDX_s110_return_constraint: 1.06102 w_k1_FDX_s11 - w_k1_FDX_s110
                                 + 0.99 b_k1_FDX_s110 - 1.01 s_k1_FDX_s110  = 0
 w_k1_FDX_s111_return_constraint: 0.822778 w_k1_FDX_s11 - w_k1_FDX_s111
                                 + 0.99 b_k1_FDX_s111 - 1.01 s_k1_FDX_s111  = 0
 w_k1_FDX_s112_return_constraint: 0.970828 w_k1_FDX_s11 - w_k1_FDX_s112
                                 + 0.99 b_k1_FDX_s112 - 1.01 s_k1_FDX_s112  = 0
 w_k1_FDX_s113_return_constraint: 1.02518 w_k1_FDX_s11 - w_k1_FDX_s113
                                 + 0.99 b_k1_FDX_s113 - 1.01 s_k1_FDX_s113  = 0
 w_k1_FDX_s114_return_constraint: 1.126 w_k1_FDX_s11 - w_k1_FDX_s114
                                 + 0.99 b_k1_FDX_s114 - 1.01 s_k1_FDX_s114  = 0
 w_k1_FDX_s115_return_constraint: 1.51805 w_k1_FDX_s11 - w_k1_FDX_s115
                                 + 0.99 b_k1_FDX_s115 - 1.01 s_k1_FDX_s115  = 0
 w_k1_FDX_s116_return_constraint: 1.15894 w_k1_FDX_s11 - w_k1_FDX_s116
                                 + 0.99 b_k1_FDX_s116 - 1.01 s_k1_FDX_s116  = 0
 w_k1_FDX_s117_return_constraint: 0.760832 w_k1_FDX_s11 - w_k1_FDX_s117
                                 + 0.99 b_k1_FDX_s117 - 1.01 s_k1_FDX_s117  = 0
 w_k1_IBM_s0_return_constraint:  1.15675 w_0_IBM - w_k1_IBM_s0
                                 + 0.99 b_k1_IBM_s0 - 1.01 s_k1_IBM_s0  = 0
 w_k1_IBM_s1_return_constraint:  0.942478 w_0_IBM - w_k1_IBM_s1
                                 + 0.99 b_k1_IBM_s1 - 1.01 s_k1_IBM_s1  = 0
 w_k1_IBM_s00_return_constraint: 1.13003 w_k1_IBM_s0 - w_k1_IBM_s00
                                 + 0.99 b_k1_IBM_s00 - 1.01 s_k1_IBM_s00  = 0
 w_k1_IBM_s01_return_constraint: 0.90449 w_k1_IBM_s0 - w_k1_IBM_s01
                                 + 0.99 b_k1_IBM_s01 - 1.01 s_k1_IBM_s01  = 0
 w_k1_IBM_s10_return_constraint: 0.915595 w_k1_IBM_s1 - w_k1_IBM_s10
                                 + 0.99 b_k1_IBM_s10 - 1.01 s_k1_IBM_s10  = 0
 w_k1_IBM_s11_return_constraint: 1.329 w_k1_IBM_s1 - w_k1_IBM_s11
                                 + 0.99 b_k1_IBM_s11 - 1.01 s_k1_IBM_s11  = 0
 w_k1_IBM_s000_return_constraint: 0.852991 w_k1_IBM_s00 - w_k1_IBM_s000
                                 + 0.99 b_k1_IBM_s000 - 1.01 s_k1_IBM_s000  = 0
 w_k1_IBM_s001_return_constraint: 1.38709 w_k1_IBM_s00 - w_k1_IBM_s001
                                 + 0.99 b_k1_IBM_s001 - 1.01 s_k1_IBM_s001  = 0
 w_k1_IBM_s002_return_constraint: 1.32407 w_k1_IBM_s00 - w_k1_IBM_s002
                                 + 0.99 b_k1_IBM_s002 - 1.01 s_k1_IBM_s002  = 0
 w_k1_IBM_s003_return_constraint: 0.989563 w_k1_IBM_s00 - w_k1_IBM_s003
                                 + 0.99 b_k1_IBM_s003 - 1.01 s_k1_IBM_s003  = 0
 w_k1_IBM_s004_return_constraint: 0.725891 w_k1_IBM_s00 - w_k1_IBM_s004
                                 + 0.99 b_k1_IBM_s004 - 1.01 s_k1_IBM_s004  = 0
 w_k1_IBM_s005_return_constraint: 0.556179 w_k1_IBM_s00 - w_k1_IBM_s005
                                 + 0.99 b_k1_IBM_s005 - 1.01 s_k1_IBM_s005  = 0
 w_k1_IBM_s006_return_constraint: 2.03699 w_k1_IBM_s00 - w_k1_IBM_s006
                                 + 0.99 b_k1_IBM_s006 - 1.01 s_k1_IBM_s006  = 0
 w_k1_IBM_s007_return_constraint: 0.922985 w_k1_IBM_s00 - w_k1_IBM_s007
                                 + 0.99 b_k1_IBM_s007 - 1.01 s_k1_IBM_s007  = 0
 w_k1_IBM_s010_return_constraint: 0.878446 w_k1_IBM_s01 - w_k1_IBM_s010
                                 + 0.99 b_k1_IBM_s010 - 1.01 s_k1_IBM_s010  = 0
 w_k1_IBM_s011_return_constraint: 1.58777 w_k1_IBM_s01 - w_k1_IBM_s011
                                 + 0.99 b_k1_IBM_s011 - 1.01 s_k1_IBM_s011  = 0
 w_k1_IBM_s012_return_constraint: 0.638413 w_k1_IBM_s01 - w_k1_IBM_s012
                                 + 0.99 b_k1_IBM_s012 - 1.01 s_k1_IBM_s012  = 0
 w_k1_IBM_s013_return_constraint: 1.07289 w_k1_IBM_s01 - w_k1_IBM_s013
                                 + 0.99 b_k1_IBM_s013 - 1.01 s_k1_IBM_s013  = 0
 w_k1_IBM_s014_return_constraint: 0.635182 w_k1_IBM_s01 - w_k1_IBM_s014
                                 + 0.99 b_k1_IBM_s014 - 1.01 s_k1_IBM_s014  = 0
 w_k1_IBM_s015_return_constraint: 1.27393 w_k1_IBM_s01 - w_k1_IBM_s015
                                 + 0.99 b_k1_IBM_s015 - 1.01 s_k1_IBM_s015  = 0
 w_k1_IBM_s016_return_constraint: 0.922994 w_k1_IBM_s01 - w_k1_IBM_s016
                                 + 0.99 b_k1_IBM_s016 - 1.01 s_k1_IBM_s016  = 0
 w_k1_IBM_s017_return_constraint: 0.735548 w_k1_IBM_s01 - w_k1_IBM_s017
                                 + 0.99 b_k1_IBM_s017 - 1.01 s_k1_IBM_s017  = 0
 w_k1_IBM_s100_return_constraint: 1.07967 w_k1_IBM_s10 - w_k1_IBM_s100
                                 + 0.99 b_k1_IBM_s100 - 1.01 s_k1_IBM_s100  = 0
 w_k1_IBM_s101_return_constraint: 0.475711 w_k1_IBM_s10 - w_k1_IBM_s101
                                 + 0.99 b_k1_IBM_s101 - 1.01 s_k1_IBM_s101  = 0
 w_k1_IBM_s102_return_constraint: 1.30281 w_k1_IBM_s10 - w_k1_IBM_s102
                                 + 0.99 b_k1_IBM_s102 - 1.01 s_k1_IBM_s102  = 0
 w_k1_IBM_s103_return_constraint: 1.73149 w_k1_IBM_s10 - w_k1_IBM_s103
                                 + 0.99 b_k1_IBM_s103 - 1.01 s_k1_IBM_s103  = 0
 w_k1_IBM_s104_return_constraint: 0.687452 w_k1_IBM_s10 - w_k1_IBM_s104
                                 + 0.99 b_k1_IBM_s104 - 1.01 s_k1_IBM_s104  = 0
 w_k1_IBM_s105_return_constraint: 0.74748 w_k1_IBM_s10 - w_k1_IBM_s105
                                 + 0.99 b_k1_IBM_s105 - 1.01 s_k1_IBM_s105  = 0
 w_k1_IBM_s106_return_constraint: 0.897867 w_k1_IBM_s10 - w_k1_IBM_s106
                                 + 0.99 b_k1_IBM_s106 - 1.01 s_k1_IBM_s106  = 0
 w_k1_IBM_s107_return_constraint: 0.95233 w_k1_IBM_s10 - w_k1_IBM_s107
                                 + 0.99 b_k1_IBM_s107 - 1.01 s_k1_IBM_s107  = 0
 w_k1_IBM_s110_return_constraint: 0.711108 w_k1_IBM_s11 - w_k1_IBM_s110
                                 + 0.99 b_k1_IBM_s110 - 1.01 s_k1_IBM_s110  = 0
 w_k1_IBM_s111_return_constraint: 0.891117 w_k1_IBM_s11 - w_k1_IBM_s111
                                 + 0.99 b_k1_IBM_s111 - 1.01 s_k1_IBM_s111  = 0
 w_k1_IBM_s112_return_constraint: 0.465342 w_k1_IBM_s11 - w_k1_IBM_s112
                                 + 0.99 b_k1_IBM_s112 - 1.01 s_k1_IBM_s112  = 0
 w_k1_IBM_s113_return_constraint: 1.39844 w_k1_IBM_s11 - w_k1_IBM_s113
                                 + 0.99 b_k1_IBM_s113 - 1.01 s_k1_IBM_s113  = 0
 w_k1_IBM_s114_return_constraint: 0.715282 w_k1_IBM_s11 - w_k1_IBM_s114
                                 + 0.99 b_k1_IBM_s114 - 1.01 s_k1_IBM_s114  = 0
 w_k1_IBM_s115_return_constraint: 0.873148 w_k1_IBM_s11 - w_k1_IBM_s115
                                 + 0.99 b_k1_IBM_s115 - 1.01 s_k1_IBM_s115  = 0
 w_k1_IBM_s116_return_constraint: 1.04794 w_k1_IBM_s11 - w_k1_IBM_s116
                                 + 0.99 b_k1_IBM_s116 - 1.01 s_k1_IBM_s116  = 0
 w_k1_IBM_s117_return_constraint: 1.57337 w_k1_IBM_s11 - w_k1_IBM_s117
                                 + 0.99 b_k1_IBM_s117 - 1.01 s_k1_IBM_s117  = 0
 w_k1_INTC_s0_return_constraint: 1.15765 w_0_INTC - w_k1_INTC_s0
                                 + 0.99 b_k1_INTC_s0 - 1.01 s_k1_INTC_s0  = 0
 w_k1_INTC_s1_return_constraint: 0.958717 w_0_INTC - w_k1_INTC_s1
                                 + 0.99 b_k1_INTC_s1 - 1.01 s_k1_INTC_s1  = 0
 w_k1_INTC_s00_return_constraint: 1.07652 w_k1_INTC_s0 - w_k1_INTC_s00
                                 + 0.99 b_k1_INTC_s00 - 1.01 s_k1_INTC_s00  = 0
 w_k1_INTC_s01_return_constraint: 0.777299 w_k1_INTC_s0 - w_k1_INTC_s01
                                 + 0.99 b_k1_INTC_s01 - 1.01 s_k1_INTC_s01  = 0
 w_k1_INTC_s10_return_constraint: 0.889101 w_k1_INTC_s1 - w_k1_INTC_s10
                                 + 0.99 b_k1_INTC_s10 - 1.01 s_k1_INTC_s10  = 0
 w_k1_INTC_s11_return_constraint: 1.06396 w_k1_INTC_s1 - w_k1_INTC_s11
                                 + 0.99 b_k1_INTC_s11 - 1.01 s_k1_INTC_s11  = 0
 w_k1_INTC_s000_return_constraint: 1.11369 w_k1_INTC_s00 - w_k1_INTC_s000
                                 + 0.99 b_k1_INTC_s000 - 1.01 s_k1_INTC_s000
                                  = 0
 w_k1_INTC_s001_return_constraint: 1.26702 w_k1_INTC_s00 - w_k1_INTC_s001
                                 + 0.99 b_k1_INTC_s001 - 1.01 s_k1_INTC_s001
                                  = 0
 w_k1_INTC_s002_return_constraint: 0.936738 w_k1_INTC_s00 - w_k1_INTC_s002
                                 + 0.99 b_k1_INTC_s002 - 1.01 s_k1_INTC_s002
                                  = 0
 w_k1_INTC_s003_return_constraint: 1.29247 w_k1_INTC_s00 - w_k1_INTC_s003
                                 + 0.99 b_k1_INTC_s003 - 1.01 s_k1_INTC_s003
                                  = 0
 w_k1_INTC_s004_return_constraint: 1.01498 w_k1_INTC_s00 - w_k1_INTC_s004
                                 + 0.99 b_k1_INTC_s004 - 1.01 s_k1_INTC_s004
                                  = 0
 w_k1_INTC_s005_return_constraint: 0.846986 w_k1_INTC_s00 - w_k1_INTC_s005
                                 + 0.99 b_k1_INTC_s005 - 1.01 s_k1_INTC_s005
                                  = 0
 w_k1_INTC_s006_return_constraint: 1.33964 w_k1_INTC_s00 - w_k1_INTC_s006
                                 + 0.99 b_k1_INTC_s006 - 1.01 s_k1_INTC_s006
                                  = 0
 w_k1_INTC_s007_return_constraint: 0.965437 w_k1_INTC_s00 - w_k1_INTC_s007
                                 + 0.99 b_k1_INTC_s007 - 1.01 s_k1_INTC_s007
                                  = 0
 w_k1_INTC_s010_return_constraint: 0.964246 w_k1_INTC_s01 - w_k1_INTC_s010
                                 + 0.99 b_k1_INTC_s010 - 1.01 s_k1_INTC_s010
                                  = 0
 w_k1_INTC_s011_return_constraint: 1.34381 w_k1_INTC_s01 - w_k1_INTC_s011
                                 + 0.99 b_k1_INTC_s011 - 1.01 s_k1_INTC_s011
                                  = 0
 w_k1_INTC_s012_return_constraint: 0.86847 w_k1_INTC_s01 - w_k1_INTC_s012
                                 + 0.99 b_k1_INTC_s012 - 1.01 s_k1_INTC_s012
                                  = 0
 w_k1_INTC_s013_return_constraint: 1.19019 w_k1_INTC_s01 - w_k1_INTC_s013
                                 + 0.99 b_k1_INTC_s013 - 1.01 s_k1_INTC_s013
                                  = 0
 w_k1_INTC_s014_return_constraint: 1.20673 w_k1_INTC_s01 - w_k1_INTC_s014
                                 + 0.99 b_k1_INTC_s014 - 1.01 s_k1_INTC_s014
                                  = 0
 w_k1_INTC_s015_return_constraint: 1.16203 w_k1_INTC_s01 - w_k1_INTC_s015
                                 + 0.99 b_k1_INTC_s015 - 1.01 s_k1_INTC_s015
                                  = 0
 w_k1_INTC_s016_return_constraint: 1.30864 w_k1_INTC_s01 - w_k1_INTC_s016
                                 + 0.99 b_k1_INTC_s016 - 1.01 s_k1_INTC_s016
                                  = 0
 w_k1_INTC_s017_return_constraint: 1.17297 w_k1_INTC_s01 - w_k1_INTC_s017
                                 + 0.99 b_k1_INTC_s017 - 1.01 s_k1_INTC_s017
                                  = 0
 w_k1_INTC_s100_return_constraint: 0.916172 w_k1_INTC_s10 - w_k1_INTC_s100
                                 + 0.99 b_k1_INTC_s100 - 1.01 s_k1_INTC_s100
                                  = 0
 w_k1_INTC_s101_return_constraint: 0.911238 w_k1_INTC_s10 - w_k1_INTC_s101
                                 + 0.99 b_k1_INTC_s101 - 1.01 s_k1_INTC_s101
                                  = 0
 w_k1_INTC_s102_return_constraint: 0.988591 w_k1_INTC_s10 - w_k1_INTC_s102
                                 + 0.99 b_k1_INTC_s102 - 1.01 s_k1_INTC_s102
                                  = 0
 w_k1_INTC_s103_return_constraint: 1.1611 w_k1_INTC_s10 - w_k1_INTC_s103
                                 + 0.99 b_k1_INTC_s103 - 1.01 s_k1_INTC_s103
                                  = 0
 w_k1_INTC_s104_return_constraint: 1.16421 w_k1_INTC_s10 - w_k1_INTC_s104
                                 + 0.99 b_k1_INTC_s104 - 1.01 s_k1_INTC_s104
                                  = 0
 w_k1_INTC_s105_return_constraint: 0.754091 w_k1_INTC_s10 - w_k1_INTC_s105
                                 + 0.99 b_k1_INTC_s105 - 1.01 s_k1_INTC_s105
                                  = 0
 w_k1_INTC_s106_return_constraint: 0.976421 w_k1_INTC_s10 - w_k1_INTC_s106
                                 + 0.99 b_k1_INTC_s106 - 1.01 s_k1_INTC_s106
                                  = 0
 w_k1_INTC_s107_return_constraint: 1.1742 w_k1_INTC_s10 - w_k1_INTC_s107
                                 + 0.99 b_k1_INTC_s107 - 1.01 s_k1_INTC_s107
                                  = 0
 w_k1_INTC_s110_return_constraint: 1.01833 w_k1_INTC_s11 - w_k1_INTC_s110
                                 + 0.99 b_k1_INTC_s110 - 1.01 s_k1_INTC_s110
                                  = 0
 w_k1_INTC_s111_return_constraint: 0.84291 w_k1_INTC_s11 - w_k1_INTC_s111
                                 + 0.99 b_k1_INTC_s111 - 1.01 s_k1_INTC_s111
                                  = 0
 w_k1_INTC_s112_return_constraint: 1.01658 w_k1_INTC_s11 - w_k1_INTC_s112
                                 + 0.99 b_k1_INTC_s112 - 1.01 s_k1_INTC_s112
                                  = 0
 w_k1_INTC_s113_return_constraint: 1.27166 w_k1_INTC_s11 - w_k1_INTC_s113
                                 + 0.99 b_k1_INTC_s113 - 1.01 s_k1_INTC_s113
                                  = 0
 w_k1_INTC_s114_return_constraint: 1.11255 w_k1_INTC_s11 - w_k1_INTC_s114
                                 + 0.99 b_k1_INTC_s114 - 1.01 s_k1_INTC_s114
                                  = 0
 w_k1_INTC_s115_return_constraint: 1.22284 w_k1_INTC_s11 - w_k1_INTC_s115
                                 + 0.99 b_k1_INTC_s115 - 1.01 s_k1_INTC_s115
                                  = 0
 w_k1_INTC_s116_return_constraint: 1.25996 w_k1_INTC_s11 - w_k1_INTC_s116
                                 + 0.99 b_k1_INTC_s116 - 1.01 s_k1_INTC_s116
                                  = 0
 w_k1_INTC_s117_return_constraint: 0.778835 w_k1_INTC_s11 - w_k1_INTC_s117
                                 + 0.99 b_k1_INTC_s117 - 1.01 s_k1_INTC_s117
                                  = 0
 w_k1_KO_s0_return_constraint:   1.18628 w_0_KO - w_k1_KO_s0 + 0.99 b_k1_KO_s0
                                 - 1.01 s_k1_KO_s0  = 0
 w_k1_KO_s1_return_constraint:   0.947561 w_0_KO - w_k1_KO_s1 + 0.99 b_k1_KO_s1
                                 - 1.01 s_k1_KO_s1  = 0
 w_k1_KO_s00_return_constraint:  1.12644 w_k1_KO_s0 - w_k1_KO_s00
                                 + 0.99 b_k1_KO_s00 - 1.01 s_k1_KO_s00  = 0
 w_k1_KO_s01_return_constraint:  0.838432 w_k1_KO_s0 - w_k1_KO_s01
                                 + 0.99 b_k1_KO_s01 - 1.01 s_k1_KO_s01  = 0
 w_k1_KO_s10_return_constraint:  0.836537 w_k1_KO_s1 - w_k1_KO_s10
                                 + 0.99 b_k1_KO_s10 - 1.01 s_k1_KO_s10  = 0
 w_k1_KO_s11_return_constraint:  1.20711 w_k1_KO_s1 - w_k1_KO_s11
                                 + 0.99 b_k1_KO_s11 - 1.01 s_k1_KO_s11  = 0
 w_k1_KO_s000_return_constraint: 1.03569 w_k1_KO_s00 - w_k1_KO_s000
                                 + 0.99 b_k1_KO_s000 - 1.01 s_k1_KO_s000  = 0
 w_k1_KO_s001_return_constraint: 1.47033 w_k1_KO_s00 - w_k1_KO_s001
                                 + 0.99 b_k1_KO_s001 - 1.01 s_k1_KO_s001  = 0
 w_k1_KO_s002_return_constraint: 1.03539 w_k1_KO_s00 - w_k1_KO_s002
                                 + 0.99 b_k1_KO_s002 - 1.01 s_k1_KO_s002  = 0
 w_k1_KO_s003_return_constraint: 1.1891 w_k1_KO_s00 - w_k1_KO_s003
                                 + 0.99 b_k1_KO_s003 - 1.01 s_k1_KO_s003  = 0
 w_k1_KO_s004_return_constraint: 0.820751 w_k1_KO_s00 - w_k1_KO_s004
                                 + 0.99 b_k1_KO_s004 - 1.01 s_k1_KO_s004  = 0
 w_k1_KO_s005_return_constraint: 0.681732 w_k1_KO_s00 - w_k1_KO_s005
                                 + 0.99 b_k1_KO_s005 - 1.01 s_k1_KO_s005  = 0
 w_k1_KO_s006_return_constraint: 1.52118 w_k1_KO_s00 - w_k1_KO_s006
                                 + 0.99 b_k1_KO_s006 - 1.01 s_k1_KO_s006  = 0
 w_k1_KO_s007_return_constraint: 0.830098 w_k1_KO_s00 - w_k1_KO_s007
                                 + 0.99 b_k1_KO_s007 - 1.01 s_k1_KO_s007  = 0
 w_k1_KO_s010_return_constraint: 0.82492 w_k1_KO_s01 - w_k1_KO_s010
                                 + 0.99 b_k1_KO_s010 - 1.01 s_k1_KO_s010  = 0
 w_k1_KO_s011_return_constraint: 1.39994 w_k1_KO_s01 - w_k1_KO_s011
                                 + 0.99 b_k1_KO_s011 - 1.01 s_k1_KO_s011  = 0
 w_k1_KO_s012_return_constraint: 0.747954 w_k1_KO_s01 - w_k1_KO_s012
                                 + 0.99 b_k1_KO_s012 - 1.01 s_k1_KO_s012  = 0
 w_k1_KO_s013_return_constraint: 0.990956 w_k1_KO_s01 - w_k1_KO_s013
                                 + 0.99 b_k1_KO_s013 - 1.01 s_k1_KO_s013  = 0
 w_k1_KO_s014_return_constraint: 0.848893 w_k1_KO_s01 - w_k1_KO_s014
                                 + 0.99 b_k1_KO_s014 - 1.01 s_k1_KO_s014  = 0
 w_k1_KO_s015_return_constraint: 1.11018 w_k1_KO_s01 - w_k1_KO_s015
                                 + 0.99 b_k1_KO_s015 - 1.01 s_k1_KO_s015  = 0
 w_k1_KO_s016_return_constraint: 1.12061 w_k1_KO_s01 - w_k1_KO_s016
                                 + 0.99 b_k1_KO_s016 - 1.01 s_k1_KO_s016  = 0
 w_k1_KO_s017_return_constraint: 0.917953 w_k1_KO_s01 - w_k1_KO_s017
                                 + 0.99 b_k1_KO_s017 - 1.01 s_k1_KO_s017  = 0
 w_k1_KO_s100_return_constraint: 1.00028 w_k1_KO_s10 - w_k1_KO_s100
                                 + 0.99 b_k1_KO_s100 - 1.01 s_k1_KO_s100  = 0
 w_k1_KO_s101_return_constraint: 0.685371 w_k1_KO_s10 - w_k1_KO_s101
                                 + 0.99 b_k1_KO_s101 - 1.01 s_k1_KO_s101  = 0
 w_k1_KO_s102_return_constraint: 1.34728 w_k1_KO_s10 - w_k1_KO_s102
                                 + 0.99 b_k1_KO_s102 - 1.01 s_k1_KO_s102  = 0
 w_k1_KO_s103_return_constraint: 1.63719 w_k1_KO_s10 - w_k1_KO_s103
                                 + 0.99 b_k1_KO_s103 - 1.01 s_k1_KO_s103  = 0
 w_k1_KO_s104_return_constraint: 0.854829 w_k1_KO_s10 - w_k1_KO_s104
                                 + 0.99 b_k1_KO_s104 - 1.01 s_k1_KO_s104  = 0
 w_k1_KO_s105_return_constraint: 0.706778 w_k1_KO_s10 - w_k1_KO_s105
                                 + 0.99 b_k1_KO_s105 - 1.01 s_k1_KO_s105  = 0
 w_k1_KO_s106_return_constraint: 0.893024 w_k1_KO_s10 - w_k1_KO_s106
                                 + 0.99 b_k1_KO_s106 - 1.01 s_k1_KO_s106  = 0
 w_k1_KO_s107_return_constraint: 1.19468 w_k1_KO_s10 - w_k1_KO_s107
                                 + 0.99 b_k1_KO_s107 - 1.01 s_k1_KO_s107  = 0
 w_k1_KO_s110_return_constraint: 0.908611 w_k1_KO_s11 - w_k1_KO_s110
                                 + 0.99 b_k1_KO_s110 - 1.01 s_k1_KO_s110  = 0
 w_k1_KO_s111_return_constraint: 0.949167 w_k1_KO_s11 - w_k1_KO_s111
                                 + 0.99 b_k1_KO_s111 - 1.01 s_k1_KO_s111  = 0
 w_k1_KO_s112_return_constraint: 0.712317 w_k1_KO_s11 - w_k1_KO_s112
                                 + 0.99 b_k1_KO_s112 - 1.01 s_k1_KO_s112  = 0
 w_k1_KO_s113_return_constraint: 1.32249 w_k1_KO_s11 - w_k1_KO_s113
                                 + 0.99 b_k1_KO_s113 - 1.01 s_k1_KO_s113  = 0
 w_k1_KO_s114_return_constraint: 0.856449 w_k1_KO_s11 - w_k1_KO_s114
                                 + 0.99 b_k1_KO_s114 - 1.01 s_k1_KO_s114  = 0
 w_k1_KO_s115_return_constraint: 1.07914 w_k1_KO_s11 - w_k1_KO_s115
                                 + 0.99 b_k1_KO_s115 - 1.01 s_k1_KO_s115  = 0
 w_k1_KO_s116_return_constraint: 1.07702 w_k1_KO_s11 - w_k1_KO_s116
                                 + 0.99 b_k1_KO_s116 - 1.01 s_k1_KO_s116  = 0
 w_k1_KO_s117_return_constraint: 1.27306 w_k1_KO_s11 - w_k1_KO_s117
                                 + 0.99 b_k1_KO_s117 - 1.01 s_k1_KO_s117  = 0
 w_k1_MAT_s0_return_constraint:  1.23969 w_0_MAT - w_k1_MAT_s0
                                 + 0.99 b_k1_MAT_s0 - 1.01 s_k1_MAT_s0  = 0
 w_k1_MAT_s1_return_constraint:  0.978525 w_0_MAT - w_k1_MAT_s1
                                 + 0.99 b_k1_MAT_s1 - 1.01 s_k1_MAT_s1  = 0
 w_k1_MAT_s00_return_constraint: 1.25754 w_k1_MAT_s0 - w_k1_MAT_s00
                                 + 0.99 b_k1_MAT_s00 - 1.01 s_k1_MAT_s00  = 0
 w_k1_MAT_s01_return_constraint: 0.829832 w_k1_MAT_s0 - w_k1_MAT_s01
                                 + 0.99 b_k1_MAT_s01 - 1.01 s_k1_MAT_s01  = 0
 w_k1_MAT_s10_return_constraint: 0.920759 w_k1_MAT_s1 - w_k1_MAT_s10
                                 + 0.99 b_k1_MAT_s10 - 1.01 s_k1_MAT_s10  = 0
 w_k1_MAT_s11_return_constraint: 1.37612 w_k1_MAT_s1 - w_k1_MAT_s11
                                 + 0.99 b_k1_MAT_s11 - 1.01 s_k1_MAT_s11  = 0
 w_k1_MAT_s000_return_constraint: 0.743167 w_k1_MAT_s00 - w_k1_MAT_s000
                                 + 0.99 b_k1_MAT_s000 - 1.01 s_k1_MAT_s000  = 0
 w_k1_MAT_s001_return_constraint: 1.26855 w_k1_MAT_s00 - w_k1_MAT_s001
                                 + 0.99 b_k1_MAT_s001 - 1.01 s_k1_MAT_s001  = 0
 w_k1_MAT_s002_return_constraint: 1.21164 w_k1_MAT_s00 - w_k1_MAT_s002
                                 + 0.99 b_k1_MAT_s002 - 1.01 s_k1_MAT_s002  = 0
 w_k1_MAT_s003_return_constraint: 1.18634 w_k1_MAT_s00 - w_k1_MAT_s003
                                 + 0.99 b_k1_MAT_s003 - 1.01 s_k1_MAT_s003  = 0
 w_k1_MAT_s004_return_constraint: 0.783162 w_k1_MAT_s00 - w_k1_MAT_s004
                                 + 0.99 b_k1_MAT_s004 - 1.01 s_k1_MAT_s004  = 0
 w_k1_MAT_s005_return_constraint: 0.473315 w_k1_MAT_s00 - w_k1_MAT_s005
                                 + 0.99 b_k1_MAT_s005 - 1.01 s_k1_MAT_s005  = 0
 w_k1_MAT_s006_return_constraint: 1.98893 w_k1_MAT_s00 - w_k1_MAT_s006
                                 + 0.99 b_k1_MAT_s006 - 1.01 s_k1_MAT_s006  = 0
 w_k1_MAT_s007_return_constraint: 0.73775 w_k1_MAT_s00 - w_k1_MAT_s007
                                 + 0.99 b_k1_MAT_s007 - 1.01 s_k1_MAT_s007  = 0
 w_k1_MAT_s010_return_constraint: 0.819523 w_k1_MAT_s01 - w_k1_MAT_s010
                                 + 0.99 b_k1_MAT_s010 - 1.01 s_k1_MAT_s010  = 0
 w_k1_MAT_s011_return_constraint: 1.78358 w_k1_MAT_s01 - w_k1_MAT_s011
                                 + 0.99 b_k1_MAT_s011 - 1.01 s_k1_MAT_s011  = 0
 w_k1_MAT_s012_return_constraint: 0.631219 w_k1_MAT_s01 - w_k1_MAT_s012
                                 + 0.99 b_k1_MAT_s012 - 1.01 s_k1_MAT_s012  = 0
 w_k1_MAT_s013_return_constraint: 1.18444 w_k1_MAT_s01 - w_k1_MAT_s013
                                 + 0.99 b_k1_MAT_s013 - 1.01 s_k1_MAT_s013  = 0
 w_k1_MAT_s014_return_constraint: 0.801337 w_k1_MAT_s01 - w_k1_MAT_s014
                                 + 0.99 b_k1_MAT_s014 - 1.01 s_k1_MAT_s014  = 0
 w_k1_MAT_s015_return_constraint: 1.2333 w_k1_MAT_s01 - w_k1_MAT_s015
                                 + 0.99 b_k1_MAT_s015 - 1.01 s_k1_MAT_s015  = 0
 w_k1_MAT_s016_return_constraint: 1.09395 w_k1_MAT_s01 - w_k1_MAT_s016
                                 + 0.99 b_k1_MAT_s016 - 1.01 s_k1_MAT_s016  = 0
 w_k1_MAT_s017_return_constraint: 0.786594 w_k1_MAT_s01 - w_k1_MAT_s017
                                 + 0.99 b_k1_MAT_s017 - 1.01 s_k1_MAT_s017  = 0
 w_k1_MAT_s100_return_constraint: 0.825349 w_k1_MAT_s10 - w_k1_MAT_s100
                                 + 0.99 b_k1_MAT_s100 - 1.01 s_k1_MAT_s100  = 0
 w_k1_MAT_s101_return_constraint: 0.422951 w_k1_MAT_s10 - w_k1_MAT_s101
                                 + 0.99 b_k1_MAT_s101 - 1.01 s_k1_MAT_s101  = 0
 w_k1_MAT_s102_return_constraint: 1.2537 w_k1_MAT_s10 - w_k1_MAT_s102
                                 + 0.99 b_k1_MAT_s102 - 1.01 s_k1_MAT_s102  = 0
 w_k1_MAT_s103_return_constraint: 1.70401 w_k1_MAT_s10 - w_k1_MAT_s103
                                 + 0.99 b_k1_MAT_s103 - 1.01 s_k1_MAT_s103  = 0
 w_k1_MAT_s104_return_constraint: 0.66622 w_k1_MAT_s10 - w_k1_MAT_s104
                                 + 0.99 b_k1_MAT_s104 - 1.01 s_k1_MAT_s104  = 0
 w_k1_MAT_s105_return_constraint: 0.734859 w_k1_MAT_s10 - w_k1_MAT_s105
                                 + 0.99 b_k1_MAT_s105 - 1.01 s_k1_MAT_s105  = 0
 w_k1_MAT_s106_return_constraint: 0.783 w_k1_MAT_s10 - w_k1_MAT_s106
                                 + 0.99 b_k1_MAT_s106 - 1.01 s_k1_MAT_s106  = 0
 w_k1_MAT_s107_return_constraint: 0.7302 w_k1_MAT_s10 - w_k1_MAT_s107
                                 + 0.99 b_k1_MAT_s107 - 1.01 s_k1_MAT_s107  = 0
 w_k1_MAT_s110_return_constraint: 0.609387 w_k1_MAT_s11 - w_k1_MAT_s110
                                 + 0.99 b_k1_MAT_s110 - 1.01 s_k1_MAT_s110  = 0
 w_k1_MAT_s111_return_constraint: 0.96307 w_k1_MAT_s11 - w_k1_MAT_s111
                                 + 0.99 b_k1_MAT_s111 - 1.01 s_k1_MAT_s111  = 0
 w_k1_MAT_s112_return_constraint: 0.449086 w_k1_MAT_s11 - w_k1_MAT_s112
                                 + 0.99 b_k1_MAT_s112 - 1.01 s_k1_MAT_s112  = 0
 w_k1_MAT_s113_return_constraint: 1.39766 w_k1_MAT_s11 - w_k1_MAT_s113
                                 + 0.99 b_k1_MAT_s113 - 1.01 s_k1_MAT_s113  = 0
 w_k1_MAT_s114_return_constraint: 0.780722 w_k1_MAT_s11 - w_k1_MAT_s114
                                 + 0.99 b_k1_MAT_s114 - 1.01 s_k1_MAT_s114  = 0
 w_k1_MAT_s115_return_constraint: 0.801424 w_k1_MAT_s11 - w_k1_MAT_s115
                                 + 0.99 b_k1_MAT_s115 - 1.01 s_k1_MAT_s115  = 0
 w_k1_MAT_s116_return_constraint: 1.04444 w_k1_MAT_s11 - w_k1_MAT_s116
                                 + 0.99 b_k1_MAT_s116 - 1.01 s_k1_MAT_s116  = 0
 w_k1_MAT_s117_return_constraint: 1.29531 w_k1_MAT_s11 - w_k1_MAT_s117
                                 + 0.99 b_k1_MAT_s117 - 1.01 s_k1_MAT_s117  = 0
 w_k1_MSFT_s0_return_constraint: 1.18203 w_0_MSFT - w_k1_MSFT_s0
                                 + 0.99 b_k1_MSFT_s0 - 1.01 s_k1_MSFT_s0  = 0
 w_k1_MSFT_s1_return_constraint: 0.959145 w_0_MSFT - w_k1_MSFT_s1
                                 + 0.99 b_k1_MSFT_s1 - 1.01 s_k1_MSFT_s1  = 0
 w_k1_MSFT_s00_return_constraint: 1.13484 w_k1_MSFT_s0 - w_k1_MSFT_s00
                                 + 0.99 b_k1_MSFT_s00 - 1.01 s_k1_MSFT_s00  = 0
 w_k1_MSFT_s01_return_constraint: 0.841284 w_k1_MSFT_s0 - w_k1_MSFT_s01
                                 + 0.99 b_k1_MSFT_s01 - 1.01 s_k1_MSFT_s01  = 0
 w_k1_MSFT_s10_return_constraint: 0.830419 w_k1_MSFT_s1 - w_k1_MSFT_s10
                                 + 0.99 b_k1_MSFT_s10 - 1.01 s_k1_MSFT_s10  = 0
 w_k1_MSFT_s11_return_constraint: 1.12411 w_k1_MSFT_s1 - w_k1_MSFT_s11
                                 + 0.99 b_k1_MSFT_s11 - 1.01 s_k1_MSFT_s11  = 0
 w_k1_MSFT_s000_return_constraint: 1.01759 w_k1_MSFT_s00 - w_k1_MSFT_s000
                                 + 0.99 b_k1_MSFT_s000 - 1.01 s_k1_MSFT_s000
                                  = 0
 w_k1_MSFT_s001_return_constraint: 1.39556 w_k1_MSFT_s00 - w_k1_MSFT_s001
                                 + 0.99 b_k1_MSFT_s001 - 1.01 s_k1_MSFT_s001
                                  = 0
 w_k1_MSFT_s002_return_constraint: 0.973586 w_k1_MSFT_s00 - w_k1_MSFT_s002
                                 + 0.99 b_k1_MSFT_s002 - 1.01 s_k1_MSFT_s002
                                  = 0
 w_k1_MSFT_s003_return_constraint: 1.24854 w_k1_MSFT_s00 - w_k1_MSFT_s003
                                 + 0.99 b_k1_MSFT_s003 - 1.01 s_k1_MSFT_s003
                                  = 0
 w_k1_MSFT_s004_return_constraint: 1.0949 w_k1_MSFT_s00 - w_k1_MSFT_s004
                                 + 0.99 b_k1_MSFT_s004 - 1.01 s_k1_MSFT_s004
                                  = 0
 w_k1_MSFT_s005_return_constraint: 0.790635 w_k1_MSFT_s00 - w_k1_MSFT_s005
                                 + 0.99 b_k1_MSFT_s005 - 1.01 s_k1_MSFT_s005
                                  = 0
 w_k1_MSFT_s006_return_constraint: 1.10927 w_k1_MSFT_s00 - w_k1_MSFT_s006
                                 + 0.99 b_k1_MSFT_s006 - 1.01 s_k1_MSFT_s006
                                  = 0
 w_k1_MSFT_s007_return_constraint: 0.806296 w_k1_MSFT_s00 - w_k1_MSFT_s007
                                 + 0.99 b_k1_MSFT_s007 - 1.01 s_k1_MSFT_s007
                                  = 0
 w_k1_MSFT_s010_return_constraint: 0.751828 w_k1_MSFT_s01 - w_k1_MSFT_s010
                                 + 0.99 b_k1_MSFT_s010 - 1.01 s_k1_MSFT_s010
                                  = 0
 w_k1_MSFT_s011_return_constraint: 1.23433 w_k1_MSFT_s01 - w_k1_MSFT_s011
                                 + 0.99 b_k1_MSFT_s011 - 1.01 s_k1_MSFT_s011
                                  = 0
 w_k1_MSFT_s012_return_constraint: 0.901864 w_k1_MSFT_s01 - w_k1_MSFT_s012
                                 + 0.99 b_k1_MSFT_s012 - 1.01 s_k1_MSFT_s012
                                  = 0
 w_k1_MSFT_s013_return_constraint: 0.95825 w_k1_MSFT_s01 - w_k1_MSFT_s013
                                 + 0.99 b_k1_MSFT_s013 - 1.01 s_k1_MSFT_s013
                                  = 0
 w_k1_MSFT_s014_return_constraint: 1.14147 w_k1_MSFT_s01 - w_k1_MSFT_s014
                                 + 0.99 b_k1_MSFT_s014 - 1.01 s_k1_MSFT_s014
                                  = 0
 w_k1_MSFT_s015_return_constraint: 0.999227 w_k1_MSFT_s01 - w_k1_MSFT_s015
                                 + 0.99 b_k1_MSFT_s015 - 1.01 s_k1_MSFT_s015
                                  = 0
 w_k1_MSFT_s016_return_constraint: 1.21188 w_k1_MSFT_s01 - w_k1_MSFT_s016
                                 + 0.99 b_k1_MSFT_s016 - 1.01 s_k1_MSFT_s016
                                  = 0
 w_k1_MSFT_s017_return_constraint: 1.02822 w_k1_MSFT_s01 - w_k1_MSFT_s017
                                 + 0.99 b_k1_MSFT_s017 - 1.01 s_k1_MSFT_s017
                                  = 0
 w_k1_MSFT_s100_return_constraint: 0.92196 w_k1_MSFT_s10 - w_k1_MSFT_s100
                                 + 0.99 b_k1_MSFT_s100 - 1.01 s_k1_MSFT_s100
                                  = 0
 w_k1_MSFT_s101_return_constraint: 0.98546 w_k1_MSFT_s10 - w_k1_MSFT_s101
                                 + 0.99 b_k1_MSFT_s101 - 1.01 s_k1_MSFT_s101
                                  = 0
 w_k1_MSFT_s102_return_constraint: 1.10598 w_k1_MSFT_s10 - w_k1_MSFT_s102
                                 + 0.99 b_k1_MSFT_s102 - 1.01 s_k1_MSFT_s102
                                  = 0
 w_k1_MSFT_s103_return_constraint: 1.35158 w_k1_MSFT_s10 - w_k1_MSFT_s103
                                 + 0.99 b_k1_MSFT_s103 - 1.01 s_k1_MSFT_s103
                                  = 0
 w_k1_MSFT_s104_return_constraint: 1.24108 w_k1_MSFT_s10 - w_k1_MSFT_s104
                                 + 0.99 b_k1_MSFT_s104 - 1.01 s_k1_MSFT_s104
                                  = 0
 w_k1_MSFT_s105_return_constraint: 0.788511 w_k1_MSFT_s10 - w_k1_MSFT_s105
                                 + 0.99 b_k1_MSFT_s105 - 1.01 s_k1_MSFT_s105
                                  = 0
 w_k1_MSFT_s106_return_constraint: 1.0778 w_k1_MSFT_s10 - w_k1_MSFT_s106
                                 + 0.99 b_k1_MSFT_s106 - 1.01 s_k1_MSFT_s106
                                  = 0
 w_k1_MSFT_s107_return_constraint: 1.37775 w_k1_MSFT_s10 - w_k1_MSFT_s107
                                 + 0.99 b_k1_MSFT_s107 - 1.01 s_k1_MSFT_s107
                                  = 0
 w_k1_MSFT_s110_return_constraint: 1.06232 w_k1_MSFT_s11 - w_k1_MSFT_s110
                                 + 0.99 b_k1_MSFT_s110 - 1.01 s_k1_MSFT_s110
                                  = 0
 w_k1_MSFT_s111_return_constraint: 0.94547 w_k1_MSFT_s11 - w_k1_MSFT_s111
                                 + 0.99 b_k1_MSFT_s111 - 1.01 s_k1_MSFT_s111
                                  = 0
 w_k1_MSFT_s112_return_constraint: 0.965161 w_k1_MSFT_s11 - w_k1_MSFT_s112
                                 + 0.99 b_k1_MSFT_s112 - 1.01 s_k1_MSFT_s112
                                  = 0
 w_k1_MSFT_s113_return_constraint: 1.1418 w_k1_MSFT_s11 - w_k1_MSFT_s113
                                 + 0.99 b_k1_MSFT_s113 - 1.01 s_k1_MSFT_s113
                                  = 0
 w_k1_MSFT_s114_return_constraint: 1.04084 w_k1_MSFT_s11 - w_k1_MSFT_s114
                                 + 0.99 b_k1_MSFT_s114 - 1.01 s_k1_MSFT_s114
                                  = 0
 w_k1_MSFT_s115_return_constraint: 1.23809 w_k1_MSFT_s11 - w_k1_MSFT_s115
                                 + 0.99 b_k1_MSFT_s115 - 1.01 s_k1_MSFT_s115
                                  = 0
 w_k1_MSFT_s116_return_constraint: 1.14693 w_k1_MSFT_s11 - w_k1_MSFT_s116
                                 + 0.99 b_k1_MSFT_s116 - 1.01 s_k1_MSFT_s116
                                  = 0
 w_k1_MSFT_s117_return_constraint: 0.818864 w_k1_MSFT_s11 - w_k1_MSFT_s117
                                 + 0.99 b_k1_MSFT_s117 - 1.01 s_k1_MSFT_s117
                                  = 0
 w_k1_PG_s0_return_constraint:   1.08547 w_0_PG - w_k1_PG_s0 + 0.99 b_k1_PG_s0
                                 - 1.01 s_k1_PG_s0  = 0
 w_k1_PG_s1_return_constraint:   0.954015 w_0_PG - w_k1_PG_s1 + 0.99 b_k1_PG_s1
                                 - 1.01 s_k1_PG_s1  = 0
 w_k1_PG_s00_return_constraint:  1.1263 w_k1_PG_s0 - w_k1_PG_s00
                                 + 0.99 b_k1_PG_s00 - 1.01 s_k1_PG_s00  = 0
 w_k1_PG_s01_return_constraint:  0.864304 w_k1_PG_s0 - w_k1_PG_s01
                                 + 0.99 b_k1_PG_s01 - 1.01 s_k1_PG_s01  = 0
 w_k1_PG_s10_return_constraint:  0.899907 w_k1_PG_s1 - w_k1_PG_s10
                                 + 0.99 b_k1_PG_s10 - 1.01 s_k1_PG_s10  = 0
 w_k1_PG_s11_return_constraint:  1.14556 w_k1_PG_s1 - w_k1_PG_s11
                                 + 0.99 b_k1_PG_s11 - 1.01 s_k1_PG_s11  = 0
 w_k1_PG_s000_return_constraint: 0.993403 w_k1_PG_s00 - w_k1_PG_s000
                                 + 0.99 b_k1_PG_s000 - 1.01 s_k1_PG_s000  = 0
 w_k1_PG_s001_return_constraint: 1.28659 w_k1_PG_s00 - w_k1_PG_s001
                                 + 0.99 b_k1_PG_s001 - 1.01 s_k1_PG_s001  = 0
 w_k1_PG_s002_return_constraint: 1.03719 w_k1_PG_s00 - w_k1_PG_s002
                                 + 0.99 b_k1_PG_s002 - 1.01 s_k1_PG_s002  = 0
 w_k1_PG_s003_return_constraint: 1.18425 w_k1_PG_s00 - w_k1_PG_s003
                                 + 0.99 b_k1_PG_s003 - 1.01 s_k1_PG_s003  = 0
 w_k1_PG_s004_return_constraint: 0.9642 w_k1_PG_s00 - w_k1_PG_s004
                                 + 0.99 b_k1_PG_s004 - 1.01 s_k1_PG_s004  = 0
 w_k1_PG_s005_return_constraint: 0.756334 w_k1_PG_s00 - w_k1_PG_s005
                                 + 0.99 b_k1_PG_s005 - 1.01 s_k1_PG_s005  = 0
 w_k1_PG_s006_return_constraint: 1.14866 w_k1_PG_s00 - w_k1_PG_s006
                                 + 0.99 b_k1_PG_s006 - 1.01 s_k1_PG_s006  = 0
 w_k1_PG_s007_return_constraint: 0.954893 w_k1_PG_s00 - w_k1_PG_s007
                                 + 0.99 b_k1_PG_s007 - 1.01 s_k1_PG_s007  = 0
 w_k1_PG_s010_return_constraint: 0.889005 w_k1_PG_s01 - w_k1_PG_s010
                                 + 0.99 b_k1_PG_s010 - 1.01 s_k1_PG_s010  = 0
 w_k1_PG_s011_return_constraint: 1.23075 w_k1_PG_s01 - w_k1_PG_s011
                                 + 0.99 b_k1_PG_s011 - 1.01 s_k1_PG_s011  = 0
 w_k1_PG_s012_return_constraint: 0.903756 w_k1_PG_s01 - w_k1_PG_s012
                                 + 0.99 b_k1_PG_s012 - 1.01 s_k1_PG_s012  = 0
 w_k1_PG_s013_return_constraint: 1.04497 w_k1_PG_s01 - w_k1_PG_s013
                                 + 0.99 b_k1_PG_s013 - 1.01 s_k1_PG_s013  = 0
 w_k1_PG_s014_return_constraint: 1.07249 w_k1_PG_s01 - w_k1_PG_s014
                                 + 0.99 b_k1_PG_s014 - 1.01 s_k1_PG_s014  = 0
 w_k1_PG_s015_return_constraint: 0.981832 w_k1_PG_s01 - w_k1_PG_s015
                                 + 0.99 b_k1_PG_s015 - 1.01 s_k1_PG_s015  = 0
 w_k1_PG_s016_return_constraint: 1.22748 w_k1_PG_s01 - w_k1_PG_s016
                                 + 0.99 b_k1_PG_s016 - 1.01 s_k1_PG_s016  = 0
 w_k1_PG_s017_return_constraint: 0.924477 w_k1_PG_s01 - w_k1_PG_s017
                                 + 0.99 b_k1_PG_s017 - 1.01 s_k1_PG_s017  = 0
 w_k1_PG_s100_return_constraint: 0.968268 w_k1_PG_s10 - w_k1_PG_s100
                                 + 0.99 b_k1_PG_s100 - 1.01 s_k1_PG_s100  = 0
 w_k1_PG_s101_return_constraint: 0.813573 w_k1_PG_s10 - w_k1_PG_s101
                                 + 0.99 b_k1_PG_s101 - 1.01 s_k1_PG_s101  = 0
 w_k1_PG_s102_return_constraint: 1.13644 w_k1_PG_s10 - w_k1_PG_s102
                                 + 0.99 b_k1_PG_s102 - 1.01 s_k1_PG_s102  = 0
 w_k1_PG_s103_return_constraint: 1.27295 w_k1_PG_s10 - w_k1_PG_s103
                                 + 0.99 b_k1_PG_s103 - 1.01 s_k1_PG_s103  = 0
 w_k1_PG_s104_return_constraint: 1.04701 w_k1_PG_s10 - w_k1_PG_s104
                                 + 0.99 b_k1_PG_s104 - 1.01 s_k1_PG_s104  = 0
 w_k1_PG_s105_return_constraint: 0.768283 w_k1_PG_s10 - w_k1_PG_s105
                                 + 0.99 b_k1_PG_s105 - 1.01 s_k1_PG_s105  = 0
 w_k1_PG_s106_return_constraint: 0.995784 w_k1_PG_s10 - w_k1_PG_s106
                                 + 0.99 b_k1_PG_s106 - 1.01 s_k1_PG_s106  = 0
 w_k1_PG_s107_return_constraint: 1.20398 w_k1_PG_s10 - w_k1_PG_s107
                                 + 0.99 b_k1_PG_s107 - 1.01 s_k1_PG_s107  = 0
 w_k1_PG_s110_return_constraint: 0.913036 w_k1_PG_s11 - w_k1_PG_s110
                                 + 0.99 b_k1_PG_s110 - 1.01 s_k1_PG_s110  = 0
 w_k1_PG_s111_return_constraint: 0.980022 w_k1_PG_s11 - w_k1_PG_s111
                                 + 0.99 b_k1_PG_s111 - 1.01 s_k1_PG_s111  = 0
 w_k1_PG_s112_return_constraint: 0.912626 w_k1_PG_s11 - w_k1_PG_s112
                                 + 0.99 b_k1_PG_s112 - 1.01 s_k1_PG_s112  = 0
 w_k1_PG_s113_return_constraint: 1.1172 w_k1_PG_s11 - w_k1_PG_s113
                                 + 0.99 b_k1_PG_s113 - 1.01 s_k1_PG_s113  = 0
 w_k1_PG_s114_return_constraint: 0.917809 w_k1_PG_s11 - w_k1_PG_s114
                                 + 0.99 b_k1_PG_s114 - 1.01 s_k1_PG_s114  = 0
 w_k1_PG_s115_return_constraint: 1.1391 w_k1_PG_s11 - w_k1_PG_s115
                                 + 0.99 b_k1_PG_s115 - 1.01 s_k1_PG_s115  = 0
 w_k1_PG_s116_return_constraint: 1.05399 w_k1_PG_s11 - w_k1_PG_s116
                                 + 0.99 b_k1_PG_s116 - 1.01 s_k1_PG_s116  = 0
 w_k1_PG_s117_return_constraint: 0.957156 w_k1_PG_s11 - w_k1_PG_s117
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
 k1_wcvar_constraint:            wcvar - var_1 - 0.243001154732859 z_k1_s0000
                                 - 0.148000578984076 z_k1_s0001
                                 - 0.346002712030902 z_k1_s0002
                                 - 0.150000700570847 z_k1_s0003
                                 - 0.720004155063546 z_k1_s0004
                                 - 0.309001205478901 z_k1_s0005
                                 - 0.402002650045264 z_k1_s0006
                                 - 0.158000691715756 z_k1_s0007
                                 - 0.0730000494572359 z_k1_s0010
                                 - 0.104000094069232 z_k1_s0011
                                 - 0.0860000755784013 z_k1_s0012
                                 - 0.108000110944982 z_k1_s0013
                                 - 0.310000446119956 z_k1_s0014
                                 - 0.233000149986788 z_k1_s0015
                                 - 0.117000464890671 z_k1_s0016
                                 - 0.118000095684312 z_k1_s0017
                                 - 0.711999542488917 z_k1_s0020
                                 - 0.364000939679329 z_k1_s0021
                                 - 1.22800092238609 z_k1_s0022
                                 - 0.277000472975256 z_k1_s0023
                                 - 0.730000539464023 z_k1_s0024
                                 - 0.42100083276046 z_k1_s0025
                                 - 0.514001087788451 z_k1_s0026
                                 - 1.19400085254794 z_k1_s0027
                                 - 0.157999514868295 z_k1_s0030
                                 - 0.164999654894198 z_k1_s0031
                                 - 0.12400009251463 z_k1_s0032
                                 - 0.134999601640641 z_k1_s0033
                                 - 0.105000058787377 z_k1_s0034
                                 - 0.064999987782634 z_k1_s0035
                                 - 0.310000550286758 z_k1_s0036
                                 - 0.214000635075522 z_k1_s0037
                                 - 0.292000278815363 z_k1_s0040
                                 - 0.47000021981209 z_k1_s0041
                                 - 0.797000101174602 z_k1_s0042
                                 - 0.208000381781054 z_k1_s0043
                                 - 0.517001457394944 z_k1_s0044
                                 - 0.221000329667268 z_k1_s0045
                                 - 0.224000434525783 z_k1_s0046
                                 - 0.309999692364809 z_k1_s0047
                                 - 0.250000095697782 z_k1_s0050
                                 - 0.235998985749787 z_k1_s0051
                                 - 0.243998347437905 z_k1_s0052
                                 - 0.239998666593846 z_k1_s0053
                                 - 0.168999432225465 z_k1_s0054
                                 - 0.513999074349771 z_k1_s0055
                                 - 0.352998558413666 z_k1_s0056
                                 - 0.220999291787051 z_k1_s0057
                                 - 0.27199982236238 z_k1_s0060
                                 - 0.217999543238692 z_k1_s0061
                                 - 0.188000834060127 z_k1_s0062
                                 - 0.200000689531878 z_k1_s0063
                                 - 0.2949993904331 z_k1_s0064
                                 - 0.129000026472696 z_k1_s0065
                                 - 0.449000943823538 z_k1_s0066
                                 - 0.108000186447052 z_k1_s0067
                                 - 0.679999394537124 z_k1_s0070
                                 - 0.314999979883997 z_k1_s0071
                                 - 0.490000796974752 z_k1_s0072
                                 - 0.424000033564148 z_k1_s0073
                                 - 0.866000571812879 z_k1_s0074
                                 - 0.601998449970065 z_k1_s0075
                                 - 0.886998675812625 z_k1_s0076
                                 - 0.394999634594365 z_k1_s0077
                                 - 0.309999080417254 z_k1_s0100
                                 - 0.154999271109867 z_k1_s0101
                                 - 0.469997130091024 z_k1_s0102
                                 - 0.450998757642081 z_k1_s0103
                                 - 0.196999129160026 z_k1_s0104
                                 - 0.185999179151592 z_k1_s0105
                                 - 0.729997660918849 z_k1_s0106
                                 - 0.191999005102324 z_k1_s0107
                                 - 0.733000160432092 z_k1_s0110
                                 - 0.447001853036054 z_k1_s0111
                                 - 0.382999119487363 z_k1_s0112
                                 - 0.490000709321293 z_k1_s0113
                                 - 0.255000384401925 z_k1_s0114
                                 - 0.804000446395298 z_k1_s0115
                                 - 0.408001063843876 z_k1_s0116
                                 - 0.220000280307452 z_k1_s0117
                                 - 0.166999478736169 z_k1_s0120
                                 - 0.497998935650223 z_k1_s0121
                                 - 0.181999427920866 z_k1_s0122
                                 - 0.271000327373433 z_k1_s0123
                                 - 0.279000033392672 z_k1_s0124
                                 - 0.247999782571943 z_k1_s0125
                                 - 0.370999988604745 z_k1_s0126
                                 - 0.20799947328064 z_k1_s0127
                                 - 0.350999244791834 z_k1_s0130
                                 - 0.170999785993259 z_k1_s0131
                                 - 0.196000635483537 z_k1_s0132
                                 - 0.236000393950043 z_k1_s0133
                                 - 0.0949998811073663 z_k1_s0134
                                 - 0.311999935036405 z_k1_s0135
                                 - 0.284998733823271 z_k1_s0136
                                 - 0.172999773916585 z_k1_s0137
                                 - 0.102999999355418 z_k1_s0140
                                 - 0.141999853820205 z_k1_s0141
                                 - 0.445000351583812 z_k1_s0142
                                 - 0.209000661361447 z_k1_s0143
                                 - 0.228000721485215 z_k1_s0144
                                 - 0.325999494756379 z_k1_s0145
                                 - 0.193999173106969 z_k1_s0146
                                 - 0.177999775633811 z_k1_s0147
                                 - 0.271000157205245 z_k1_s0150
                                 - 0.29000021013848 z_k1_s0151
                                 - 0.235000007094873 z_k1_s0152
                                 - 0.853002110739294 z_k1_s0153
                                 - 0.396000073713101 z_k1_s0154
                                 - 0.262000198552701 z_k1_s0155
                                 - 0.205000039752995 z_k1_s0156
                                 - 0.642998868843971 z_k1_s0157
                                 - 0.422998222628639 z_k1_s0160
                                 - 0.650997838314332 z_k1_s0161
                                 - 0.25499960058528 z_k1_s0162
                                 - 0.443998339283724 z_k1_s0163
                                 - 0.159999857016129 z_k1_s0164
                                 - 0.417000665581458 z_k1_s0165
                                 - 0.284999808506782 z_k1_s0166
                                 - 0.253999718844418 z_k1_s0167
                                 - 0.22100007993932 z_k1_s0170
                                 - 0.498001067939994 z_k1_s0171
                                 - 0.473001657913402 z_k1_s0172
                                 - 0.508000831950631 z_k1_s0173
                                 - 0.165000007978637 z_k1_s0174
                                 - 0.135000158546279 z_k1_s0175
                                 - 0.650001382704803 z_k1_s0176
                                 - 0.137000111348407 z_k1_s0177
                                 - 0.540000798919154 z_k1_s1000
                                 - 0.903001414787179 z_k1_s1001
                                 - 2.16800467875967 z_k1_s1002
                                 - 0.771001534763091 z_k1_s1003
                                 - 0.497000866662877 z_k1_s1004
                                 - 2.15800755391477 z_k1_s1005
                                 - 1.58100226024093 z_k1_s1006
                                 - 0.84000155803039 z_k1_s1007
                                 - 0.686999359421771 z_k1_s1010
                                 - 0.309999579276194 z_k1_s1011
                                 - 0.381000358316908 z_k1_s1012
                                 - 0.298998718805066 z_k1_s1013
                                 - 0.382001232359772 z_k1_s1014
                                 - 0.531000154740873 z_k1_s1015
                                 - 0.141000100438539 z_k1_s1016
                                 - 0.187000038008555 z_k1_s1017
                                 - 1.96900525720164 z_k1_s1020
                                 - 1.94499960813099 z_k1_s1021
                                 - 0.973005089445858 z_k1_s1022
                                 - 0.421000104114871 z_k1_s1023
                                 - 0.733000964678738 z_k1_s1024
                                 - 1.00300066702553 z_k1_s1025
                                 - 0.756000636492126 z_k1_s1026
                                 - 1.38699918208684 z_k1_s1027
                                 - 0.61399902742433 z_k1_s1030
                                 - 0.230000068030705 z_k1_s1031
                                 - 0.257000224609993 z_k1_s1032
                                 - 0.506999054521362 z_k1_s1033
                                 - 1.1150013146621 z_k1_s1034
                                 - 0.63800199256208 z_k1_s1035
                                 - 0.295999876757731 z_k1_s1036
                                 - 0.317000086852752 z_k1_s1037
                                 - 0.377999130740558 z_k1_s1040
                                 - 0.488000171540653 z_k1_s1041
                                 - 0.489999967346577 z_k1_s1042
                                 - 1.1759981036356 z_k1_s1043
                                 - 0.316999450172262 z_k1_s1044
                                 - 0.686998034191983 z_k1_s1045
                                 - 0.619000431800092 z_k1_s1046
                                 - 0.389999269073244 z_k1_s1047
                                 - 0.469999780443186 z_k1_s1050
                                 - 0.35500002082511 z_k1_s1051
                                 - 1.32800006292605 z_k1_s1052
                                 - 0.740998141331356 z_k1_s1053
                                 - 0.946000325088795 z_k1_s1054
                                 - 0.364000103729099 z_k1_s1055
                                 - 0.644000142620728 z_k1_s1056
                                 - 0.468999653076115 z_k1_s1057
                                 - 0.625999456028879 z_k1_s1060
                                 - 0.705999429131608 z_k1_s1061
                                 - 0.159000113291659 z_k1_s1062
                                 - 0.329999889048755 z_k1_s1063
                                 - 0.505999496374786 z_k1_s1064
                                 - 0.205000097825727 z_k1_s1065
                                 - 0.163999944860593 z_k1_s1066
                                 - 0.638999284908087 z_k1_s1067
                                 - 0.245000132653774 z_k1_s1070
                                 - 0.489998788907296 z_k1_s1071
                                 - 0.499001139448856 z_k1_s1072
                                 - 0.154999880542149 z_k1_s1073
                                 - 0.52500054790099 z_k1_s1074
                                 - 0.728998461833552 z_k1_s1075
                                 - 0.669001246559641 z_k1_s1076
                                 - 0.37899932689977 z_k1_s1077
                                 - 0.0920001161383752 z_k1_s1100
                                 - 0.147000042351513 z_k1_s1101
                                 - 0.171000129935445 z_k1_s1102
                                 - 0.345999615668115 z_k1_s1103
                                 - 0.447000258750538 z_k1_s1104
                                 - 0.267999880020416 z_k1_s1105
                                 - 0.326999820830876 z_k1_s1106
                                 - 0.398000903433466 z_k1_s1107
                                 - 0.257999328365184 z_k1_s1110
                                 - 0.352999237561326 z_k1_s1111
                                 - 0.363000535789652 z_k1_s1112
                                 - 0.58099777723477 z_k1_s1113
                                 - 0.240999606171621 z_k1_s1114
                                 - 0.321998319046754 z_k1_s1115
                                 - 0.584998296526101 z_k1_s1116
                                 - 0.402999516716478 z_k1_s1117
                                 - 0.118000027558883 z_k1_s1120
                                 - 0.122999569193043 z_k1_s1121
                                 - 0.0319999584566792 z_k1_s1122
                                 - 0.110000037944713 z_k1_s1123
                                 - 0.112999762925209 z_k1_s1124
                                 - 0.029000016576329 z_k1_s1125
                                 - 0.111999854598377 z_k1_s1126
                                 - 0.0860000691022037 z_k1_s1127
                                 - 0.74699899802459 z_k1_s1130
                                 - 0.319999592958311 z_k1_s1131
                                 - 0.232999962560106 z_k1_s1132
                                 - 0.174999932761478 z_k1_s1133
                                 - 0.312000846033564 z_k1_s1134
                                 - 0.481999076168653 z_k1_s1135
                                 - 0.266999884786948 z_k1_s1136
                                 - 0.225999954201654 z_k1_s1137
                                 - 0.125000294937458 z_k1_s1140
                                 - 0.119000025540358 z_k1_s1141
                                 - 0.103000016148375 z_k1_s1142
                                 - 0.0629999926684166 z_k1_s1143
                                 - 0.0629999926684166 z_k1_s1144
                                 - 0.153999824522733 z_k1_s1145
                                 - 0.0430000163784516 z_k1_s1146
                                 - 0.03900004948047 z_k1_s1147
                                 - 0.0809997374380516 z_k1_s1150
                                 - 0.162000106473933 z_k1_s1151
                                 - 0.113999618772113 z_k1_s1152
                                 - 0.119999640259976 z_k1_s1153
                                 - 0.383998848831923 z_k1_s1154
                                 - 0.249999053167295 z_k1_s1155
                                 - 0.199998610936005 z_k1_s1156
                                 - 0.267999591329257 z_k1_s1157
                                 - 0.251000263757606 z_k1_s1160
                                 - 0.137999740185609 z_k1_s1161
                                 - 0.166000520672231 z_k1_s1162
                                 - 0.108000011749526 z_k1_s1163
                                 - 0.0579999741229056 z_k1_s1164
                                 - 0.17700001554228 z_k1_s1165
                                 - 0.288000387865268 z_k1_s1166
                                 - 0.150999386304666 z_k1_s1167
                                 - 0.263000149534122 z_k1_s1170
                                 - 0.102999758216572 z_k1_s1171
                                 - 0.393999375212359 z_k1_s1172
                                 - 0.305998759308435 z_k1_s1173
                                 - 0.322999008380547 z_k1_s1174
                                 - 0.221999212125523 z_k1_s1175
                                 - 0.0969998723319291 z_k1_s1176
                                 - 0.200000012147806 z_k1_s1177 >= 0
 z_k1_s0000_constraint:          var_1 + z_k1_s0000 + 1.58801 w_k1_ADM_s000
                                 + 1.54273 w_k1_AXP_s000
                                 + 1.62238 w_k1_DIS_s000
                                 + 1.37986 w_k1_FDX_s000
                                 + 1.26384 w_k1_IBM_s000
                                 + 1.50393 w_k1_INTC_s000
                                 + 1.35282 w_k1_KO_s000 + 1.2352 w_k1_MAT_s000
                                 + 1.40361 w_k1_MSFT_s000
                                 + 1.21248 w_k1_PG_s000 >= 1
 z_k1_s0001_constraint:          var_1 + z_k1_s0001 + 0.92707 w_k1_ADM_s000
                                 + 0.741865 w_k1_AXP_s000
                                 + 0.641399 w_k1_DIS_s000
                                 + 0.811298 w_k1_FDX_s000
                                 + 0.5989 w_k1_IBM_s000
                                 + 0.926188 w_k1_INTC_s000
                                 + 0.742375 w_k1_KO_s000
                                 + 0.566504 w_k1_MAT_s000
                                 + 0.938706 w_k1_MSFT_s000
                                 + 0.945482 w_k1_PG_s000 >= 1
 z_k1_s0002_constraint:          var_1 + z_k1_s0002 + 0.930849 w_k1_ADM_s000
                                 + 0.996636 w_k1_AXP_s000
                                 + 0.959238 w_k1_DIS_s000
                                 + 0.929298 w_k1_FDX_s000
                                 + 1.38895 w_k1_IBM_s000
                                 + 0.98251 w_k1_INTC_s000
                                 + 1.07108 w_k1_KO_s000 + 1.66617 w_k1_MAT_s000
                                 + 0.969383 w_k1_MSFT_s000
                                 + 1.20716 w_k1_PG_s000 >= 1
 z_k1_s0003_constraint:          var_1 + z_k1_s0003 + 1.09805 w_k1_ADM_s000
                                 + 0.82848 w_k1_AXP_s000
                                 + 0.66056 w_k1_DIS_s000
                                 + 0.801647 w_k1_FDX_s000
                                 + 1.17575 w_k1_IBM_s000
                                 + 0.81422 w_k1_INTC_s000
                                 + 0.944662 w_k1_KO_s000
                                 + 0.901247 w_k1_MAT_s000
                                 + 0.876656 w_k1_MSFT_s000
                                 + 0.861443 w_k1_PG_s000 >= 1
 z_k1_s0004_constraint:          var_1 + z_k1_s0004 + 0.92955 w_k1_ADM_s000
                                 + 0.837786 w_k1_AXP_s000
                                 + 0.888958 w_k1_DIS_s000
                                 + 0.95913 w_k1_FDX_s000
                                 + 0.888905 w_k1_IBM_s000
                                 + 1.02558 w_k1_INTC_s000
                                 + 1.11452 w_k1_KO_s000
                                 + 0.976788 w_k1_MAT_s000
                                 + 1.14038 w_k1_MSFT_s000
                                 + 1.07173 w_k1_PG_s000 >= 1
 z_k1_s0005_constraint:          var_1 + z_k1_s0005 + 1.30524 w_k1_ADM_s000
                                 + 1.06375 w_k1_AXP_s000
                                 + 1.06716 w_k1_DIS_s000
                                 + 1.21787 w_k1_FDX_s000
                                 + 0.78197 w_k1_IBM_s000
                                 + 1.27864 w_k1_INTC_s000
                                 + 0.977172 w_k1_KO_s000
                                 + 0.955774 w_k1_MAT_s000
                                 + 1.34088 w_k1_MSFT_s000
                                 + 1.11223 w_k1_PG_s000 >= 1
 z_k1_s0006_constraint:          var_1 + z_k1_s0006 + 1.17752 w_k1_ADM_s000
                                 + 1.34486 w_k1_AXP_s000
                                 + 1.18125 w_k1_DIS_s000
                                 + 1.19894 w_k1_FDX_s000
                                 + 1.33107 w_k1_IBM_s000
                                 + 1.10961 w_k1_INTC_s000
                                 + 1.12519 w_k1_KO_s000 + 1.73231 w_k1_MAT_s000
                                 + 1.26205 w_k1_MSFT_s000
                                 + 1.21988 w_k1_PG_s000 >= 1
 z_k1_s0007_constraint:          var_1 + z_k1_s0007 + 0.817486 w_k1_ADM_s000
                                 + 0.665545 w_k1_AXP_s000
                                 + 0.615293 w_k1_DIS_s000
                                 + 0.728147 w_k1_FDX_s000
                                 + 0.865789 w_k1_IBM_s000
                                 + 0.813869 w_k1_INTC_s000
                                 + 0.854475 w_k1_KO_s000
                                 + 0.859859 w_k1_MAT_s000
                                 + 0.882318 w_k1_MSFT_s000
                                 + 0.881207 w_k1_PG_s000 >= 1
 z_k1_s0010_constraint:          var_1 + z_k1_s0010 + 0.732119 w_k1_ADM_s001
                                 + 0.602897 w_k1_AXP_s001
                                 + 0.57294 w_k1_DIS_s001
                                 + 0.725749 w_k1_FDX_s001
                                 + 0.595628 w_k1_IBM_s001
                                 + 0.74808 w_k1_INTC_s001
                                 + 0.690925 w_k1_KO_s001
                                 + 0.524884 w_k1_MAT_s001
                                 + 0.804215 w_k1_MSFT_s001
                                 + 0.743775 w_k1_PG_s001 >= 1
 z_k1_s0011_constraint:          var_1 + z_k1_s0011 + 0.969196 w_k1_ADM_s001
                                 + 0.944043 w_k1_AXP_s001
                                 + 0.849215 w_k1_DIS_s001
                                 + 1.04662 w_k1_FDX_s001
                                 + 0.687652 w_k1_IBM_s001
                                 + 1.1063 w_k1_INTC_s001 + 0.7474 w_k1_KO_s001
                                 + 0.62923 w_k1_MAT_s001
                                 + 0.98145 w_k1_MSFT_s001
                                 + 1.09032 w_k1_PG_s001 >= 1
 z_k1_s0012_constraint:          var_1 + z_k1_s0012 + 0.869915 w_k1_ADM_s001
                                 + 0.884051 w_k1_AXP_s001
                                 + 0.651142 w_k1_DIS_s001
                                 + 0.703798 w_k1_FDX_s001
                                 + 1.04353 w_k1_IBM_s001
                                 + 1.08003 w_k1_INTC_s001
                                 + 0.915542 w_k1_KO_s001 + 1.0088 w_k1_MAT_s001
                                 + 0.744803 w_k1_MSFT_s001
                                 + 0.876583 w_k1_PG_s001 >= 1
 z_k1_s0013_constraint:          var_1 + z_k1_s0013 + 0.70337 w_k1_ADM_s001
                                 + 0.955889 w_k1_AXP_s001
                                 + 1.1973 w_k1_DIS_s001
                                 + 0.884654 w_k1_FDX_s001
                                 + 1.51894 w_k1_IBM_s001
                                 + 1.14327 w_k1_INTC_s001
                                 + 1.42785 w_k1_KO_s001 + 1.82947 w_k1_MAT_s001
                                 + 0.979981 w_k1_MSFT_s001
                                 + 1.02321 w_k1_PG_s001 >= 1
 z_k1_s0014_constraint:          var_1 + z_k1_s0014 + 1.03107 w_k1_ADM_s001
                                 + 1.14776 w_k1_AXP_s001
                                 + 1.34334 w_k1_DIS_s001
                                 + 1.05109 w_k1_FDX_s001
                                 + 1.10298 w_k1_IBM_s001
                                 + 1.46496 w_k1_INTC_s001
                                 + 1.30155 w_k1_KO_s001
                                 + 0.971363 w_k1_MAT_s001
                                 + 1.1851 w_k1_MSFT_s001 + 1.20507 w_k1_PG_s001
                                 >= 1
 z_k1_s0015_constraint:          var_1 + z_k1_s0015 + 1.43429 w_k1_ADM_s001
                                 + 1.69517 w_k1_AXP_s001
                                 + 1.19485 w_k1_DIS_s001
                                 + 1.35094 w_k1_FDX_s001
                                 + 0.964092 w_k1_IBM_s001
                                 + 1.60276 w_k1_INTC_s001
                                 + 1.20225 w_k1_KO_s001 + 1.36222 w_k1_MAT_s001
                                 + 1.36246 w_k1_MSFT_s001 + 1.3539 w_k1_PG_s001
                                 >= 1
 z_k1_s0016_constraint:          var_1 + z_k1_s0016 + 0.705812 w_k1_ADM_s001
                                 + 0.816925 w_k1_AXP_s001
                                 + 0.723257 w_k1_DIS_s001
                                 + 0.873045 w_k1_FDX_s001
                                 + 0.68648 w_k1_IBM_s001
                                 + 0.833664 w_k1_INTC_s001
                                 + 0.662987 w_k1_KO_s001
                                 + 0.765243 w_k1_MAT_s001
                                 + 0.852232 w_k1_MSFT_s001
                                 + 0.840218 w_k1_PG_s001 >= 1
 z_k1_s0017_constraint:          var_1 + z_k1_s0017 + 0.748705 w_k1_ADM_s001
                                 + 1.19788 w_k1_AXP_s001
                                 + 0.825212 w_k1_DIS_s001
                                 + 0.90865 w_k1_FDX_s001
                                 + 1.14447 w_k1_IBM_s001
                                 + 1.09714 w_k1_INTC_s001
                                 + 1.05464 w_k1_KO_s001 + 1.56286 w_k1_MAT_s001
                                 + 0.911514 w_k1_MSFT_s001
                                 + 1.08143 w_k1_PG_s001 >= 1
 z_k1_s0020_constraint:          var_1 + z_k1_s0020 + 0.969563 w_k1_ADM_s002
                                 + 1.01098 w_k1_AXP_s002
                                 + 0.703448 w_k1_DIS_s002
                                 + 0.880666 w_k1_FDX_s002
                                 + 0.888525 w_k1_IBM_s002
                                 + 0.835402 w_k1_INTC_s002
                                 + 0.892336 w_k1_KO_s002
                                 + 1.04134 w_k1_MAT_s002
                                 + 0.809088 w_k1_MSFT_s002
                                 + 0.98151 w_k1_PG_s002 >= 1
 z_k1_s0021_constraint:          var_1 + z_k1_s0021 + 1.40531 w_k1_ADM_s002
                                 + 1.61714 w_k1_AXP_s002
                                 + 1.77658 w_k1_DIS_s002
                                 + 1.51737 w_k1_FDX_s002
                                 + 1.31136 w_k1_IBM_s002
                                 + 1.80092 w_k1_INTC_s002
                                 + 1.61357 w_k1_KO_s002 + 1.58936 w_k1_MAT_s002
                                 + 1.73033 w_k1_MSFT_s002
                                 + 1.42551 w_k1_PG_s002 >= 1
 z_k1_s0022_constraint:          var_1 + z_k1_s0022 + 1.07759 w_k1_ADM_s002
                                 + 1.36944 w_k1_AXP_s002
                                 + 1.13133 w_k1_DIS_s002 + 1.0663 w_k1_FDX_s002
                                 + 1.38231 w_k1_IBM_s002
                                 + 1.16718 w_k1_INTC_s002
                                 + 1.31381 w_k1_KO_s002 + 1.38582 w_k1_MAT_s002
                                 + 1.00329 w_k1_MSFT_s002
                                 + 1.05257 w_k1_PG_s002 >= 1
 z_k1_s0023_constraint:          var_1 + z_k1_s0023 + 1.08707 w_k1_ADM_s002
                                 + 0.910599 w_k1_AXP_s002
                                 + 1.03529 w_k1_DIS_s002
                                 + 1.07185 w_k1_FDX_s002
                                 + 0.647762 w_k1_IBM_s002
                                 + 1.17988 w_k1_INTC_s002
                                 + 0.950358 w_k1_KO_s002
                                 + 0.546817 w_k1_MAT_s002
                                 + 1.10189 w_k1_MSFT_s002
                                 + 0.861255 w_k1_PG_s002 >= 1
 z_k1_s0024_constraint:          var_1 + z_k1_s0024 + 0.890029 w_k1_ADM_s002
                                 + 0.639105 w_k1_AXP_s002
                                 + 0.638981 w_k1_DIS_s002
                                 + 0.734094 w_k1_FDX_s002
                                 + 0.751392 w_k1_IBM_s002
                                 + 0.931512 w_k1_INTC_s002
                                 + 0.888758 w_k1_KO_s002
                                 + 0.547529 w_k1_MAT_s002
                                 + 0.882002 w_k1_MSFT_s002
                                 + 0.917648 w_k1_PG_s002 >= 1
 z_k1_s0025_constraint:          var_1 + z_k1_s0025 + 1.29033 w_k1_ADM_s002
                                 + 1.11349 w_k1_AXP_s002
                                 + 1.40391 w_k1_DIS_s002
                                 + 1.24331 w_k1_FDX_s002
                                 + 0.966413 w_k1_IBM_s002
                                 + 1.32806 w_k1_INTC_s002
                                 + 1.32289 w_k1_KO_s002
                                 + 0.992703 w_k1_MAT_s002
                                 + 1.40118 w_k1_MSFT_s002
                                 + 1.05653 w_k1_PG_s002 >= 1
 z_k1_s0026_constraint:          var_1 + z_k1_s0026 + 1.05972 w_k1_ADM_s002
                                 + 1.44087 w_k1_AXP_s002
                                 + 1.07509 w_k1_DIS_s002
                                 + 1.19001 w_k1_FDX_s002
                                 + 0.789877 w_k1_IBM_s002
                                 + 1.37924 w_k1_INTC_s002
                                 + 1.09247 w_k1_KO_s002 + 1.05348 w_k1_MAT_s002
                                 + 1.20921 w_k1_MSFT_s002
                                 + 1.13685 w_k1_PG_s002 >= 1
 z_k1_s0027_constraint:          var_1 + z_k1_s0027 + 1.0023 w_k1_ADM_s002
                                 + 0.918749 w_k1_AXP_s002
                                 + 0.977055 w_k1_DIS_s002
                                 + 0.99859 w_k1_FDX_s002
                                 + 1.01809 w_k1_IBM_s002
                                 + 1.11624 w_k1_INTC_s002
                                 + 0.98246 w_k1_KO_s002
                                 + 0.773395 w_k1_MAT_s002
                                 + 1.01219 w_k1_MSFT_s002
                                 + 0.919388 w_k1_PG_s002 >= 1
 z_k1_s0030_constraint:          var_1 + z_k1_s0030 + 1.20319 w_k1_ADM_s003
                                 + 1.10937 w_k1_AXP_s003
                                 + 1.35182 w_k1_DIS_s003 + 1.1346 w_k1_FDX_s003
                                 + 1.45294 w_k1_IBM_s003
                                 + 1.01889 w_k1_INTC_s003
                                 + 1.47081 w_k1_KO_s003 + 1.31462 w_k1_MAT_s003
                                 + 1.10631 w_k1_MSFT_s003
                                 + 1.28052 w_k1_PG_s003 >= 1
 z_k1_s0031_constraint:          var_1 + z_k1_s0031 + 1.06351 w_k1_ADM_s003
                                 + 0.728093 w_k1_AXP_s003
                                 + 0.803959 w_k1_DIS_s003
                                 + 0.959416 w_k1_FDX_s003
                                 + 0.753368 w_k1_IBM_s003
                                 + 0.815249 w_k1_INTC_s003
                                 + 0.77423 w_k1_KO_s003
                                 + 0.513864 w_k1_MAT_s003
                                 + 0.909137 w_k1_MSFT_s003
                                 + 0.864929 w_k1_PG_s003 >= 1
 z_k1_s0032_constraint:          var_1 + z_k1_s0032 + 1.16893 w_k1_ADM_s003
                                 + 0.837545 w_k1_AXP_s003
                                 + 0.914409 w_k1_DIS_s003
                                 + 1.02763 w_k1_FDX_s003
                                 + 0.994489 w_k1_IBM_s003
                                 + 1.00826 w_k1_INTC_s003
                                 + 1.00824 w_k1_KO_s003
                                 + 0.949361 w_k1_MAT_s003
                                 + 1.04519 w_k1_MSFT_s003
                                 + 1.05008 w_k1_PG_s003 >= 1
 z_k1_s0033_constraint:          var_1 + z_k1_s0033 + 0.777737 w_k1_ADM_s003
                                 + 0.693928 w_k1_AXP_s003
                                 + 0.626375 w_k1_DIS_s003
                                 + 0.759216 w_k1_FDX_s003
                                 + 0.824707 w_k1_IBM_s003
                                 + 0.717391 w_k1_INTC_s003
                                 + 0.721524 w_k1_KO_s003
                                 + 0.618558 w_k1_MAT_s003
                                 + 0.781427 w_k1_MSFT_s003
                                 + 0.729198 w_k1_PG_s003 >= 1
 z_k1_s0034_constraint:          var_1 + z_k1_s0034 + 1.06153 w_k1_ADM_s003
                                 + 1.22186 w_k1_AXP_s003
                                 + 0.849062 w_k1_DIS_s003
                                 + 1.05825 w_k1_FDX_s003
                                 + 1.04805 w_k1_IBM_s003
                                 + 0.946335 w_k1_INTC_s003
                                 + 0.829447 w_k1_KO_s003
                                 + 0.869627 w_k1_MAT_s003
                                 + 0.815968 w_k1_MSFT_s003
                                 + 0.875202 w_k1_PG_s003 >= 1
 z_k1_s0035_constraint:          var_1 + z_k1_s0035 + 1.60066 w_k1_ADM_s003
                                 + 1.65569 w_k1_AXP_s003
                                 + 1.64586 w_k1_DIS_s003
                                 + 1.58082 w_k1_FDX_s003
                                 + 1.37211 w_k1_IBM_s003
                                 + 1.70603 w_k1_INTC_s003
                                 + 1.54381 w_k1_KO_s003 + 1.19503 w_k1_MAT_s003
                                 + 1.39752 w_k1_MSFT_s003
                                 + 1.31791 w_k1_PG_s003 >= 1
 z_k1_s0036_constraint:          var_1 + z_k1_s0036 + 1.34111 w_k1_ADM_s003
                                 + 1.08273 w_k1_AXP_s003
                                 + 1.08402 w_k1_DIS_s003
                                 + 1.24971 w_k1_FDX_s003
                                 + 0.947451 w_k1_IBM_s003
                                 + 0.972417 w_k1_INTC_s003
                                 + 0.970765 w_k1_KO_s003
                                 + 0.960085 w_k1_MAT_s003
                                 + 1.0633 w_k1_MSFT_s003
                                 + 0.990778 w_k1_PG_s003 >= 1
 z_k1_s0037_constraint:          var_1 + z_k1_s0037 + 0.966649 w_k1_ADM_s003
                                 + 0.991318 w_k1_AXP_s003
                                 + 0.924527 w_k1_DIS_s003
                                 + 0.931408 w_k1_FDX_s003
                                 + 1.29004 w_k1_IBM_s003
                                 + 0.996482 w_k1_INTC_s003
                                 + 1.19365 w_k1_KO_s003 + 1.09535 w_k1_MAT_s003
                                 + 0.986601 w_k1_MSFT_s003
                                 + 1.01395 w_k1_PG_s003 >= 1
 z_k1_s0040_constraint:          var_1 + z_k1_s0040 + 1.40329 w_k1_ADM_s004
                                 + 1.1328 w_k1_AXP_s004 + 1.33451 w_k1_DIS_s004
                                 + 1.35328 w_k1_FDX_s004
                                 + 0.725871 w_k1_IBM_s004
                                 + 1.06439 w_k1_INTC_s004
                                 + 1.06677 w_k1_KO_s004 + 1.03637 w_k1_MAT_s004
                                 + 1.15226 w_k1_MSFT_s004
                                 + 1.21927 w_k1_PG_s004 >= 1
 z_k1_s0041_constraint:          var_1 + z_k1_s0041 + 0.776783 w_k1_ADM_s004
                                 + 0.801554 w_k1_AXP_s004
                                 + 1.06659 w_k1_DIS_s004
                                 + 0.977201 w_k1_FDX_s004
                                 + 0.755603 w_k1_IBM_s004
                                 + 0.981505 w_k1_INTC_s004
                                 + 0.980897 w_k1_KO_s004
                                 + 0.731692 w_k1_MAT_s004
                                 + 0.925352 w_k1_MSFT_s004
                                 + 0.988077 w_k1_PG_s004 >= 1
 z_k1_s0042_constraint:          var_1 + z_k1_s0042 + 0.834677 w_k1_ADM_s004
                                 + 0.832216 w_k1_AXP_s004
                                 + 0.746706 w_k1_DIS_s004
                                 + 0.807238 w_k1_FDX_s004
                                 + 0.845939 w_k1_IBM_s004
                                 + 0.925506 w_k1_INTC_s004
                                 + 0.907142 w_k1_KO_s004
                                 + 0.578212 w_k1_MAT_s004
                                 + 0.777801 w_k1_MSFT_s004
                                 + 0.970909 w_k1_PG_s004 >= 1
 z_k1_s0043_constraint:          var_1 + z_k1_s0043 + 0.849824 w_k1_ADM_s004
                                 + 0.926951 w_k1_AXP_s004
                                 + 1.01007 w_k1_DIS_s004
                                 + 0.883053 w_k1_FDX_s004
                                 + 1.52288 w_k1_IBM_s004
                                 + 0.939522 w_k1_INTC_s004
                                 + 1.34979 w_k1_KO_s004 + 1.60322 w_k1_MAT_s004
                                 + 0.868709 w_k1_MSFT_s004
                                 + 1.11097 w_k1_PG_s004 >= 1
 z_k1_s0044_constraint:          var_1 + z_k1_s0044 + 1.0673 w_k1_ADM_s004
                                 + 1.26355 w_k1_AXP_s004
                                 + 1.26339 w_k1_DIS_s004
                                 + 1.20058 w_k1_FDX_s004
                                 + 1.16922 w_k1_IBM_s004
                                 + 1.19656 w_k1_INTC_s004
                                 + 1.06256 w_k1_KO_s004 + 1.21261 w_k1_MAT_s004
                                 + 0.958344 w_k1_MSFT_s004
                                 + 1.09679 w_k1_PG_s004 >= 1
 z_k1_s0045_constraint:          var_1 + z_k1_s0045 + 1.11739 w_k1_ADM_s004
                                 + 1.21658 w_k1_AXP_s004
                                 + 1.54991 w_k1_DIS_s004
                                 + 1.16182 w_k1_FDX_s004
                                 + 1.07374 w_k1_IBM_s004
                                 + 1.04819 w_k1_INTC_s004
                                 + 1.32752 w_k1_KO_s004 + 1.211 w_k1_MAT_s004
                                 + 0.919617 w_k1_MSFT_s004
                                 + 1.06511 w_k1_PG_s004 >= 1
 z_k1_s0046_constraint:          var_1 + z_k1_s0046 + 1.15791 w_k1_ADM_s004
                                 + 0.893806 w_k1_AXP_s004
                                 + 0.957202 w_k1_DIS_s004
                                 + 1.03713 w_k1_FDX_s004
                                 + 0.719465 w_k1_IBM_s004
                                 + 1.03518 w_k1_INTC_s004
                                 + 1.05578 w_k1_KO_s004
                                 + 0.534196 w_k1_MAT_s004
                                 + 1.02689 w_k1_MSFT_s004
                                 + 0.995168 w_k1_PG_s004 >= 1
 z_k1_s0047_constraint:          var_1 + z_k1_s0047 + 0.930398 w_k1_ADM_s004
                                 + 1.1582 w_k1_AXP_s004
                                 + 0.967907 w_k1_DIS_s004
                                 + 0.940891 w_k1_FDX_s004
                                 + 1.22225 w_k1_IBM_s004
                                 + 1.14043 w_k1_INTC_s004
                                 + 1.13441 w_k1_KO_s004 + 1.06661 w_k1_MAT_s004
                                 + 0.878501 w_k1_MSFT_s004
                                 + 0.966495 w_k1_PG_s004 >= 1
 z_k1_s0050_constraint:          var_1 + z_k1_s0050 + 0.847283 w_k1_ADM_s005
                                 + 0.599607 w_k1_AXP_s005
                                 + 0.619337 w_k1_DIS_s005
                                 + 0.724494 w_k1_FDX_s005
                                 + 0.819052 w_k1_IBM_s005
                                 + 0.896943 w_k1_INTC_s005
                                 + 0.740833 w_k1_KO_s005
                                 + 0.77018 w_k1_MAT_s005
                                 + 0.864707 w_k1_MSFT_s005
                                 + 0.851454 w_k1_PG_s005 >= 1
 z_k1_s0051_constraint:          var_1 + z_k1_s0051 + 0.903452 w_k1_ADM_s005
                                 + 0.913815 w_k1_AXP_s005
                                 + 0.840344 w_k1_DIS_s005
                                 + 0.920168 w_k1_FDX_s005
                                 + 0.721374 w_k1_IBM_s005
                                 + 0.901368 w_k1_INTC_s005
                                 + 0.779081 w_k1_KO_s005
                                 + 0.963438 w_k1_MAT_s005
                                 + 0.88802 w_k1_MSFT_s005
                                 + 0.997042 w_k1_PG_s005 >= 1
 z_k1_s0052_constraint:          var_1 + z_k1_s0052 + 1.15726 w_k1_ADM_s005
                                 + 1.05137 w_k1_AXP_s005
                                 + 1.01991 w_k1_DIS_s005
                                 + 1.08062 w_k1_FDX_s005
                                 + 0.754988 w_k1_IBM_s005
                                 + 1.15682 w_k1_INTC_s005
                                 + 1.02245 w_k1_KO_s005
                                 + 0.852556 w_k1_MAT_s005
                                 + 1.23154 w_k1_MSFT_s005
                                 + 1.22059 w_k1_PG_s005 >= 1
 z_k1_s0053_constraint:          var_1 + z_k1_s0053 + 1.15293 w_k1_ADM_s005
                                 + 1.28227 w_k1_AXP_s005
                                 + 0.897712 w_k1_DIS_s005
                                 + 0.994427 w_k1_FDX_s005
                                 + 1.04289 w_k1_IBM_s005
                                 + 1.04592 w_k1_INTC_s005
                                 + 0.952676 w_k1_KO_s005
                                 + 1.57041 w_k1_MAT_s005
                                 + 0.962728 w_k1_MSFT_s005
                                 + 1.16965 w_k1_PG_s005 >= 1
 z_k1_s0054_constraint:          var_1 + z_k1_s0054 + 1.23281 w_k1_ADM_s005
                                 + 1.73353 w_k1_AXP_s005
                                 + 1.37829 w_k1_DIS_s005
                                 + 1.35854 w_k1_FDX_s005
                                 + 1.14199 w_k1_IBM_s005
                                 + 1.16068 w_k1_INTC_s005
                                 + 1.35346 w_k1_KO_s005 + 1.87688 w_k1_MAT_s005
                                 + 1.39756 w_k1_MSFT_s005
                                 + 1.47376 w_k1_PG_s005 >= 1
 z_k1_s0055_constraint:          var_1 + z_k1_s0055 + 1.25169 w_k1_ADM_s005
                                 + 1.23161 w_k1_AXP_s005
                                 + 1.45923 w_k1_DIS_s005
                                 + 1.16185 w_k1_FDX_s005
                                 + 1.32033 w_k1_IBM_s005
                                 + 1.17236 w_k1_INTC_s005
                                 + 1.27906 w_k1_KO_s005 + 1.44484 w_k1_MAT_s005
                                 + 1.36769 w_k1_MSFT_s005
                                 + 1.34832 w_k1_PG_s005 >= 1
 z_k1_s0056_constraint:          var_1 + z_k1_s0056 + 0.708064 w_k1_ADM_s005
                                 + 0.747624 w_k1_AXP_s005
                                 + 0.952127 w_k1_DIS_s005
                                 + 0.800631 w_k1_FDX_s005
                                 + 1.07669 w_k1_IBM_s005
                                 + 0.93146 w_k1_INTC_s005
                                 + 0.98489 w_k1_KO_s005
                                 + 0.918938 w_k1_MAT_s005
                                 + 0.923489 w_k1_MSFT_s005
                                 + 1.0297 w_k1_PG_s005 >= 1
 z_k1_s0057_constraint:          var_1 + z_k1_s0057 + 1.0916 w_k1_ADM_s005
                                 + 1.1053 w_k1_AXP_s005 + 1.18903 w_k1_DIS_s005
                                 + 0.940147 w_k1_FDX_s005
                                 + 1.55669 w_k1_IBM_s005
                                 + 1.12008 w_k1_INTC_s005
                                 + 1.17319 w_k1_KO_s005 + 1.22808 w_k1_MAT_s005
                                 + 1.01835 w_k1_MSFT_s005
                                 + 1.05177 w_k1_PG_s005 >= 1
 z_k1_s0060_constraint:          var_1 + z_k1_s0060 + 1.21027 w_k1_ADM_s006
                                 + 0.793091 w_k1_AXP_s006
                                 + 0.756686 w_k1_DIS_s006
                                 + 0.827768 w_k1_FDX_s006
                                 + 1.34427 w_k1_IBM_s006
                                 + 0.596075 w_k1_INTC_s006
                                 + 1.03388 w_k1_KO_s006 + 1.28424 w_k1_MAT_s006
                                 + 0.750281 w_k1_MSFT_s006
                                 + 0.985005 w_k1_PG_s006 >= 1
 z_k1_s0061_constraint:          var_1 + z_k1_s0061 + 1.09258 w_k1_ADM_s006
                                 + 1.32518 w_k1_AXP_s006
                                 + 1.04347 w_k1_DIS_s006
                                 + 1.10782 w_k1_FDX_s006
                                 + 0.976074 w_k1_IBM_s006
                                 + 0.881786 w_k1_INTC_s006
                                 + 1.07188 w_k1_KO_s006 + 1.27399 w_k1_MAT_s006
                                 + 1.06798 w_k1_MSFT_s006
                                 + 1.10517 w_k1_PG_s006 >= 1
 z_k1_s0062_constraint:          var_1 + z_k1_s0062 + 1.28822 w_k1_ADM_s006
                                 + 1.12454 w_k1_AXP_s006
                                 + 1.81712 w_k1_DIS_s006
                                 + 1.31905 w_k1_FDX_s006
                                 + 0.944675 w_k1_IBM_s006
                                 + 1.1005 w_k1_INTC_s006 + 1.21311 w_k1_KO_s006
                                 + 1.15257 w_k1_MAT_s006
                                 + 1.37959 w_k1_MSFT_s006
                                 + 1.34859 w_k1_PG_s006 >= 1
 z_k1_s0063_constraint:          var_1 + z_k1_s0063 + 1.1679 w_k1_ADM_s006
                                 + 0.998222 w_k1_AXP_s006
                                 + 0.922888 w_k1_DIS_s006
                                 + 1.08017 w_k1_FDX_s006
                                 + 0.751596 w_k1_IBM_s006
                                 + 0.79555 w_k1_INTC_s006
                                 + 0.827308 w_k1_KO_s006 + 0.8713 w_k1_MAT_s006
                                 + 0.989715 w_k1_MSFT_s006
                                 + 1.02004 w_k1_PG_s006 >= 1
 z_k1_s0064_constraint:          var_1 + z_k1_s0064 + 1.19329 w_k1_ADM_s006
                                 + 1.15504 w_k1_AXP_s006
                                 + 1.57781 w_k1_DIS_s006
                                 + 1.26692 w_k1_FDX_s006
                                 + 1.51792 w_k1_IBM_s006
                                 + 0.875382 w_k1_INTC_s006
                                 + 1.46935 w_k1_KO_s006 + 1.70102 w_k1_MAT_s006
                                 + 1.24271 w_k1_MSFT_s006 + 1.1371 w_k1_PG_s006
                                 >= 1
 z_k1_s0065_constraint:          var_1 + z_k1_s0065 + 1.18552 w_k1_ADM_s006
                                 + 0.550573 w_k1_AXP_s006
                                 + 1.11091 w_k1_DIS_s006
                                 + 0.927579 w_k1_FDX_s006
                                 + 0.937492 w_k1_IBM_s006
                                 + 0.761971 w_k1_INTC_s006
                                 + 1.00162 w_k1_KO_s006
                                 + 0.446037 w_k1_MAT_s006
                                 + 1.06366 w_k1_MSFT_s006
                                 + 0.960067 w_k1_PG_s006 >= 1
 z_k1_s0066_constraint:          var_1 + z_k1_s0066 + 0.931306 w_k1_ADM_s006
                                 + 0.711989 w_k1_AXP_s006
                                 + 0.689819 w_k1_DIS_s006
                                 + 0.781873 w_k1_FDX_s006
                                 + 0.851118 w_k1_IBM_s006
                                 + 0.70474 w_k1_INTC_s006
                                 + 0.818699 w_k1_KO_s006
                                 + 0.632533 w_k1_MAT_s006
                                 + 0.741805 w_k1_MSFT_s006
                                 + 0.823073 w_k1_PG_s006 >= 1
 z_k1_s0067_constraint:          var_1 + z_k1_s0067 + 1.18824 w_k1_ADM_s006
                                 + 1.16275 w_k1_AXP_s006
                                 + 1.20155 w_k1_DIS_s006
                                 + 1.26482 w_k1_FDX_s006
                                 + 0.735376 w_k1_IBM_s006
                                 + 0.973474 w_k1_INTC_s006
                                 + 0.998285 w_k1_KO_s006
                                 + 0.953682 w_k1_MAT_s006
                                 + 1.23981 w_k1_MSFT_s006
                                 + 1.01721 w_k1_PG_s006 >= 1
 z_k1_s0070_constraint:          var_1 + z_k1_s0070 + 0.925345 w_k1_ADM_s007
                                 + 0.82779 w_k1_AXP_s007
                                 + 0.767149 w_k1_DIS_s007
                                 + 0.882929 w_k1_FDX_s007
                                 + 0.688732 w_k1_IBM_s007
                                 + 0.81865 w_k1_INTC_s007
                                 + 0.775048 w_k1_KO_s007
                                 + 0.807733 w_k1_MAT_s007
                                 + 0.877574 w_k1_MSFT_s007
                                 + 0.768633 w_k1_PG_s007 >= 1
 z_k1_s0071_constraint:          var_1 + z_k1_s0071 + 0.936083 w_k1_ADM_s007
                                 + 1.31824 w_k1_AXP_s007
                                 + 1.55057 w_k1_DIS_s007
                                 + 1.14107 w_k1_FDX_s007
                                 + 1.64797 w_k1_IBM_s007
                                 + 1.00816 w_k1_INTC_s007
                                 + 1.66181 w_k1_KO_s007 + 2.44549 w_k1_MAT_s007
                                 + 1.2421 w_k1_MSFT_s007 + 1.14241 w_k1_PG_s007
                                 >= 1
 z_k1_s0072_constraint:          var_1 + z_k1_s0072 + 1.4567 w_k1_ADM_s007
                                 + 1.6564 w_k1_AXP_s007 + 1.57314 w_k1_DIS_s007
                                 + 1.43548 w_k1_FDX_s007
                                 + 1.12824 w_k1_IBM_s007
                                 + 1.05634 w_k1_INTC_s007
                                 + 1.30429 w_k1_KO_s007 + 1.89759 w_k1_MAT_s007
                                 + 1.35686 w_k1_MSFT_s007
                                 + 1.16599 w_k1_PG_s007 >= 1
 z_k1_s0073_constraint:          var_1 + z_k1_s0073 + 0.957577 w_k1_ADM_s007
                                 + 0.818107 w_k1_AXP_s007
                                 + 0.731104 w_k1_DIS_s007
                                 + 0.832872 w_k1_FDX_s007
                                 + 1.02434 w_k1_IBM_s007
                                 + 0.743496 w_k1_INTC_s007
                                 + 0.830244 w_k1_KO_s007
                                 + 0.887403 w_k1_MAT_s007
                                 + 0.849558 w_k1_MSFT_s007
                                 + 0.861709 w_k1_PG_s007 >= 1
 z_k1_s0074_constraint:          var_1 + z_k1_s0074 + 0.987059 w_k1_ADM_s007
                                 + 0.883411 w_k1_AXP_s007
                                 + 1.07903 w_k1_DIS_s007
                                 + 1.00587 w_k1_FDX_s007
                                 + 1.03225 w_k1_IBM_s007
                                 + 1.01365 w_k1_INTC_s007
                                 + 1.21678 w_k1_KO_s007 + 1.02197 w_k1_MAT_s007
                                 + 1.13352 w_k1_MSFT_s007
                                 + 0.951577 w_k1_PG_s007 >= 1
 z_k1_s0075_constraint:          var_1 + z_k1_s0075 + 0.948862 w_k1_ADM_s007
                                 + 0.961969 w_k1_AXP_s007
                                 + 0.95763 w_k1_DIS_s007
                                 + 0.841248 w_k1_FDX_s007
                                 + 1.42932 w_k1_IBM_s007
                                 + 0.861076 w_k1_INTC_s007
                                 + 1.18118 w_k1_KO_s007 + 1.40356 w_k1_MAT_s007
                                 + 0.903821 w_k1_MSFT_s007
                                 + 0.87778 w_k1_PG_s007 >= 1
 z_k1_s0076_constraint:          var_1 + z_k1_s0076 + 1.11189 w_k1_ADM_s007
                                 + 1.28382 w_k1_AXP_s007
                                 + 1.02665 w_k1_DIS_s007
                                 + 1.15274 w_k1_FDX_s007
                                 + 0.86132 w_k1_IBM_s007
                                 + 0.848163 w_k1_INTC_s007
                                 + 0.8815 w_k1_KO_s007 + 1.28355 w_k1_MAT_s007
                                 + 0.907397 w_k1_MSFT_s007
                                 + 0.988403 w_k1_PG_s007 >= 1
 z_k1_s0077_constraint:          var_1 + z_k1_s0077 + 0.695677 w_k1_ADM_s007
                                 + 0.622316 w_k1_AXP_s007
                                 + 0.628521 w_k1_DIS_s007
                                 + 0.675827 w_k1_FDX_s007
                                 + 0.788291 w_k1_IBM_s007
                                 + 0.592997 w_k1_INTC_s007
                                 + 0.702567 w_k1_KO_s007
                                 + 0.764669 w_k1_MAT_s007
                                 + 0.701041 w_k1_MSFT_s007
                                 + 0.713673 w_k1_PG_s007 >= 1
 z_k1_s0100_constraint:          var_1 + z_k1_s0100 + 1.22522 w_k1_ADM_s010
                                 + 1.11602 w_k1_AXP_s010
                                 + 1.62179 w_k1_DIS_s010
                                 + 1.28582 w_k1_FDX_s010
                                 + 1.17365 w_k1_IBM_s010
                                 + 1.16134 w_k1_INTC_s010
                                 + 1.28149 w_k1_KO_s010 + 1.33504 w_k1_MAT_s010
                                 + 1.46557 w_k1_MSFT_s010
                                 + 1.26779 w_k1_PG_s010 >= 1
 z_k1_s0101_constraint:          var_1 + z_k1_s0101 + 1.02436 w_k1_ADM_s010
                                 + 0.698699 w_k1_AXP_s010
                                 + 0.874352 w_k1_DIS_s010
                                 + 0.89832 w_k1_FDX_s010
                                 + 0.530394 w_k1_IBM_s010
                                 + 0.795533 w_k1_INTC_s010
                                 + 0.67206 w_k1_KO_s010
                                 + 0.558098 w_k1_MAT_s010
                                 + 0.960297 w_k1_MSFT_s010
                                 + 0.750429 w_k1_PG_s010 >= 1
 z_k1_s0102_constraint:          var_1 + z_k1_s0102 + 1.27703 w_k1_ADM_s010
                                 + 1.09046 w_k1_AXP_s010 + 1.1889 w_k1_DIS_s010
                                 + 0.98735 w_k1_FDX_s010
                                 + 1.39525 w_k1_IBM_s010
                                 + 0.865845 w_k1_INTC_s010
                                 + 1.06492 w_k1_KO_s010 + 1.13704 w_k1_MAT_s010
                                 + 0.953304 w_k1_MSFT_s010
                                 + 0.992307 w_k1_PG_s010 >= 1
 z_k1_s0103_constraint:          var_1 + z_k1_s0103 + 0.939079 w_k1_ADM_s010
                                 + 0.627421 w_k1_AXP_s010
                                 + 0.728548 w_k1_DIS_s010
                                 + 0.765165 w_k1_FDX_s010
                                 + 0.84952 w_k1_IBM_s010
                                 + 0.691198 w_k1_INTC_s010
                                 + 0.874604 w_k1_KO_s010
                                 + 0.911797 w_k1_MAT_s010
                                 + 0.926755 w_k1_MSFT_s010
                                 + 0.875327 w_k1_PG_s010 >= 1
 z_k1_s0104_constraint:          var_1 + z_k1_s0104 + 0.851709 w_k1_ADM_s010
                                 + 0.67444 w_k1_AXP_s010
                                 + 1.12189 w_k1_DIS_s010
                                 + 0.962043 w_k1_FDX_s010
                                 + 0.813619 w_k1_IBM_s010
                                 + 0.933057 w_k1_INTC_s010
                                 + 0.840099 w_k1_KO_s010
                                 + 0.657745 w_k1_MAT_s010
                                 + 1.02629 w_k1_MSFT_s010
                                 + 0.863098 w_k1_PG_s010 >= 1
 z_k1_s0105_constraint:          var_1 + z_k1_s0105 + 1.22449 w_k1_ADM_s010
                                 + 1.6164 w_k1_AXP_s010 + 1.90069 w_k1_DIS_s010
                                 + 1.28814 w_k1_FDX_s010
                                 + 1.62754 w_k1_IBM_s010
                                 + 0.961418 w_k1_INTC_s010
                                 + 1.37134 w_k1_KO_s010 + 2.02389 w_k1_MAT_s010
                                 + 1.35344 w_k1_MSFT_s010
                                 + 1.32599 w_k1_PG_s010 >= 1
 z_k1_s0106_constraint:          var_1 + z_k1_s0106 + 1.20793 w_k1_ADM_s010
                                 + 1.20673 w_k1_AXP_s010
                                 + 1.04301 w_k1_DIS_s010
                                 + 1.07866 w_k1_FDX_s010
                                 + 0.928857 w_k1_IBM_s010
                                 + 1.15054 w_k1_INTC_s010
                                 + 0.975408 w_k1_KO_s010 + 1.016 w_k1_MAT_s010
                                 + 1.01426 w_k1_MSFT_s010
                                 + 1.00771 w_k1_PG_s010 >= 1
 z_k1_s0107_constraint:          var_1 + z_k1_s0107 + 0.922014 w_k1_ADM_s010
                                 + 0.70277 w_k1_AXP_s010
                                 + 0.968766 w_k1_DIS_s010
                                 + 0.865915 w_k1_FDX_s010
                                 + 0.76105 w_k1_IBM_s010
                                 + 0.907632 w_k1_INTC_s010
                                 + 0.854115 w_k1_KO_s010
                                 + 0.715442 w_k1_MAT_s010
                                 + 1.00304 w_k1_MSFT_s010
                                 + 0.941064 w_k1_PG_s010 >= 1
 z_k1_s0110_constraint:          var_1 + z_k1_s0110 + 1.19591 w_k1_ADM_s011
                                 + 1.19526 w_k1_AXP_s011
                                 + 1.25062 w_k1_DIS_s011
                                 + 1.14256 w_k1_FDX_s011
                                 + 0.923345 w_k1_IBM_s011
                                 + 1.23205 w_k1_INTC_s011
                                 + 0.980188 w_k1_KO_s011
                                 + 0.955875 w_k1_MAT_s011
                                 + 1.00335 w_k1_MSFT_s011
                                 + 1.07095 w_k1_PG_s011 >= 1
 z_k1_s0111_constraint:          var_1 + z_k1_s0111 + 0.828225 w_k1_ADM_s011
                                 + 0.864708 w_k1_AXP_s011
                                 + 0.87944 w_k1_DIS_s011
                                 + 0.89905 w_k1_FDX_s011
                                 + 0.918309 w_k1_IBM_s011
                                 + 1.04503 w_k1_INTC_s011
                                 + 0.939162 w_k1_KO_s011
                                 + 0.818394 w_k1_MAT_s011
                                 + 0.853771 w_k1_MSFT_s011
                                 + 0.932468 w_k1_PG_s011 >= 1
 z_k1_s0112_constraint:          var_1 + z_k1_s0112 + 1.17426 w_k1_ADM_s011
                                 + 1.50127 w_k1_AXP_s011
                                 + 1.89584 w_k1_DIS_s011
                                 + 1.61083 w_k1_FDX_s011
                                 + 1.08947 w_k1_IBM_s011 + 1.337 w_k1_INTC_s011
                                 + 1.26166 w_k1_KO_s011 + 1.07187 w_k1_MAT_s011
                                 + 1.42713 w_k1_MSFT_s011
                                 + 1.35491 w_k1_PG_s011 >= 1
 z_k1_s0113_constraint:          var_1 + z_k1_s0113 + 0.747302 w_k1_ADM_s011
                                 + 0.770998 w_k1_AXP_s011
                                 + 0.77131 w_k1_DIS_s011
                                 + 0.712438 w_k1_FDX_s011
                                 + 1.15023 w_k1_IBM_s011
                                 + 0.871279 w_k1_INTC_s011
                                 + 0.864764 w_k1_KO_s011
                                 + 0.967633 w_k1_MAT_s011
                                 + 0.801407 w_k1_MSFT_s011
                                 + 0.803739 w_k1_PG_s011 >= 1
 z_k1_s0114_constraint:          var_1 + z_k1_s0114 + 1.21842 w_k1_ADM_s011
                                 + 1.41377 w_k1_AXP_s011
                                 + 1.20718 w_k1_DIS_s011
                                 + 1.13813 w_k1_FDX_s011
                                 + 1.41819 w_k1_IBM_s011
                                 + 1.10526 w_k1_INTC_s011
                                 + 1.25858 w_k1_KO_s011 + 1.66191 w_k1_MAT_s011
                                 + 1.08911 w_k1_MSFT_s011
                                 + 1.24591 w_k1_PG_s011 >= 1
 z_k1_s0115_constraint:          var_1 + z_k1_s0115 + 0.899925 w_k1_ADM_s011
                                 + 0.862317 w_k1_AXP_s011
                                 + 0.83481 w_k1_DIS_s011
                                 + 0.971358 w_k1_FDX_s011
                                 + 0.680556 w_k1_IBM_s011
                                 + 0.86473 w_k1_INTC_s011
                                 + 0.744485 w_k1_KO_s011
                                 + 0.595019 w_k1_MAT_s011
                                 + 0.832976 w_k1_MSFT_s011
                                 + 0.865598 w_k1_PG_s011 >= 1
 z_k1_s0116_constraint:          var_1 + z_k1_s0116 + 0.880969 w_k1_ADM_s011
                                 + 1.2402 w_k1_AXP_s011 + 1.18349 w_k1_DIS_s011
                                 + 0.945341 w_k1_FDX_s011
                                 + 1.73682 w_k1_IBM_s011
                                 + 0.969248 w_k1_INTC_s011
                                 + 1.24431 w_k1_KO_s011 + 1.66378 w_k1_MAT_s011
                                 + 0.945786 w_k1_MSFT_s011
                                 + 0.937741 w_k1_PG_s011 >= 1
 z_k1_s0117_constraint:          var_1 + z_k1_s0117 + 0.959166 w_k1_ADM_s011
                                 + 0.796456 w_k1_AXP_s011
                                 + 1.31972 w_k1_DIS_s011
                                 + 0.996137 w_k1_FDX_s011
                                 + 1.06903 w_k1_IBM_s011
                                 + 0.881622 w_k1_INTC_s011
                                 + 1.0478 w_k1_KO_s011 + 0.951294 w_k1_MAT_s011
                                 + 1.07378 w_k1_MSFT_s011
                                 + 1.10276 w_k1_PG_s011 >= 1
 z_k1_s0120_constraint:          var_1 + z_k1_s0120 + 1.15207 w_k1_ADM_s012
                                 + 1.35817 w_k1_AXP_s012
                                 + 1.90719 w_k1_DIS_s012
                                 + 1.43005 w_k1_FDX_s012
                                 + 1.31773 w_k1_IBM_s012
                                 + 1.39716 w_k1_INTC_s012 + 1.7691 w_k1_KO_s012
                                 + 1.51655 w_k1_MAT_s012
                                 + 1.39975 w_k1_MSFT_s012
                                 + 1.33322 w_k1_PG_s012 >= 1
 z_k1_s0121_constraint:          var_1 + z_k1_s0121 + 0.886379 w_k1_ADM_s012
                                 + 0.836229 w_k1_AXP_s012
                                 + 0.884533 w_k1_DIS_s012
                                 + 0.920935 w_k1_FDX_s012
                                 + 0.72902 w_k1_IBM_s012
                                 + 0.939824 w_k1_INTC_s012
                                 + 0.733342 w_k1_KO_s012
                                 + 0.654771 w_k1_MAT_s012
                                 + 0.913092 w_k1_MSFT_s012
                                 + 0.948607 w_k1_PG_s012 >= 1
 z_k1_s0122_constraint:          var_1 + z_k1_s0122 + 0.976693 w_k1_ADM_s012
                                 + 0.959615 w_k1_AXP_s012
                                 + 1.22205 w_k1_DIS_s012
                                 + 0.992712 w_k1_FDX_s012
                                 + 1.60742 w_k1_IBM_s012
                                 + 1.03317 w_k1_INTC_s012
                                 + 1.24989 w_k1_KO_s012 + 1.37466 w_k1_MAT_s012
                                 + 1.03915 w_k1_MSFT_s012
                                 + 1.16685 w_k1_PG_s012 >= 1
 z_k1_s0123_constraint:          var_1 + z_k1_s0123 + 1.06409 w_k1_ADM_s012
                                 + 1.2272 w_k1_AXP_s012 + 1.11865 w_k1_DIS_s012
                                 + 1.22312 w_k1_FDX_s012
                                 + 1.26924 w_k1_IBM_s012
                                 + 1.25958 w_k1_INTC_s012
                                 + 1.18289 w_k1_KO_s012 + 1.12254 w_k1_MAT_s012
                                 + 1.07228 w_k1_MSFT_s012
                                 + 1.15876 w_k1_PG_s012 >= 1
 z_k1_s0124_constraint:          var_1 + z_k1_s0124 + 1.15197 w_k1_ADM_s012
                                 + 0.961255 w_k1_AXP_s012
                                 + 1.2269 w_k1_DIS_s012 + 1.28195 w_k1_FDX_s012
                                 + 0.763251 w_k1_IBM_s012
                                 + 1.18039 w_k1_INTC_s012
                                 + 1.05818 w_k1_KO_s012
                                 + 0.875872 w_k1_MAT_s012
                                 + 1.07481 w_k1_MSFT_s012
                                 + 1.07738 w_k1_PG_s012 >= 1
 z_k1_s0125_constraint:          var_1 + z_k1_s0125 + 0.995285 w_k1_ADM_s012
                                 + 0.848356 w_k1_AXP_s012
                                 + 0.806731 w_k1_DIS_s012
                                 + 0.879039 w_k1_FDX_s012
                                 + 1.15602 w_k1_IBM_s012
                                 + 0.84285 w_k1_INTC_s012
                                 + 1.10405 w_k1_KO_s012 + 1.50712 w_k1_MAT_s012
                                 + 0.8906 w_k1_MSFT_s012
                                 + 0.998498 w_k1_PG_s012 >= 1
 z_k1_s0126_constraint:          var_1 + z_k1_s0126 + 0.675455 w_k1_ADM_s012
                                 + 0.678028 w_k1_AXP_s012
                                 + 0.588932 w_k1_DIS_s012
                                 + 0.769988 w_k1_FDX_s012
                                 + 0.707491 w_k1_IBM_s012
                                 + 0.959989 w_k1_INTC_s012
                                 + 0.738031 w_k1_KO_s012
                                 + 0.605261 w_k1_MAT_s012
                                 + 0.735179 w_k1_MSFT_s012
                                 + 0.770812 w_k1_PG_s012 >= 1
 z_k1_s0127_constraint:          var_1 + z_k1_s0127 + 0.926734 w_k1_ADM_s012
                                 + 1.16861 w_k1_AXP_s012
                                 + 1.03485 w_k1_DIS_s012
                                 + 1.01766 w_k1_FDX_s012
                                 + 1.17584 w_k1_IBM_s012
                                 + 1.02744 w_k1_INTC_s012
                                 + 0.904409 w_k1_KO_s012
                                 + 1.50629 w_k1_MAT_s012
                                 + 1.05712 w_k1_MSFT_s012
                                 + 1.04166 w_k1_PG_s012 >= 1
 z_k1_s0130_constraint:          var_1 + z_k1_s0130 + 1.26315 w_k1_ADM_s013
                                 + 0.908153 w_k1_AXP_s013
                                 + 0.949886 w_k1_DIS_s013
                                 + 0.956254 w_k1_FDX_s013
                                 + 0.835747 w_k1_IBM_s013
                                 + 0.880653 w_k1_INTC_s013
                                 + 0.829734 w_k1_KO_s013
                                 + 0.703383 w_k1_MAT_s013
                                 + 0.938426 w_k1_MSFT_s013
                                 + 0.954097 w_k1_PG_s013 >= 1
 z_k1_s0131_constraint:          var_1 + z_k1_s0131 + 1.14645 w_k1_ADM_s013
                                 + 1.18409 w_k1_AXP_s013 + 1.4121 w_k1_DIS_s013
                                 + 1.1304 w_k1_FDX_s013
                                 + 0.859779 w_k1_IBM_s013
                                 + 1.09564 w_k1_INTC_s013 + 1.0784 w_k1_KO_s013
                                 + 0.76899 w_k1_MAT_s013
                                 + 1.1006 w_k1_MSFT_s013 + 1.10835 w_k1_PG_s013
                                 >= 1
 z_k1_s0132_constraint:          var_1 + z_k1_s0132 + 1.57844 w_k1_ADM_s013
                                 + 1.52152 w_k1_AXP_s013
                                 + 1.72177 w_k1_DIS_s013
                                 + 1.28677 w_k1_FDX_s013 + 1.0738 w_k1_IBM_s013
                                 + 1.12798 w_k1_INTC_s013
                                 + 1.18393 w_k1_KO_s013 + 1.25821 w_k1_MAT_s013
                                 + 1.19094 w_k1_MSFT_s013
                                 + 1.30723 w_k1_PG_s013 >= 1
 z_k1_s0133_constraint:          var_1 + z_k1_s0133 + 1.14362 w_k1_ADM_s013
                                 + 0.705091 w_k1_AXP_s013
                                 + 0.700091 w_k1_DIS_s013
                                 + 0.734982 w_k1_FDX_s013
                                 + 0.762141 w_k1_IBM_s013
                                 + 0.844984 w_k1_INTC_s013
                                 + 0.840286 w_k1_KO_s013
                                 + 0.697664 w_k1_MAT_s013
                                 + 0.815035 w_k1_MSFT_s013
                                 + 0.952832 w_k1_PG_s013 >= 1
 z_k1_s0134_constraint:          var_1 + z_k1_s0134 + 1.056 w_k1_ADM_s013
                                 + 0.775315 w_k1_AXP_s013
                                 + 0.82406 w_k1_DIS_s013
                                 + 0.755511 w_k1_FDX_s013
                                 + 1.31869 w_k1_IBM_s013
                                 + 0.586975 w_k1_INTC_s013
                                 + 1.00194 w_k1_KO_s013 + 1.42116 w_k1_MAT_s013
                                 + 0.792386 w_k1_MSFT_s013
                                 + 0.853734 w_k1_PG_s013 >= 1
 z_k1_s0135_constraint:          var_1 + z_k1_s0135 + 1.18106 w_k1_ADM_s013
                                 + 1.11051 w_k1_AXP_s013
                                 + 1.32733 w_k1_DIS_s013
                                 + 1.06607 w_k1_FDX_s013
                                 + 1.10574 w_k1_IBM_s013
                                 + 0.999882 w_k1_INTC_s013
                                 + 1.10815 w_k1_KO_s013 + 1.08741 w_k1_MAT_s013
                                 + 1.12184 w_k1_MSFT_s013 + 1.0081 w_k1_PG_s013
                                 >= 1
 z_k1_s0136_constraint:          var_1 + z_k1_s0136 + 1.11614 w_k1_ADM_s013
                                 + 0.948241 w_k1_AXP_s013
                                 + 1.03385 w_k1_DIS_s013
                                 + 0.898827 w_k1_FDX_s013
                                 + 1.0618 w_k1_IBM_s013
                                 + 0.856969 w_k1_INTC_s013
                                 + 1.06072 w_k1_KO_s013 + 1.23779 w_k1_MAT_s013
                                 + 0.831964 w_k1_MSFT_s013
                                 + 1.09572 w_k1_PG_s013 >= 1
 z_k1_s0137_constraint:          var_1 + z_k1_s0137 + 1.29957 w_k1_ADM_s013
                                 + 1.41279 w_k1_AXP_s013
                                 + 1.39458 w_k1_DIS_s013
                                 + 1.11475 w_k1_FDX_s013 + 1.6242 w_k1_IBM_s013
                                 + 1.28577 w_k1_INTC_s013
                                 + 1.33783 w_k1_KO_s013 + 1.5755 w_k1_MAT_s013
                                 + 1.15403 w_k1_MSFT_s013
                                 + 1.17904 w_k1_PG_s013 >= 1
 z_k1_s0140_constraint:          var_1 + z_k1_s0140 + 0.669371 w_k1_ADM_s014
                                 + 0.680384 w_k1_AXP_s014
                                 + 0.592038 w_k1_DIS_s014
                                 + 0.738637 w_k1_FDX_s014
                                 + 0.732081 w_k1_IBM_s014
                                 + 0.877186 w_k1_INTC_s014
                                 + 0.720302 w_k1_KO_s014
                                 + 0.69634 w_k1_MAT_s014
                                 + 0.734658 w_k1_MSFT_s014
                                 + 0.804971 w_k1_PG_s014 >= 1
 z_k1_s0141_constraint:          var_1 + z_k1_s0141 + 1.36458 w_k1_ADM_s014
                                 + 1.35838 w_k1_AXP_s014
                                 + 1.87962 w_k1_DIS_s014
                                 + 1.53873 w_k1_FDX_s014
                                 + 1.15654 w_k1_IBM_s014
                                 + 1.24279 w_k1_INTC_s014
                                 + 1.26294 w_k1_KO_s014 + 1.40903 w_k1_MAT_s014
                                 + 1.22961 w_k1_MSFT_s014
                                 + 1.19796 w_k1_PG_s014 >= 1
 z_k1_s0142_constraint:          var_1 + z_k1_s0142 + 1.14379 w_k1_ADM_s014
                                 + 1.02143 w_k1_AXP_s014
                                 + 1.03474 w_k1_DIS_s014
                                 + 0.968306 w_k1_FDX_s014
                                 + 1.30744 w_k1_IBM_s014
                                 + 0.948879 w_k1_INTC_s014
                                 + 1.31944 w_k1_KO_s014
                                 + 0.815968 w_k1_MAT_s014
                                 + 0.975718 w_k1_MSFT_s014
                                 + 1.06933 w_k1_PG_s014 >= 1
 z_k1_s0143_constraint:          var_1 + z_k1_s0143 + 0.903584 w_k1_ADM_s014
                                 + 0.712974 w_k1_AXP_s014
                                 + 0.842593 w_k1_DIS_s014
                                 + 0.865386 w_k1_FDX_s014
                                 + 0.799055 w_k1_IBM_s014
                                 + 0.98272 w_k1_INTC_s014
                                 + 0.846842 w_k1_KO_s014
                                 + 0.635364 w_k1_MAT_s014
                                 + 0.870966 w_k1_MSFT_s014
                                 + 0.925667 w_k1_PG_s014 >= 1
 z_k1_s0144_constraint:          var_1 + z_k1_s0144 + 0.876317 w_k1_ADM_s014
                                 + 0.880856 w_k1_AXP_s014
                                 + 0.826777 w_k1_DIS_s014
                                 + 0.88174 w_k1_FDX_s014
                                 + 0.999352 w_k1_IBM_s014
                                 + 0.992251 w_k1_INTC_s014
                                 + 0.9112 w_k1_KO_s014 + 0.730497 w_k1_MAT_s014
                                 + 0.845527 w_k1_MSFT_s014
                                 + 0.976011 w_k1_PG_s014 >= 1
 z_k1_s0145_constraint:          var_1 + z_k1_s0145 + 1.09029 w_k1_ADM_s014
                                 + 0.91044 w_k1_AXP_s014
                                 + 1.11076 w_k1_DIS_s014
                                 + 1.08695 w_k1_FDX_s014
                                 + 0.73291 w_k1_IBM_s014
                                 + 0.896431 w_k1_INTC_s014
                                 + 0.940731 w_k1_KO_s014
                                 + 0.531625 w_k1_MAT_s014
                                 + 1.05706 w_k1_MSFT_s014
                                 + 0.94483 w_k1_PG_s014 >= 1
 z_k1_s0146_constraint:          var_1 + z_k1_s0146 + 1.25245 w_k1_ADM_s014
                                 + 1.26485 w_k1_AXP_s014
                                 + 1.45437 w_k1_DIS_s014
                                 + 1.15539 w_k1_FDX_s014
                                 + 1.34165 w_k1_IBM_s014
                                 + 1.14429 w_k1_INTC_s014
                                 + 1.31884 w_k1_KO_s014 + 1.06835 w_k1_MAT_s014
                                 + 1.15553 w_k1_MSFT_s014
                                 + 1.20516 w_k1_PG_s014 >= 1
 z_k1_s0147_constraint:          var_1 + z_k1_s0147 + 1.01672 w_k1_ADM_s014
                                 + 1.2826 w_k1_AXP_s014 + 1.27769 w_k1_DIS_s014
                                 + 1.24607 w_k1_FDX_s014
                                 + 0.948114 w_k1_IBM_s014
                                 + 1.36306 w_k1_INTC_s014 + 1.0376 w_k1_KO_s014
                                 + 0.936306 w_k1_MAT_s014
                                 + 1.1379 w_k1_MSFT_s014 + 1.04682 w_k1_PG_s014
                                 >= 1
 z_k1_s0150_constraint:          var_1 + z_k1_s0150 + 1.3233 w_k1_ADM_s015
                                 + 1.30401 w_k1_AXP_s015
                                 + 1.74324 w_k1_DIS_s015
                                 + 1.29486 w_k1_FDX_s015 + 1.3397 w_k1_IBM_s015
                                 + 1.20025 w_k1_INTC_s015
                                 + 1.36532 w_k1_KO_s015 + 1.09236 w_k1_MAT_s015
                                 + 1.22329 w_k1_MSFT_s015
                                 + 1.42533 w_k1_PG_s015 >= 1
 z_k1_s0151_constraint:          var_1 + z_k1_s0151 + 1.07145 w_k1_ADM_s015
                                 + 1.03187 w_k1_AXP_s015
                                 + 1.03296 w_k1_DIS_s015
                                 + 0.936606 w_k1_FDX_s015
                                 + 1.59665 w_k1_IBM_s015
                                 + 0.729801 w_k1_INTC_s015
                                 + 1.09303 w_k1_KO_s015 + 1.44456 w_k1_MAT_s015
                                 + 0.746566 w_k1_MSFT_s015
                                 + 1.07729 w_k1_PG_s015 >= 1
 z_k1_s0152_constraint:          var_1 + z_k1_s0152 + 0.96126 w_k1_ADM_s015
                                 + 1.09327 w_k1_AXP_s015
                                 + 1.00992 w_k1_DIS_s015
                                 + 1.00388 w_k1_FDX_s015
                                 + 1.18608 w_k1_IBM_s015
                                 + 0.930316 w_k1_INTC_s015
                                 + 1.19097 w_k1_KO_s015 + 1.10786 w_k1_MAT_s015
                                 + 1.0936 w_k1_MSFT_s015 + 1.12628 w_k1_PG_s015
                                 >= 1
 z_k1_s0153_constraint:          var_1 + z_k1_s0153 + 0.933789 w_k1_ADM_s015
                                 + 0.721808 w_k1_AXP_s015
                                 + 0.888849 w_k1_DIS_s015
                                 + 0.924558 w_k1_FDX_s015
                                 + 0.746344 w_k1_IBM_s015
                                 + 0.832654 w_k1_INTC_s015
                                 + 0.865609 w_k1_KO_s015
                                 + 0.632696 w_k1_MAT_s015
                                 + 0.883842 w_k1_MSFT_s015
                                 + 1.01125 w_k1_PG_s015 >= 1
 z_k1_s0154_constraint:          var_1 + z_k1_s0154 + 0.861139 w_k1_ADM_s015
                                 + 0.964238 w_k1_AXP_s015
                                 + 0.743133 w_k1_DIS_s015
                                 + 0.87462 w_k1_FDX_s015
                                 + 0.925892 w_k1_IBM_s015
                                 + 0.711344 w_k1_INTC_s015
                                 + 0.816691 w_k1_KO_s015
                                 + 1.00832 w_k1_MAT_s015
                                 + 0.646704 w_k1_MSFT_s015
                                 + 0.951696 w_k1_PG_s015 >= 1
 z_k1_s0155_constraint:          var_1 + z_k1_s0155 + 0.906953 w_k1_ADM_s015
                                 + 1.1684 w_k1_AXP_s015 + 1.57553 w_k1_DIS_s015
                                 + 1.42905 w_k1_FDX_s015
                                 + 0.865385 w_k1_IBM_s015
                                 + 1.17735 w_k1_INTC_s015
                                 + 1.20501 w_k1_KO_s015
                                 + 0.827231 w_k1_MAT_s015
                                 + 1.37954 w_k1_MSFT_s015
                                 + 1.26144 w_k1_PG_s015 >= 1
 z_k1_s0156_constraint:          var_1 + z_k1_s0156 + 0.816261 w_k1_ADM_s015
                                 + 0.829954 w_k1_AXP_s015
                                 + 0.729029 w_k1_DIS_s015
                                 + 0.750277 w_k1_FDX_s015
                                 + 1.21843 w_k1_IBM_s015
                                 + 0.766784 w_k1_INTC_s015
                                 + 0.910784 w_k1_KO_s015
                                 + 1.05618 w_k1_MAT_s015
                                 + 0.751161 w_k1_MSFT_s015
                                 + 0.942053 w_k1_PG_s015 >= 1
 z_k1_s0157_constraint:          var_1 + z_k1_s0157 + 1.03944 w_k1_ADM_s015
                                 + 1.1925 w_k1_AXP_s015 + 1.15356 w_k1_DIS_s015
                                 + 1.12437 w_k1_FDX_s015
                                 + 0.952432 w_k1_IBM_s015
                                 + 0.925887 w_k1_INTC_s015
                                 + 1.02851 w_k1_KO_s015 + 1.20457 w_k1_MAT_s015
                                 + 1.05305 w_k1_MSFT_s015
                                 + 1.08368 w_k1_PG_s015 >= 1
 z_k1_s0160_constraint:          var_1 + z_k1_s0160 + 1.04415 w_k1_ADM_s016
                                 + 0.815993 w_k1_AXP_s016
                                 + 0.88862 w_k1_DIS_s016
                                 + 0.919947 w_k1_FDX_s016
                                 + 0.62252 w_k1_IBM_s016
                                 + 0.997406 w_k1_INTC_s016
                                 + 0.821514 w_k1_KO_s016
                                 + 0.579149 w_k1_MAT_s016
                                 + 0.985458 w_k1_MSFT_s016
                                 + 0.879645 w_k1_PG_s016 >= 1
 z_k1_s0161_constraint:          var_1 + z_k1_s0161 + 1.02859 w_k1_ADM_s016
                                 + 0.777744 w_k1_AXP_s016
                                 + 0.690143 w_k1_DIS_s016
                                 + 0.774404 w_k1_FDX_s016
                                 + 0.854326 w_k1_IBM_s016
                                 + 0.923096 w_k1_INTC_s016
                                 + 0.851288 w_k1_KO_s016
                                 + 0.601948 w_k1_MAT_s016
                                 + 0.913152 w_k1_MSFT_s016
                                 + 0.928969 w_k1_PG_s016 >= 1
 z_k1_s0162_constraint:          var_1 + z_k1_s0162 + 1.18568 w_k1_ADM_s016
                                 + 1.20463 w_k1_AXP_s016
                                 + 1.16298 w_k1_DIS_s016
                                 + 1.29023 w_k1_FDX_s016
                                 + 0.800408 w_k1_IBM_s016
                                 + 1.16609 w_k1_INTC_s016
                                 + 0.983655 w_k1_KO_s016
                                 + 0.94232 w_k1_MAT_s016
                                 + 1.29575 w_k1_MSFT_s016
                                 + 0.996933 w_k1_PG_s016 >= 1
 z_k1_s0163_constraint:          var_1 + z_k1_s0163 + 1.03295 w_k1_ADM_s016
                                 + 1.00569 w_k1_AXP_s016
                                 + 1.20749 w_k1_DIS_s016
                                 + 1.05363 w_k1_FDX_s016
                                 + 0.986346 w_k1_IBM_s016
                                 + 1.07273 w_k1_INTC_s016
                                 + 1.11511 w_k1_KO_s016 + 0.92171 w_k1_MAT_s016
                                 + 1.18361 w_k1_MSFT_s016
                                 + 1.02168 w_k1_PG_s016 >= 1
 z_k1_s0164_constraint:          var_1 + z_k1_s0164 + 1.37909 w_k1_ADM_s016
                                 + 1.24066 w_k1_AXP_s016
                                 + 1.83259 w_k1_DIS_s016
                                 + 1.34512 w_k1_FDX_s016
                                 + 1.36517 w_k1_IBM_s016
                                 + 1.22581 w_k1_INTC_s016
                                 + 1.57714 w_k1_KO_s016 + 1.4106 w_k1_MAT_s016
                                 + 1.37356 w_k1_MSFT_s016
                                 + 1.33721 w_k1_PG_s016 >= 1
 z_k1_s0165_constraint:          var_1 + z_k1_s0165 + 1.11772 w_k1_ADM_s016
                                 + 1.05248 w_k1_AXP_s016
                                 + 0.963321 w_k1_DIS_s016
                                 + 0.938112 w_k1_FDX_s016
                                 + 1.23904 w_k1_IBM_s016
                                 + 1.07893 w_k1_INTC_s016
                                 + 1.05618 w_k1_KO_s016 + 1.3717 w_k1_MAT_s016
                                 + 1.06217 w_k1_MSFT_s016
                                 + 1.13467 w_k1_PG_s016 >= 1
 z_k1_s0166_constraint:          var_1 + z_k1_s0166 + 0.851532 w_k1_ADM_s016
                                 + 0.665621 w_k1_AXP_s016
                                 + 0.936491 w_k1_DIS_s016
                                 + 0.745674 w_k1_FDX_s016
                                 + 1.2544 w_k1_IBM_s016
                                 + 0.919302 w_k1_INTC_s016
                                 + 1.16718 w_k1_KO_s016
                                 + 0.818891 w_k1_MAT_s016
                                 + 0.897056 w_k1_MSFT_s016
                                 + 0.919602 w_k1_PG_s016 >= 1
 z_k1_s0167_constraint:          var_1 + z_k1_s0167 + 1.14193 w_k1_ADM_s016
                                 + 1.39935 w_k1_AXP_s016
                                 + 1.46033 w_k1_DIS_s016
                                 + 1.22691 w_k1_FDX_s016
                                 + 1.29816 w_k1_IBM_s016
                                 + 1.37651 w_k1_INTC_s016
                                 + 1.43418 w_k1_KO_s016 + 1.27579 w_k1_MAT_s016
                                 + 1.25347 w_k1_MSFT_s016
                                 + 1.16195 w_k1_PG_s016 >= 1
 z_k1_s0170_constraint:          var_1 + z_k1_s0170 + 1.17328 w_k1_ADM_s017
                                 + 1.04813 w_k1_AXP_s017
                                 + 1.18083 w_k1_DIS_s017
                                 + 1.21492 w_k1_FDX_s017
                                 + 0.684146 w_k1_IBM_s017
                                 + 1.03398 w_k1_INTC_s017
                                 + 0.93127 w_k1_KO_s017
                                 + 0.705654 w_k1_MAT_s017
                                 + 1.10324 w_k1_MSFT_s017
                                 + 1.04073 w_k1_PG_s017 >= 1
 z_k1_s0171_constraint:          var_1 + z_k1_s0171 + 1.08263 w_k1_ADM_s017
                                 + 1.34179 w_k1_AXP_s017
                                 + 1.02007 w_k1_DIS_s017
                                 + 1.00138 w_k1_FDX_s017
                                 + 1.26527 w_k1_IBM_s017
                                 + 0.863562 w_k1_INTC_s017
                                 + 1.0631 w_k1_KO_s017 + 1.64787 w_k1_MAT_s017
                                 + 0.971712 w_k1_MSFT_s017
                                 + 1.10229 w_k1_PG_s017 >= 1
 z_k1_s0172_constraint:          var_1 + z_k1_s0172 + 1.28145 w_k1_ADM_s017
                                 + 1.19574 w_k1_AXP_s017
                                 + 1.47165 w_k1_DIS_s017
                                 + 1.26804 w_k1_FDX_s017
                                 + 1.18632 w_k1_IBM_s017
                                 + 1.07529 w_k1_INTC_s017
                                 + 1.20067 w_k1_KO_s017
                                 + 0.964208 w_k1_MAT_s017
                                 + 1.19961 w_k1_MSFT_s017
                                 + 1.38126 w_k1_PG_s017 >= 1
 z_k1_s0173_constraint:          var_1 + z_k1_s0173 + 0.901548 w_k1_ADM_s017
                                 + 1.13513 w_k1_AXP_s017
                                 + 0.944523 w_k1_DIS_s017
                                 + 1.01675 w_k1_FDX_s017
                                 + 0.876649 w_k1_IBM_s017
                                 + 0.859164 w_k1_INTC_s017
                                 + 0.881389 w_k1_KO_s017
                                 + 0.93306 w_k1_MAT_s017
                                 + 0.857753 w_k1_MSFT_s017
                                 + 1.04868 w_k1_PG_s017 >= 1
 z_k1_s0174_constraint:          var_1 + z_k1_s0174 + 1.06382 w_k1_ADM_s017
                                 + 1.04016 w_k1_AXP_s017 + 1.579 w_k1_DIS_s017
                                 + 1.17994 w_k1_FDX_s017
                                 + 0.97782 w_k1_IBM_s017
                                 + 1.27244 w_k1_INTC_s017
                                 + 1.14382 w_k1_KO_s017
                                 + 0.833436 w_k1_MAT_s017
                                 + 1.3134 w_k1_MSFT_s017 + 1.27023 w_k1_PG_s017
                                 >= 1
 z_k1_s0175_constraint:          var_1 + z_k1_s0175 + 0.840329 w_k1_ADM_s017
                                 + 0.881971 w_k1_AXP_s017
                                 + 0.926582 w_k1_DIS_s017
                                 + 1.04467 w_k1_FDX_s017
                                 + 0.604965 w_k1_IBM_s017
                                 + 0.955121 w_k1_INTC_s017
                                 + 0.764401 w_k1_KO_s017
                                 + 0.461439 w_k1_MAT_s017
                                 + 0.930464 w_k1_MSFT_s017
                                 + 0.956212 w_k1_PG_s017 >= 1
 z_k1_s0176_constraint:          var_1 + z_k1_s0176 + 0.832033 w_k1_ADM_s017
                                 + 0.831274 w_k1_AXP_s017
                                 + 0.758438 w_k1_DIS_s017
                                 + 0.815153 w_k1_FDX_s017
                                 + 0.811243 w_k1_IBM_s017
                                 + 0.829854 w_k1_INTC_s017
                                 + 0.792142 w_k1_KO_s017
                                 + 0.794866 w_k1_MAT_s017
                                 + 0.829565 w_k1_MSFT_s017
                                 + 0.894087 w_k1_PG_s017 >= 1
 z_k1_s0177_constraint:          var_1 + z_k1_s0177 + 1.06055 w_k1_ADM_s017
                                 + 1.15811 w_k1_AXP_s017
                                 + 1.21082 w_k1_DIS_s017
                                 + 0.987659 w_k1_FDX_s017
                                 + 1.81125 w_k1_IBM_s017
                                 + 0.70185 w_k1_INTC_s017
                                 + 1.48542 w_k1_KO_s017 + 2.00563 w_k1_MAT_s017
                                 + 1.04709 w_k1_MSFT_s017
                                 + 1.29755 w_k1_PG_s017 >= 1
 z_k1_s1000_constraint:          var_1 + z_k1_s1000 + 0.908095 w_k1_ADM_s100
                                 + 0.689278 w_k1_AXP_s100
                                 + 0.588769 w_k1_DIS_s100
                                 + 0.871804 w_k1_FDX_s100
                                 + 0.526311 w_k1_IBM_s100
                                 + 0.841444 w_k1_INTC_s100
                                 + 0.695698 w_k1_KO_s100
                                 + 0.532975 w_k1_MAT_s100
                                 + 0.911802 w_k1_MSFT_s100
                                 + 0.802035 w_k1_PG_s100 >= 1
 z_k1_s1001_constraint:          var_1 + z_k1_s1001 + 1.28393 w_k1_ADM_s100
                                 + 1.1042 w_k1_AXP_s100 + 1.466 w_k1_DIS_s100
                                 + 1.39713 w_k1_FDX_s100
                                 + 0.910888 w_k1_IBM_s100
                                 + 1.28826 w_k1_INTC_s100
                                 + 1.23576 w_k1_KO_s100 + 1.06677 w_k1_MAT_s100
                                 + 1.28863 w_k1_MSFT_s100 + 1.1485 w_k1_PG_s100
                                 >= 1
 z_k1_s1002_constraint:          var_1 + z_k1_s1002 + 1.05099 w_k1_ADM_s100
                                 + 0.926978 w_k1_AXP_s100
                                 + 0.869217 w_k1_DIS_s100
                                 + 1.01729 w_k1_FDX_s100
                                 + 0.763186 w_k1_IBM_s100
                                 + 0.999531 w_k1_INTC_s100
                                 + 0.875235 w_k1_KO_s100
                                 + 0.755222 w_k1_MAT_s100
                                 + 1.04427 w_k1_MSFT_s100
                                 + 0.985129 w_k1_PG_s100 >= 1
 z_k1_s1003_constraint:          var_1 + z_k1_s1003 + 0.925538 w_k1_ADM_s100
                                 + 0.845393 w_k1_AXP_s100
                                 + 0.581628 w_k1_DIS_s100
                                 + 0.797394 w_k1_FDX_s100
                                 + 0.942655 w_k1_IBM_s100
                                 + 0.822507 w_k1_INTC_s100
                                 + 0.796269 w_k1_KO_s100
                                 + 1.16472 w_k1_MAT_s100
                                 + 0.750014 w_k1_MSFT_s100
                                 + 0.834411 w_k1_PG_s100 >= 1
 z_k1_s1004_constraint:          var_1 + z_k1_s1004 + 0.745469 w_k1_ADM_s100
                                 + 0.669014 w_k1_AXP_s100
                                 + 0.61394 w_k1_DIS_s100
                                 + 0.712004 w_k1_FDX_s100
                                 + 0.792418 w_k1_IBM_s100
                                 + 0.781245 w_k1_INTC_s100
                                 + 0.762936 w_k1_KO_s100
                                 + 0.638385 w_k1_MAT_s100
                                 + 0.768516 w_k1_MSFT_s100
                                 + 0.769769 w_k1_PG_s100 >= 1
 z_k1_s1005_constraint:          var_1 + z_k1_s1005 + 0.982956 w_k1_ADM_s100
                                 + 1.17569 w_k1_AXP_s100
                                 + 1.08661 w_k1_DIS_s100
                                 + 1.09758 w_k1_FDX_s100
                                 + 1.14593 w_k1_IBM_s100
                                 + 1.21307 w_k1_INTC_s100
                                 + 1.09554 w_k1_KO_s100 + 1.12427 w_k1_MAT_s100
                                 + 1.15899 w_k1_MSFT_s100
                                 + 1.02717 w_k1_PG_s100 >= 1
 z_k1_s1006_constraint:          var_1 + z_k1_s1006 + 0.882468 w_k1_ADM_s100
                                 + 0.76689 w_k1_AXP_s100
                                 + 0.832795 w_k1_DIS_s100
                                 + 0.787222 w_k1_FDX_s100
                                 + 1.25753 w_k1_IBM_s100
                                 + 0.840473 w_k1_INTC_s100
                                 + 1.09403 w_k1_KO_s100 + 1.25605 w_k1_MAT_s100
                                 + 0.840692 w_k1_MSFT_s100
                                 + 0.943364 w_k1_PG_s100 >= 1
 z_k1_s1007_constraint:          var_1 + z_k1_s1007 + 1.2292 w_k1_ADM_s100
                                 + 1.16702 w_k1_AXP_s100
                                 + 1.17122 w_k1_DIS_s100
                                 + 0.993608 w_k1_FDX_s100
                                 + 1.74218 w_k1_IBM_s100
                                 + 1.18528 w_k1_INTC_s100
                                 + 1.42461 w_k1_KO_s100 + 1.57249 w_k1_MAT_s100
                                 + 1.13981 w_k1_MSFT_s100 + 1.1349 w_k1_PG_s100
                                 >= 1
 z_k1_s1010_constraint:          var_1 + z_k1_s1010 + 0.960074 w_k1_ADM_s101
                                 + 0.876572 w_k1_AXP_s101
                                 + 0.641802 w_k1_DIS_s101
                                 + 0.778504 w_k1_FDX_s101
                                 + 0.884754 w_k1_IBM_s101
                                 + 0.677081 w_k1_INTC_s101
                                 + 0.786578 w_k1_KO_s101
                                 + 1.15907 w_k1_MAT_s101
                                 + 0.728581 w_k1_MSFT_s101
                                 + 0.866475 w_k1_PG_s101 >= 1
 z_k1_s1011_constraint:          var_1 + z_k1_s1011 + 1.14934 w_k1_ADM_s101
                                 + 1.20064 w_k1_AXP_s101
                                 + 1.51619 w_k1_DIS_s101
                                 + 1.28801 w_k1_FDX_s101
                                 + 0.985407 w_k1_IBM_s101
                                 + 1.03341 w_k1_INTC_s101
                                 + 1.10392 w_k1_KO_s101 + 1.1312 w_k1_MAT_s101
                                 + 1.16702 w_k1_MSFT_s101
                                 + 1.07558 w_k1_PG_s101 >= 1
 z_k1_s1012_constraint:          var_1 + z_k1_s1012 + 1.43195 w_k1_ADM_s101
                                 + 1.56254 w_k1_AXP_s101
                                 + 1.76985 w_k1_DIS_s101
                                 + 1.32735 w_k1_FDX_s101
                                 + 1.37086 w_k1_IBM_s101
                                 + 1.0844 w_k1_INTC_s101 + 1.34755 w_k1_KO_s101
                                 + 1.99127 w_k1_MAT_s101
                                 + 1.30812 w_k1_MSFT_s101
                                 + 1.22785 w_k1_PG_s101 >= 1
 z_k1_s1013_constraint:          var_1 + z_k1_s1013 + 0.959917 w_k1_ADM_s101
                                 + 0.885631 w_k1_AXP_s101
                                 + 1.23817 w_k1_DIS_s101
                                 + 1.03104 w_k1_FDX_s101
                                 + 1.43405 w_k1_IBM_s101
                                 + 1.0519 w_k1_INTC_s101 + 1.2562 w_k1_KO_s101
                                 + 0.914346 w_k1_MAT_s101
                                 + 1.116 w_k1_MSFT_s101 + 1.08027 w_k1_PG_s101
                                 >= 1
 z_k1_s1014_constraint:          var_1 + z_k1_s1014 + 0.952423 w_k1_ADM_s101
                                 + 1.07857 w_k1_AXP_s101
                                 + 1.00383 w_k1_DIS_s101 + 1.0109 w_k1_FDX_s101
                                 + 1.19235 w_k1_IBM_s101
                                 + 1.02543 w_k1_INTC_s101
                                 + 1.06021 w_k1_KO_s101 + 1.30885 w_k1_MAT_s101
                                 + 0.887407 w_k1_MSFT_s101
                                 + 1.08291 w_k1_PG_s101 >= 1
 z_k1_s1015_constraint:          var_1 + z_k1_s1015 + 0.918596 w_k1_ADM_s101
                                 + 0.943012 w_k1_AXP_s101
                                 + 1.0005 w_k1_DIS_s101
                                 + 0.975233 w_k1_FDX_s101
                                 + 0.823293 w_k1_IBM_s101
                                 + 0.9397 w_k1_INTC_s101
                                 + 0.872738 w_k1_KO_s101
                                 + 0.729049 w_k1_MAT_s101
                                 + 0.872716 w_k1_MSFT_s101
                                 + 0.936478 w_k1_PG_s101 >= 1
 z_k1_s1016_constraint:          var_1 + z_k1_s1016 + 1.44783 w_k1_ADM_s101
                                 + 1.55126 w_k1_AXP_s101
                                 + 1.04146 w_k1_DIS_s101 + 1.2168 w_k1_FDX_s101
                                 + 1.0754 w_k1_IBM_s101
                                 + 1.07854 w_k1_INTC_s101
                                 + 0.908423 w_k1_KO_s101
                                 + 0.870582 w_k1_MAT_s101
                                 + 0.865763 w_k1_MSFT_s101
                                 + 1.16098 w_k1_PG_s101 >= 1
 z_k1_s1017_constraint:          var_1 + z_k1_s1017 + 0.972459 w_k1_ADM_s101
                                 + 0.966725 w_k1_AXP_s101
                                 + 0.955092 w_k1_DIS_s101
                                 + 0.785015 w_k1_FDX_s101
                                 + 1.29888 w_k1_IBM_s101
                                 + 0.741262 w_k1_INTC_s101
                                 + 1.00782 w_k1_KO_s101 + 1.47868 w_k1_MAT_s101
                                 + 0.806277 w_k1_MSFT_s101
                                 + 0.939776 w_k1_PG_s101 >= 1
 z_k1_s1020_constraint:          var_1 + z_k1_s1020 + 1.21615 w_k1_ADM_s102
                                 + 1.62978 w_k1_AXP_s102
                                 + 1.40722 w_k1_DIS_s102
                                 + 1.33893 w_k1_FDX_s102
                                 + 1.05288 w_k1_IBM_s102
                                 + 1.37342 w_k1_INTC_s102
                                 + 1.10703 w_k1_KO_s102 + 1.26913 w_k1_MAT_s102
                                 + 1.33166 w_k1_MSFT_s102
                                 + 1.15564 w_k1_PG_s102 >= 1
 z_k1_s1021_constraint:          var_1 + z_k1_s1021 + 1.00677 w_k1_ADM_s102
                                 + 1.06619 w_k1_AXP_s102
                                 + 1.02616 w_k1_DIS_s102
                                 + 1.07425 w_k1_FDX_s102
                                 + 0.815016 w_k1_IBM_s102
                                 + 0.941486 w_k1_INTC_s102 + 0.783 w_k1_KO_s102
                                 + 0.877558 w_k1_MAT_s102
                                 + 0.975393 w_k1_MSFT_s102
                                 + 1.03773 w_k1_PG_s102 >= 1
 z_k1_s1022_constraint:          var_1 + z_k1_s1022 + 1.10178 w_k1_ADM_s102
                                 + 1.34866 w_k1_AXP_s102
                                 + 0.799971 w_k1_DIS_s102
                                 + 0.969356 w_k1_FDX_s102
                                 + 1.09201 w_k1_IBM_s102
                                 + 1.17171 w_k1_INTC_s102
                                 + 0.824431 w_k1_KO_s102
                                 + 0.985572 w_k1_MAT_s102
                                 + 1.00599 w_k1_MSFT_s102
                                 + 0.930361 w_k1_PG_s102 >= 1
 z_k1_s1023_constraint:          var_1 + z_k1_s1023 + 0.699811 w_k1_ADM_s102
                                 + 0.910614 w_k1_AXP_s102
                                 + 0.609799 w_k1_DIS_s102
                                 + 0.740548 w_k1_FDX_s102
                                 + 1.28395 w_k1_IBM_s102
                                 + 0.977297 w_k1_INTC_s102
                                 + 0.821148 w_k1_KO_s102 + 1.1196 w_k1_MAT_s102
                                 + 0.824646 w_k1_MSFT_s102
                                 + 0.810763 w_k1_PG_s102 >= 1
 z_k1_s1024_constraint:          var_1 + z_k1_s1024 + 0.863801 w_k1_ADM_s102
                                 + 1.13082 w_k1_AXP_s102 + 1.1214 w_k1_DIS_s102
                                 + 0.955989 w_k1_FDX_s102
                                 + 1.85147 w_k1_IBM_s102
                                 + 1.12438 w_k1_INTC_s102
                                 + 1.16239 w_k1_KO_s102 + 1.72448 w_k1_MAT_s102
                                 + 0.990694 w_k1_MSFT_s102
                                 + 0.977524 w_k1_PG_s102 >= 1
 z_k1_s1025_constraint:          var_1 + z_k1_s1025 + 1.04434 w_k1_ADM_s102
                                 + 0.867555 w_k1_AXP_s102
                                 + 1.09641 w_k1_DIS_s102
                                 + 0.984961 w_k1_FDX_s102
                                 + 1.30526 w_k1_IBM_s102
                                 + 0.986611 w_k1_INTC_s102
                                 + 1.2432 w_k1_KO_s102 + 1.17079 w_k1_MAT_s102
                                 + 1.11066 w_k1_MSFT_s102
                                 + 1.00765 w_k1_PG_s102 >= 1
 z_k1_s1026_constraint:          var_1 + z_k1_s1026 + 0.774002 w_k1_ADM_s102
                                 + 0.910656 w_k1_AXP_s102
                                 + 0.660327 w_k1_DIS_s102
                                 + 0.907918 w_k1_FDX_s102
                                 + 0.599314 w_k1_IBM_s102
                                 + 1.23331 w_k1_INTC_s102
                                 + 0.675784 w_k1_KO_s102
                                 + 0.580702 w_k1_MAT_s102
                                 + 0.868718 w_k1_MSFT_s102
                                 + 0.826855 w_k1_PG_s102 >= 1
 z_k1_s1027_constraint:          var_1 + z_k1_s1027 + 0.861025 w_k1_ADM_s102
                                 + 0.796499 w_k1_AXP_s102
                                 + 0.809204 w_k1_DIS_s102
                                 + 0.784387 w_k1_FDX_s102
                                 + 0.864785 w_k1_IBM_s102
                                 + 1.07982 w_k1_INTC_s102
                                 + 0.829081 w_k1_KO_s102
                                 + 0.748557 w_k1_MAT_s102
                                 + 0.928466 w_k1_MSFT_s102
                                 + 0.857549 w_k1_PG_s102 >= 1
 z_k1_s1030_constraint:          var_1 + z_k1_s1030 + 1.17409 w_k1_ADM_s103
                                 + 1.26532 w_k1_AXP_s103
                                 + 1.34107 w_k1_DIS_s103
                                 + 1.29172 w_k1_FDX_s103
                                 + 0.796889 w_k1_IBM_s103
                                 + 1.25123 w_k1_INTC_s103
                                 + 1.03823 w_k1_KO_s103
                                 + 0.860613 w_k1_MAT_s103
                                 + 1.13268 w_k1_MSFT_s103
                                 + 1.08071 w_k1_PG_s103 >= 1
 z_k1_s1031_constraint:          var_1 + z_k1_s1031 + 0.948972 w_k1_ADM_s103
                                 + 1.20576 w_k1_AXP_s103
                                 + 0.943256 w_k1_DIS_s103
                                 + 1.24664 w_k1_FDX_s103
                                 + 0.617217 w_k1_IBM_s103
                                 + 1.19672 w_k1_INTC_s103
                                 + 0.761113 w_k1_KO_s103
                                 + 0.813393 w_k1_MAT_s103
                                 + 1.1374 w_k1_MSFT_s103
                                 + 0.969851 w_k1_PG_s103 >= 1
 z_k1_s1032_constraint:          var_1 + z_k1_s1032 + 1.36902 w_k1_ADM_s103
                                 + 1.45746 w_k1_AXP_s103
                                 + 1.87389 w_k1_DIS_s103
                                 + 1.60163 w_k1_FDX_s103
                                 + 0.782736 w_k1_IBM_s103
                                 + 1.25996 w_k1_INTC_s103
                                 + 1.03171 w_k1_KO_s103
                                 + 0.940863 w_k1_MAT_s103
                                 + 1.30343 w_k1_MSFT_s103
                                 + 1.25956 w_k1_PG_s103 >= 1
 z_k1_s1033_constraint:          var_1 + z_k1_s1033 + 1.01234 w_k1_ADM_s103
                                 + 0.934353 w_k1_AXP_s103
                                 + 1.22309 w_k1_DIS_s103
                                 + 0.955463 w_k1_FDX_s103
                                 + 1.34345 w_k1_IBM_s103
                                 + 0.96506 w_k1_INTC_s103
                                 + 1.24144 w_k1_KO_s103 + 1.05119 w_k1_MAT_s103
                                 + 1.01399 w_k1_MSFT_s103
                                 + 1.06135 w_k1_PG_s103 >= 1
 z_k1_s1034_constraint:          var_1 + z_k1_s1034 + 0.945035 w_k1_ADM_s103
                                 + 1.01021 w_k1_AXP_s103
                                 + 1.01702 w_k1_DIS_s103
                                 + 1.02962 w_k1_FDX_s103
                                 + 0.990047 w_k1_IBM_s103
                                 + 1.17885 w_k1_INTC_s103
                                 + 1.07242 w_k1_KO_s103 + 1.21387 w_k1_MAT_s103
                                 + 0.953078 w_k1_MSFT_s103
                                 + 0.999956 w_k1_PG_s103 >= 1
 z_k1_s1035_constraint:          var_1 + z_k1_s1035 + 0.799297 w_k1_ADM_s103
                                 + 0.726331 w_k1_AXP_s103
                                 + 0.670317 w_k1_DIS_s103
                                 + 0.868448 w_k1_FDX_s103
                                 + 0.659386 w_k1_IBM_s103
                                 + 1.1208 w_k1_INTC_s103
                                 + 0.707559 w_k1_KO_s103
                                 + 0.659529 w_k1_MAT_s103
                                 + 0.826498 w_k1_MSFT_s103
                                 + 0.878984 w_k1_PG_s103 >= 1
 z_k1_s1036_constraint:          var_1 + z_k1_s1036 + 0.569405 w_k1_ADM_s103
                                 + 0.656689 w_k1_AXP_s103
                                 + 0.649354 w_k1_DIS_s103
                                 + 0.661434 w_k1_FDX_s103
                                 + 1.53438 w_k1_IBM_s103
                                 + 0.874616 w_k1_INTC_s103
                                 + 0.968088 w_k1_KO_s103
                                 + 1.18918 w_k1_MAT_s103
                                 + 0.732063 w_k1_MSFT_s103
                                 + 0.847998 w_k1_PG_s103 >= 1
 z_k1_s1037_constraint:          var_1 + z_k1_s1037 + 0.880228 w_k1_ADM_s103
                                 + 1.49509 w_k1_AXP_s103
                                 + 1.27043 w_k1_DIS_s103
                                 + 1.08753 w_k1_FDX_s103
                                 + 1.48034 w_k1_IBM_s103
                                 + 1.19311 w_k1_INTC_s103
                                 + 1.24569 w_k1_KO_s103 + 1.95436 w_k1_MAT_s103
                                 + 1.0395 w_k1_MSFT_s103 + 1.09373 w_k1_PG_s103
                                 >= 1
 z_k1_s1040_constraint:          var_1 + z_k1_s1040 + 1.05633 w_k1_ADM_s104
                                 + 1.04257 w_k1_AXP_s104
                                 + 1.03284 w_k1_DIS_s104 + 1.0204 w_k1_FDX_s104
                                 + 0.88959 w_k1_IBM_s104
                                 + 1.00578 w_k1_INTC_s104
                                 + 1.14491 w_k1_KO_s104
                                 + 0.826847 w_k1_MAT_s104
                                 + 0.953932 w_k1_MSFT_s104
                                 + 1.05389 w_k1_PG_s104 >= 1
 z_k1_s1041_constraint:          var_1 + z_k1_s1041 + 1.01934 w_k1_ADM_s104
                                 + 1.19381 w_k1_AXP_s104
                                 + 1.08766 w_k1_DIS_s104
                                 + 0.990342 w_k1_FDX_s104
                                 + 1.37815 w_k1_IBM_s104
                                 + 0.952367 w_k1_INTC_s104
                                 + 1.23569 w_k1_KO_s104 + 1.24282 w_k1_MAT_s104
                                 + 1.00929 w_k1_MSFT_s104
                                 + 1.08169 w_k1_PG_s104 >= 1
 z_k1_s1042_constraint:          var_1 + z_k1_s1042 + 0.854647 w_k1_ADM_s104
                                 + 0.816232 w_k1_AXP_s104
                                 + 0.907552 w_k1_DIS_s104
                                 + 0.792704 w_k1_FDX_s104
                                 + 1.18437 w_k1_IBM_s104
                                 + 0.87674 w_k1_INTC_s104
                                 + 1.10215 w_k1_KO_s104 + 1.06787 w_k1_MAT_s104
                                 + 0.845124 w_k1_MSFT_s104
                                 + 0.926259 w_k1_PG_s104 >= 1
 z_k1_s1043_constraint:          var_1 + z_k1_s1043 + 0.926317 w_k1_ADM_s104
                                 + 1.00254 w_k1_AXP_s104
                                 + 1.16848 w_k1_DIS_s104 + 1.1123 w_k1_FDX_s104
                                 + 0.862908 w_k1_IBM_s104
                                 + 0.947469 w_k1_INTC_s104
                                 + 1.16496 w_k1_KO_s104 + 0.79893 w_k1_MAT_s104
                                 + 1.0647 w_k1_MSFT_s104 + 1.02878 w_k1_PG_s104
                                 >= 1
 z_k1_s1044_constraint:          var_1 + z_k1_s1044 + 0.951732 w_k1_ADM_s104
                                 + 0.803361 w_k1_AXP_s104
                                 + 1.39456 w_k1_DIS_s104
                                 + 0.950302 w_k1_FDX_s104
                                 + 1.10678 w_k1_IBM_s104
                                 + 0.848428 w_k1_INTC_s104
                                 + 1.19836 w_k1_KO_s104
                                 + 0.798542 w_k1_MAT_s104
                                 + 1.00448 w_k1_MSFT_s104
                                 + 1.06657 w_k1_PG_s104 >= 1
 z_k1_s1045_constraint:          var_1 + z_k1_s1045 + 1.38267 w_k1_ADM_s104
                                 + 1.60832 w_k1_AXP_s104
                                 + 1.62936 w_k1_DIS_s104
                                 + 1.29285 w_k1_FDX_s104
                                 + 1.20525 w_k1_IBM_s104
                                 + 1.29281 w_k1_INTC_s104
                                 + 1.53578 w_k1_KO_s104 + 1.39001 w_k1_MAT_s104
                                 + 1.34264 w_k1_MSFT_s104
                                 + 1.15761 w_k1_PG_s104 >= 1
 z_k1_s1046_constraint:          var_1 + z_k1_s1046 + 0.77135 w_k1_ADM_s104
                                 + 0.749922 w_k1_AXP_s104
                                 + 0.73967 w_k1_DIS_s104
                                 + 0.727745 w_k1_FDX_s104
                                 + 0.880374 w_k1_IBM_s104
                                 + 0.838124 w_k1_INTC_s104
                                 + 1.06475 w_k1_KO_s104
                                 + 0.741133 w_k1_MAT_s104
                                 + 0.815647 w_k1_MSFT_s104
                                 + 0.820539 w_k1_PG_s104 >= 1
 z_k1_s1047_constraint:          var_1 + z_k1_s1047 + 0.80925 w_k1_ADM_s104
                                 + 0.785539 w_k1_AXP_s104
                                 + 0.751898 w_k1_DIS_s104
                                 + 0.844703 w_k1_FDX_s104
                                 + 0.658791 w_k1_IBM_s104
                                 + 0.748981 w_k1_INTC_s104
                                 + 0.815371 w_k1_KO_s104
                                 + 0.572053 w_k1_MAT_s104
                                 + 0.731169 w_k1_MSFT_s104
                                 + 0.857212 w_k1_PG_s104 >= 1
 z_k1_s1050_constraint:          var_1 + z_k1_s1050 + 1.40578 w_k1_ADM_s105
                                 + 1.35836 w_k1_AXP_s105
                                 + 1.20988 w_k1_DIS_s105 + 1.1881 w_k1_FDX_s105
                                 + 0.722434 w_k1_IBM_s105
                                 + 1.12254 w_k1_INTC_s105
                                 + 0.979463 w_k1_KO_s105
                                 + 0.83869 w_k1_MAT_s105
                                 + 1.10787 w_k1_MSFT_s105
                                 + 1.04802 w_k1_PG_s105 >= 1
 z_k1_s1051_constraint:          var_1 + z_k1_s1051 + 1.22548 w_k1_ADM_s105
                                 + 1.85275 w_k1_AXP_s105
                                 + 1.95357 w_k1_DIS_s105
                                 + 1.52083 w_k1_FDX_s105
                                 + 1.15083 w_k1_IBM_s105
                                 + 1.47152 w_k1_INTC_s105
                                 + 1.41796 w_k1_KO_s105 + 1.48143 w_k1_MAT_s105
                                 + 1.41613 w_k1_MSFT_s105
                                 + 1.39805 w_k1_PG_s105 >= 1
 z_k1_s1052_constraint:          var_1 + z_k1_s1052 + 1.15585 w_k1_ADM_s105
                                 + 1.17269 w_k1_AXP_s105
                                 + 1.09782 w_k1_DIS_s105
                                 + 1.00236 w_k1_FDX_s105
                                 + 1.39965 w_k1_IBM_s105
                                 + 1.18022 w_k1_INTC_s105
                                 + 1.40987 w_k1_KO_s105 + 1.2629 w_k1_MAT_s105
                                 + 1.04966 w_k1_MSFT_s105
                                 + 1.14455 w_k1_PG_s105 >= 1
 z_k1_s1053_constraint:          var_1 + z_k1_s1053 + 0.84681 w_k1_ADM_s105
                                 + 0.731179 w_k1_AXP_s105
                                 + 0.857584 w_k1_DIS_s105
                                 + 0.76049 w_k1_FDX_s105
                                 + 1.01188 w_k1_IBM_s105
                                 + 0.804493 w_k1_INTC_s105
                                 + 0.919647 w_k1_KO_s105
                                 + 0.842656 w_k1_MAT_s105
                                 + 0.931402 w_k1_MSFT_s105
                                 + 0.941589 w_k1_PG_s105 >= 1
 z_k1_s1054_constraint:          var_1 + z_k1_s1054 + 1.21717 w_k1_ADM_s105
                                 + 1.21612 w_k1_AXP_s105
                                 + 1.52687 w_k1_DIS_s105
                                 + 1.17024 w_k1_FDX_s105
                                 + 0.917622 w_k1_IBM_s105
                                 + 1.15332 w_k1_INTC_s105
                                 + 1.14269 w_k1_KO_s105
                                 + 0.815663 w_k1_MAT_s105
                                 + 1.19893 w_k1_MSFT_s105
                                 + 1.20655 w_k1_PG_s105 >= 1
 z_k1_s1055_constraint:          var_1 + z_k1_s1055 + 0.708198 w_k1_ADM_s105
                                 + 0.569462 w_k1_AXP_s105
                                 + 0.575977 w_k1_DIS_s105
                                 + 0.618751 w_k1_FDX_s105
                                 + 0.702356 w_k1_IBM_s105
                                 + 0.804775 w_k1_INTC_s105
                                 + 0.729782 w_k1_KO_s105
                                 + 0.49154 w_k1_MAT_s105 + 0.683 w_k1_MSFT_s105
                                 + 0.882607 w_k1_PG_s105 >= 1
 z_k1_s1056_constraint:          var_1 + z_k1_s1056 + 0.933023 w_k1_ADM_s105
                                 + 0.779456 w_k1_AXP_s105
                                 + 0.915831 w_k1_DIS_s105
                                 + 0.955922 w_k1_FDX_s105
                                 + 0.624908 w_k1_IBM_s105
                                 + 0.926147 w_k1_INTC_s105
                                 + 0.733412 w_k1_KO_s105
                                 + 0.48518 w_k1_MAT_s105
                                 + 1.01157 w_k1_MSFT_s105
                                 + 0.963624 w_k1_PG_s105 >= 1
 z_k1_s1057_constraint:          var_1 + z_k1_s1057 + 1.33797 w_k1_ADM_s105
                                 + 1.05619 w_k1_AXP_s105
                                 + 1.06273 w_k1_DIS_s105
                                 + 0.978452 w_k1_FDX_s105
                                 + 1.03648 w_k1_IBM_s105
                                 + 0.988325 w_k1_INTC_s105
                                 + 1.03713 w_k1_KO_s105
                                 + 0.763353 w_k1_MAT_s105
                                 + 1.01376 w_k1_MSFT_s105
                                 + 1.09107 w_k1_PG_s105 >= 1
 z_k1_s1060_constraint:          var_1 + z_k1_s1060 + 1.05803 w_k1_ADM_s106
                                 + 0.892596 w_k1_AXP_s106
                                 + 1.34393 w_k1_DIS_s106
                                 + 0.949251 w_k1_FDX_s106
                                 + 1.09318 w_k1_IBM_s106
                                 + 0.97223 w_k1_INTC_s106
                                 + 1.21985 w_k1_KO_s106 + 1.07165 w_k1_MAT_s106
                                 + 1.08723 w_k1_MSFT_s106
                                 + 1.07242 w_k1_PG_s106 >= 1
 z_k1_s1061_constraint:          var_1 + z_k1_s1061 + 1.05426 w_k1_ADM_s106
                                 + 1.11495 w_k1_AXP_s106
                                 + 0.945392 w_k1_DIS_s106
                                 + 0.959837 w_k1_FDX_s106
                                 + 1.19186 w_k1_IBM_s106
                                 + 0.978493 w_k1_INTC_s106
                                 + 1.0925 w_k1_KO_s106 + 1.30753 w_k1_MAT_s106
                                 + 0.971855 w_k1_MSFT_s106
                                 + 1.0456 w_k1_PG_s106 >= 1
 z_k1_s1062_constraint:          var_1 + z_k1_s1062 + 1.62588 w_k1_ADM_s106
                                 + 1.42241 w_k1_AXP_s106
                                 + 1.57758 w_k1_DIS_s106
                                 + 1.26408 w_k1_FDX_s106
                                 + 1.72461 w_k1_IBM_s106
                                 + 1.16649 w_k1_INTC_s106
                                 + 1.65551 w_k1_KO_s106 + 1.42623 w_k1_MAT_s106
                                 + 1.30316 w_k1_MSFT_s106
                                 + 1.27276 w_k1_PG_s106 >= 1
 z_k1_s1063_constraint:          var_1 + z_k1_s1063 + 0.858717 w_k1_ADM_s106
                                 + 0.905251 w_k1_AXP_s106
                                 + 0.719634 w_k1_DIS_s106
                                 + 0.826302 w_k1_FDX_s106
                                 + 0.74307 w_k1_IBM_s106
                                 + 0.935484 w_k1_INTC_s106
                                 + 0.893625 w_k1_KO_s106
                                 + 0.945444 w_k1_MAT_s106
                                 + 0.889553 w_k1_MSFT_s106
                                 + 0.932253 w_k1_PG_s106 >= 1
 z_k1_s1064_constraint:          var_1 + z_k1_s1064 + 1.11694 w_k1_ADM_s106
                                 + 1.04867 w_k1_AXP_s106
                                 + 0.95841 w_k1_DIS_s106
                                 + 1.01101 w_k1_FDX_s106
                                 + 0.833268 w_k1_IBM_s106
                                 + 1.0818 w_k1_INTC_s106 + 1.03834 w_k1_KO_s106
                                 + 0.921176 w_k1_MAT_s106
                                 + 1.04306 w_k1_MSFT_s106
                                 + 0.969184 w_k1_PG_s106 >= 1
 z_k1_s1065_constraint:          var_1 + z_k1_s1065 + 0.829395 w_k1_ADM_s106
                                 + 0.77451 w_k1_AXP_s106
                                 + 0.644398 w_k1_DIS_s106
                                 + 0.6937 w_k1_FDX_s106
                                 + 0.803051 w_k1_IBM_s106
                                 + 0.929649 w_k1_INTC_s106
                                 + 0.807031 w_k1_KO_s106
                                 + 0.67979 w_k1_MAT_s106
                                 + 0.711493 w_k1_MSFT_s106
                                 + 0.810167 w_k1_PG_s106 >= 1
 z_k1_s1066_constraint:          var_1 + z_k1_s1066 + 1.34782 w_k1_ADM_s106
                                 + 0.908326 w_k1_AXP_s106
                                 + 0.992271 w_k1_DIS_s106
                                 + 1.01492 w_k1_FDX_s106
                                 + 0.678969 w_k1_IBM_s106
                                 + 0.91267 w_k1_INTC_s106
                                 + 1.01785 w_k1_KO_s106
                                 + 0.911776 w_k1_MAT_s106
                                 + 0.976199 w_k1_MSFT_s106
                                 + 0.973676 w_k1_PG_s106 >= 1
 z_k1_s1067_constraint:          var_1 + z_k1_s1067 + 1.5015 w_k1_ADM_s106
                                 + 1.51104 w_k1_AXP_s106
                                 + 1.25265 w_k1_DIS_s106
                                 + 1.20015 w_k1_FDX_s106
                                 + 1.11164 w_k1_IBM_s106
                                 + 1.05331 w_k1_INTC_s106
                                 + 1.23581 w_k1_KO_s106 + 1.38997 w_k1_MAT_s106
                                 + 0.981625 w_k1_MSFT_s106 + 1.186 w_k1_PG_s106
                                 >= 1
 z_k1_s1070_constraint:          var_1 + z_k1_s1070 + 0.731887 w_k1_ADM_s107
                                 + 0.757524 w_k1_AXP_s107
                                 + 0.637219 w_k1_DIS_s107
                                 + 0.804986 w_k1_FDX_s107
                                 + 0.63737 w_k1_IBM_s107
                                 + 1.02531 w_k1_INTC_s107
                                 + 0.794348 w_k1_KO_s107
                                 + 0.855151 w_k1_MAT_s107
                                 + 0.917114 w_k1_MSFT_s107
                                 + 0.850104 w_k1_PG_s107 >= 1
 z_k1_s1071_constraint:          var_1 + z_k1_s1071 + 0.968291 w_k1_ADM_s107
                                 + 0.837178 w_k1_AXP_s107
                                 + 1.11707 w_k1_DIS_s107
                                 + 0.945726 w_k1_FDX_s107
                                 + 0.998261 w_k1_IBM_s107
                                 + 1.18091 w_k1_INTC_s107
                                 + 1.29162 w_k1_KO_s107 + 1.22665 w_k1_MAT_s107
                                 + 1.14898 w_k1_MSFT_s107
                                 + 1.02842 w_k1_PG_s107 >= 1
 z_k1_s1072_constraint:          var_1 + z_k1_s1072 + 1.06255 w_k1_ADM_s107
                                 + 1.48209 w_k1_AXP_s107
                                 + 1.26598 w_k1_DIS_s107
                                 + 1.08542 w_k1_FDX_s107
                                 + 1.33541 w_k1_IBM_s107
                                 + 1.26465 w_k1_INTC_s107
                                 + 1.21128 w_k1_KO_s107 + 1.92562 w_k1_MAT_s107
                                 + 1.14476 w_k1_MSFT_s107
                                 + 1.02333 w_k1_PG_s107 >= 1
 z_k1_s1073_constraint:          var_1 + z_k1_s1073 + 0.863916 w_k1_ADM_s107
                                 + 0.940092 w_k1_AXP_s107
                                 + 0.848032 w_k1_DIS_s107
                                 + 1.04021 w_k1_FDX_s107
                                 + 0.588549 w_k1_IBM_s107
                                 + 1.03971 w_k1_INTC_s107
                                 + 0.693859 w_k1_KO_s107
                                 + 0.832213 w_k1_MAT_s107
                                 + 0.996236 w_k1_MSFT_s107
                                 + 0.829706 w_k1_PG_s107 >= 1
 z_k1_s1074_constraint:          var_1 + z_k1_s1074 + 1.22612 w_k1_ADM_s107
                                 + 1.0937 w_k1_AXP_s107
                                 + 0.951342 w_k1_DIS_s107
                                 + 0.975348 w_k1_FDX_s107
                                 + 0.924977 w_k1_IBM_s107
                                 + 1.10352 w_k1_INTC_s107
                                 + 1.10165 w_k1_KO_s107 + 1.20626 w_k1_MAT_s107
                                 + 1.06035 w_k1_MSFT_s107
                                 + 1.00024 w_k1_PG_s107 >= 1
 z_k1_s1075_constraint:          var_1 + z_k1_s1075 + 1.09359 w_k1_ADM_s107
                                 + 0.794856 w_k1_AXP_s107
                                 + 0.850114 w_k1_DIS_s107
                                 + 0.862121 w_k1_FDX_s107
                                 + 1.04381 w_k1_IBM_s107
                                 + 0.813024 w_k1_INTC_s107
                                 + 0.935008 w_k1_KO_s107
                                 + 0.956343 w_k1_MAT_s107
                                 + 0.928452 w_k1_MSFT_s107
                                 + 0.911799 w_k1_PG_s107 >= 1
 z_k1_s1076_constraint:          var_1 + z_k1_s1076 + 1.39438 w_k1_ADM_s107
                                 + 1.31155 w_k1_AXP_s107
                                 + 1.54112 w_k1_DIS_s107
                                 + 1.29492 w_k1_FDX_s107
                                 + 0.907859 w_k1_IBM_s107
                                 + 1.20565 w_k1_INTC_s107
                                 + 1.35823 w_k1_KO_s107 + 1.47169 w_k1_MAT_s107
                                 + 1.34486 w_k1_MSFT_s107 + 1.1525 w_k1_PG_s107
                                 >= 1
 z_k1_s1077_constraint:          var_1 + z_k1_s1077 + 0.667439 w_k1_ADM_s107
                                 + 0.746309 w_k1_AXP_s107
                                 + 0.710523 w_k1_DIS_s107
                                 + 0.677505 w_k1_FDX_s107
                                 + 1.03441 w_k1_IBM_s107
                                 + 0.889609 w_k1_INTC_s107
                                 + 0.919014 w_k1_KO_s107
                                 + 1.08625 w_k1_MAT_s107
                                 + 0.830763 w_k1_MSFT_s107
                                 + 0.891627 w_k1_PG_s107 >= 1
 z_k1_s1100_constraint:          var_1 + z_k1_s1100 + 0.984326 w_k1_ADM_s110
                                 + 0.986973 w_k1_AXP_s110
                                 + 0.690925 w_k1_DIS_s110
                                 + 0.891201 w_k1_FDX_s110
                                 + 0.528028 w_k1_IBM_s110
                                 + 0.989791 w_k1_INTC_s110
                                 + 0.632044 w_k1_KO_s110
                                 + 0.813645 w_k1_MAT_s110
                                 + 0.883882 w_k1_MSFT_s110
                                 + 0.893655 w_k1_PG_s110 >= 1
 z_k1_s1101_constraint:          var_1 + z_k1_s1101 + 1.2288 w_k1_ADM_s110
                                 + 1.63141 w_k1_AXP_s110
                                 + 1.73124 w_k1_DIS_s110
                                 + 1.44828 w_k1_FDX_s110
                                 + 1.22186 w_k1_IBM_s110
                                 + 1.65269 w_k1_INTC_s110
                                 + 1.37467 w_k1_KO_s110 + 1.23732 w_k1_MAT_s110
                                 + 1.3624 w_k1_MSFT_s110 + 1.2845 w_k1_PG_s110
                                 >= 1
 z_k1_s1102_constraint:          var_1 + z_k1_s1102 + 0.729758 w_k1_ADM_s110
                                 + 0.798704 w_k1_AXP_s110
                                 + 0.820072 w_k1_DIS_s110
                                 + 0.837184 w_k1_FDX_s110
                                 + 0.72591 w_k1_IBM_s110
                                 + 0.994216 w_k1_INTC_s110
                                 + 0.859547 w_k1_KO_s110
                                 + 0.752475 w_k1_MAT_s110
                                 + 0.92543 w_k1_MSFT_s110
                                 + 0.974244 w_k1_PG_s110 >= 1
 z_k1_s1103_constraint:          var_1 + z_k1_s1103 + 0.744511 w_k1_ADM_s110
                                 + 0.764972 w_k1_AXP_s110
                                 + 0.650989 w_k1_DIS_s110
                                 + 0.796668 w_k1_FDX_s110
                                 + 0.793978 w_k1_IBM_s110
                                 + 0.908442 w_k1_INTC_s110
                                 + 0.695118 w_k1_KO_s110
                                 + 0.805446 w_k1_MAT_s110
                                 + 0.825178 w_k1_MSFT_s110
                                 + 0.852019 w_k1_PG_s110 >= 1
 z_k1_s1104_constraint:          var_1 + z_k1_s1104 + 0.984392 w_k1_ADM_s110
                                 + 1.32852 w_k1_AXP_s110
                                 + 1.25963 w_k1_DIS_s110
                                 + 1.08324 w_k1_FDX_s110
                                 + 1.29444 w_k1_IBM_s110
                                 + 1.25458 w_k1_INTC_s110
                                 + 1.18512 w_k1_KO_s110 + 1.15569 w_k1_MAT_s110
                                 + 1.12645 w_k1_MSFT_s110
                                 + 1.10079 w_k1_PG_s110 >= 1
 z_k1_s1105_constraint:          var_1 + z_k1_s1105 + 0.896164 w_k1_ADM_s110
                                 + 1.05936 w_k1_AXP_s110
                                 + 0.864399 w_k1_DIS_s110
                                 + 0.858019 w_k1_FDX_s110
                                 + 1.25637 w_k1_IBM_s110
                                 + 0.931917 w_k1_INTC_s110
                                 + 1.06731 w_k1_KO_s110 + 1.1364 w_k1_MAT_s110
                                 + 0.916689 w_k1_MSFT_s110
                                 + 1.03512 w_k1_PG_s110 >= 1
 z_k1_s1106_constraint:          var_1 + z_k1_s1106 + 1.07019 w_k1_ADM_s110
                                 + 1.21207 w_k1_AXP_s110 + 1.2092 w_k1_DIS_s110
                                 + 1.09988 w_k1_FDX_s110
                                 + 0.833864 w_k1_IBM_s110
                                 + 1.12413 w_k1_INTC_s110
                                 + 1.04599 w_k1_KO_s110
                                 + 0.895053 w_k1_MAT_s110
                                 + 1.0506 w_k1_MSFT_s110 + 1.04829 w_k1_PG_s110
                                 >= 1
 z_k1_s1107_constraint:          var_1 + z_k1_s1107 + 0.857922 w_k1_ADM_s110
                                 + 1.06828 w_k1_AXP_s110
                                 + 0.854034 w_k1_DIS_s110
                                 + 0.933854 w_k1_FDX_s110
                                 + 0.938643 w_k1_IBM_s110
                                 + 1.13201 w_k1_INTC_s110
                                 + 0.896337 w_k1_KO_s110
                                 + 0.872546 w_k1_MAT_s110
                                 + 0.924121 w_k1_MSFT_s110
                                 + 0.98588 w_k1_PG_s110 >= 1
 z_k1_s1110_constraint:          var_1 + z_k1_s1110 + 0.877298 w_k1_ADM_s111
                                 + 1.08309 w_k1_AXP_s111
                                 + 1.13649 w_k1_DIS_s111 + 1.0556 w_k1_FDX_s111
                                 + 1.39229 w_k1_IBM_s111
                                 + 1.1041 w_k1_INTC_s111 + 1.14957 w_k1_KO_s111
                                 + 1.19198 w_k1_MAT_s111
                                 + 1.08417 w_k1_MSFT_s111 + 1.1194 w_k1_PG_s111
                                 >= 1
 z_k1_s1111_constraint:          var_1 + z_k1_s1111 + 1.32926 w_k1_ADM_s111
                                 + 1.7245 w_k1_AXP_s111 + 1.49956 w_k1_DIS_s111
                                 + 1.35136 w_k1_FDX_s111
                                 + 1.48018 w_k1_IBM_s111
                                 + 1.15017 w_k1_INTC_s111
                                 + 1.23965 w_k1_KO_s111 + 1.29443 w_k1_MAT_s111
                                 + 1.03347 w_k1_MSFT_s111 + 1.1051 w_k1_PG_s111
                                 >= 1
 z_k1_s1112_constraint:          var_1 + z_k1_s1112 + 0.614193 w_k1_ADM_s111
                                 + 0.65359 w_k1_AXP_s111
                                 + 0.575022 w_k1_DIS_s111
                                 + 0.699937 w_k1_FDX_s111
                                 + 0.880057 w_k1_IBM_s111
                                 + 1.02684 w_k1_INTC_s111
                                 + 0.738865 w_k1_KO_s111
                                 + 0.649116 w_k1_MAT_s111
                                 + 0.766342 w_k1_MSFT_s111
                                 + 0.793928 w_k1_PG_s111 >= 1
 z_k1_s1113_constraint:          var_1 + z_k1_s1113 + 1.00557 w_k1_ADM_s111
                                 + 1.00187 w_k1_AXP_s111
                                 + 0.957198 w_k1_DIS_s111
                                 + 0.964981 w_k1_FDX_s111
                                 + 0.929137 w_k1_IBM_s111
                                 + 0.855335 w_k1_INTC_s111
                                 + 0.893247 w_k1_KO_s111
                                 + 1.16932 w_k1_MAT_s111
                                 + 0.861872 w_k1_MSFT_s111
                                 + 0.884607 w_k1_PG_s111 >= 1
 z_k1_s1114_constraint:          var_1 + z_k1_s1114 + 0.843683 w_k1_ADM_s111
                                 + 0.695253 w_k1_AXP_s111
                                 + 0.717141 w_k1_DIS_s111
                                 + 0.838806 w_k1_FDX_s111
                                 + 0.621712 w_k1_IBM_s111
                                 + 0.844035 w_k1_INTC_s111
                                 + 0.69755 w_k1_KO_s111
                                 + 0.534347 w_k1_MAT_s111
                                 + 0.789491 w_k1_MSFT_s111
                                 + 0.707094 w_k1_PG_s111 >= 1
 z_k1_s1115_constraint:          var_1 + z_k1_s1115 + 0.921869 w_k1_ADM_s111
                                 + 0.989885 w_k1_AXP_s111
                                 + 1.07539 w_k1_DIS_s111
                                 + 1.10331 w_k1_FDX_s111
                                 + 0.757626 w_k1_IBM_s111
                                 + 1.15245 w_k1_INTC_s111
                                 + 0.89425 w_k1_KO_s111
                                 + 0.494603 w_k1_MAT_s111
                                 + 1.03007 w_k1_MSFT_s111
                                 + 0.861008 w_k1_PG_s111 >= 1
 z_k1_s1116_constraint:          var_1 + z_k1_s1116 + 0.870897 w_k1_ADM_s111
                                 + 0.975588 w_k1_AXP_s111
                                 + 1.0111 w_k1_DIS_s111
                                 + 0.928878 w_k1_FDX_s111
                                 + 1.17589 w_k1_IBM_s111
                                 + 1.25104 w_k1_INTC_s111
                                 + 0.98472 w_k1_KO_s111 + 1.24857 w_k1_MAT_s111
                                 + 0.913548 w_k1_MSFT_s111
                                 + 0.864439 w_k1_PG_s111 >= 1
 z_k1_s1117_constraint:          var_1 + z_k1_s1117 + 1.02565 w_k1_ADM_s111
                                 + 1.38424 w_k1_AXP_s111 + 1.4182 w_k1_DIS_s111
                                 + 1.29972 w_k1_FDX_s111
                                 + 0.998556 w_k1_IBM_s111
                                 + 1.46915 w_k1_INTC_s111
                                 + 1.05011 w_k1_KO_s111
                                 + 0.857598 w_k1_MAT_s111
                                 + 1.19658 w_k1_MSFT_s111
                                 + 1.05392 w_k1_PG_s111 >= 1
 z_k1_s1120_constraint:          var_1 + z_k1_s1120 + 0.738582 w_k1_ADM_s112
                                 + 0.970858 w_k1_AXP_s112
                                 + 0.797977 w_k1_DIS_s112
                                 + 0.84964 w_k1_FDX_s112
                                 + 1.24474 w_k1_IBM_s112
                                 + 0.862016 w_k1_INTC_s112
                                 + 0.914777 w_k1_KO_s112 + 1.4316 w_k1_MAT_s112
                                 + 0.844418 w_k1_MSFT_s112
                                 + 0.938684 w_k1_PG_s112 >= 1
 z_k1_s1121_constraint:          var_1 + z_k1_s1121 + 1.302 w_k1_ADM_s112
                                 + 1.42339 w_k1_AXP_s112
                                 + 1.45795 w_k1_DIS_s112
                                 + 1.37234 w_k1_FDX_s112
                                 + 1.25806 w_k1_IBM_s112
                                 + 1.21696 w_k1_INTC_s112
                                 + 1.04292 w_k1_KO_s112
                                 + 0.949786 w_k1_MAT_s112
                                 + 1.14169 w_k1_MSFT_s112
                                 + 0.979774 w_k1_PG_s112 >= 1
 z_k1_s1122_constraint:          var_1 + z_k1_s1122 + 1.15099 w_k1_ADM_s112
                                 + 1.94487 w_k1_AXP_s112
                                 + 1.31006 w_k1_DIS_s112
                                 + 1.30537 w_k1_FDX_s112
                                 + 1.50828 w_k1_IBM_s112
                                 + 1.17834 w_k1_INTC_s112
                                 + 1.16017 w_k1_KO_s112 + 2.01936 w_k1_MAT_s112
                                 + 1.26317 w_k1_MSFT_s112
                                 + 1.14999 w_k1_PG_s112 >= 1
 z_k1_s1123_constraint:          var_1 + z_k1_s1123 + 0.725981 w_k1_ADM_s112
                                 + 0.906379 w_k1_AXP_s112
                                 + 0.585645 w_k1_DIS_s112
                                 + 0.682405 w_k1_FDX_s112
                                 + 1.05174 w_k1_IBM_s112
                                 + 0.706032 w_k1_INTC_s112
                                 + 0.760935 w_k1_KO_s112
                                 + 0.939917 w_k1_MAT_s112
                                 + 0.557088 w_k1_MSFT_s112
                                 + 0.657496 w_k1_PG_s112 >= 1
 z_k1_s1124_constraint:          var_1 + z_k1_s1124 + 0.962971 w_k1_ADM_s112
                                 + 1.29285 w_k1_AXP_s112
                                 + 0.958096 w_k1_DIS_s112
                                 + 1.09588 w_k1_FDX_s112
                                 + 1.01931 w_k1_IBM_s112
                                 + 0.884985 w_k1_INTC_s112
                                 + 0.912012 w_k1_KO_s112
                                 + 1.23603 w_k1_MAT_s112
                                 + 0.885166 w_k1_MSFT_s112
                                 + 1.02588 w_k1_PG_s112 >= 1
 z_k1_s1125_constraint:          var_1 + z_k1_s1125 + 0.812334 w_k1_ADM_s112
                                 + 0.828779 w_k1_AXP_s112
                                 + 0.944052 w_k1_DIS_s112
                                 + 1.07507 w_k1_FDX_s112
                                 + 0.628497 w_k1_IBM_s112
                                 + 0.909603 w_k1_INTC_s112
                                 + 0.79117 w_k1_KO_s112
                                 + 0.663512 w_k1_MAT_s112
                                 + 0.998216 w_k1_MSFT_s112
                                 + 0.837089 w_k1_PG_s112 >= 1
 z_k1_s1126_constraint:          var_1 + z_k1_s1126 + 1.05601 w_k1_ADM_s112
                                 + 1.06748 w_k1_AXP_s112
                                 + 1.14448 w_k1_DIS_s112 + 1.0671 w_k1_FDX_s112
                                 + 1.05601 w_k1_IBM_s112
                                 + 0.855269 w_k1_INTC_s112
                                 + 1.08773 w_k1_KO_s112 + 1.16713 w_k1_MAT_s112
                                 + 1.14693 w_k1_MSFT_s112
                                 + 1.05059 w_k1_PG_s112 >= 1
 z_k1_s1127_constraint:          var_1 + z_k1_s1127 + 0.815225 w_k1_ADM_s112
                                 + 0.84027 w_k1_AXP_s112
                                 + 0.72536 w_k1_DIS_s112
                                 + 0.833469 w_k1_FDX_s112
                                 + 0.773086 w_k1_IBM_s112
                                 + 0.865942 w_k1_INTC_s112
                                 + 0.834406 w_k1_KO_s112
                                 + 1.06373 w_k1_MAT_s112
                                 + 0.913473 w_k1_MSFT_s112
                                 + 0.814608 w_k1_PG_s112 >= 1
 z_k1_s1130_constraint:          var_1 + z_k1_s1130 + 1.10475 w_k1_ADM_s113
                                 + 1.59764 w_k1_AXP_s113
                                 + 1.38369 w_k1_DIS_s113
                                 + 1.25952 w_k1_FDX_s113
                                 + 1.33133 w_k1_IBM_s113
                                 + 1.15482 w_k1_INTC_s113
                                 + 1.27319 w_k1_KO_s113 + 1.71471 w_k1_MAT_s113
                                 + 1.09404 w_k1_MSFT_s113
                                 + 1.12284 w_k1_PG_s113 >= 1
 z_k1_s1131_constraint:          var_1 + z_k1_s1131 + 0.647863 w_k1_ADM_s113
                                 + 0.854358 w_k1_AXP_s113
                                 + 0.899609 w_k1_DIS_s113
                                 + 0.907744 w_k1_FDX_s113
                                 + 1.29444 w_k1_IBM_s113
                                 + 0.828994 w_k1_INTC_s113
                                 + 1.07511 w_k1_KO_s113 + 1.56532 w_k1_MAT_s113
                                 + 0.879476 w_k1_MSFT_s113
                                 + 0.986047 w_k1_PG_s113 >= 1
 z_k1_s1132_constraint:          var_1 + z_k1_s1132 + 1.07751 w_k1_ADM_s113
                                 + 1.02525 w_k1_AXP_s113
                                 + 0.824436 w_k1_DIS_s113
                                 + 1.02091 w_k1_FDX_s113
                                 + 0.957417 w_k1_IBM_s113
                                 + 0.769142 w_k1_INTC_s113
                                 + 0.863837 w_k1_KO_s113
                                 + 1.17128 w_k1_MAT_s113
                                 + 0.903735 w_k1_MSFT_s113
                                 + 1.01539 w_k1_PG_s113 >= 1
 z_k1_s1133_constraint:          var_1 + z_k1_s1133 + 0.99339 w_k1_ADM_s113
                                 + 0.89713 w_k1_AXP_s113
                                 + 1.01133 w_k1_DIS_s113
                                 + 1.11571 w_k1_FDX_s113
                                 + 0.952243 w_k1_IBM_s113
                                 + 1.08905 w_k1_INTC_s113
                                 + 0.95162 w_k1_KO_s113
                                 + 0.781834 w_k1_MAT_s113
                                 + 1.07057 w_k1_MSFT_s113
                                 + 0.999556 w_k1_PG_s113 >= 1
 z_k1_s1134_constraint:          var_1 + z_k1_s1134 + 1.0376 w_k1_ADM_s113
                                 + 0.989763 w_k1_AXP_s113
                                 + 0.898952 w_k1_DIS_s113
                                 + 1.05328 w_k1_FDX_s113
                                 + 0.682748 w_k1_IBM_s113
                                 + 0.799291 w_k1_INTC_s113
                                 + 0.79554 w_k1_KO_s113
                                 + 0.777871 w_k1_MAT_s113
                                 + 0.910666 w_k1_MSFT_s113
                                 + 1.0215 w_k1_PG_s113 >= 1
 z_k1_s1135_constraint:          var_1 + z_k1_s1135 + 0.718469 w_k1_ADM_s113
                                 + 0.577257 w_k1_AXP_s113
                                 + 0.645712 w_k1_DIS_s113
                                 + 0.7915 w_k1_FDX_s113
                                 + 0.717261 w_k1_IBM_s113
                                 + 0.70459 w_k1_INTC_s113
                                 + 0.695311 w_k1_KO_s113
                                 + 0.748339 w_k1_MAT_s113
                                 + 0.790766 w_k1_MSFT_s113
                                 + 0.795399 w_k1_PG_s113 >= 1
 z_k1_s1136_constraint:          var_1 + z_k1_s1136 + 1.10283 w_k1_ADM_s113
                                 + 1.0993 w_k1_AXP_s113 + 1.36982 w_k1_DIS_s113
                                 + 1.46414 w_k1_FDX_s113
                                 + 0.992283 w_k1_IBM_s113
                                 + 1.07963 w_k1_INTC_s113
                                 + 1.11873 w_k1_KO_s113 + 1.23048 w_k1_MAT_s113
                                 + 1.20794 w_k1_MSFT_s113
                                 + 1.14804 w_k1_PG_s113 >= 1
 z_k1_s1137_constraint:          var_1 + z_k1_s1137 + 0.916687 w_k1_ADM_s113
                                 + 0.677936 w_k1_AXP_s113
                                 + 0.89246 w_k1_DIS_s113
                                 + 0.878785 w_k1_FDX_s113
                                 + 0.896584 w_k1_IBM_s113
                                 + 0.628617 w_k1_INTC_s113
                                 + 0.840692 w_k1_KO_s113
                                 + 0.780317 w_k1_MAT_s113
                                 + 0.828464 w_k1_MSFT_s113
                                 + 0.914276 w_k1_PG_s113 >= 1
 z_k1_s1140_constraint:          var_1 + z_k1_s1140 + 0.741384 w_k1_ADM_s114
                                 + 0.730018 w_k1_AXP_s114
                                 + 0.651582 w_k1_DIS_s114
                                 + 0.797853 w_k1_FDX_s114
                                 + 0.757958 w_k1_IBM_s114
                                 + 1.05822 w_k1_INTC_s114
                                 + 0.796104 w_k1_KO_s114
                                 + 0.753642 w_k1_MAT_s114
                                 + 0.851591 w_k1_MSFT_s114
                                 + 0.888936 w_k1_PG_s114 >= 1
 z_k1_s1141_constraint:          var_1 + z_k1_s1141 + 0.91561 w_k1_ADM_s114
                                 + 1.00399 w_k1_AXP_s114
                                 + 0.924759 w_k1_DIS_s114
                                 + 0.972213 w_k1_FDX_s114
                                 + 1.40065 w_k1_IBM_s114
                                 + 0.849693 w_k1_INTC_s114
                                 + 1.18637 w_k1_KO_s114 + 1.5613 w_k1_MAT_s114
                                 + 0.808661 w_k1_MSFT_s114
                                 + 1.10826 w_k1_PG_s114 >= 1
 z_k1_s1142_constraint:          var_1 + z_k1_s1142 + 0.923682 w_k1_ADM_s114
                                 + 1.13169 w_k1_AXP_s114
                                 + 1.02481 w_k1_DIS_s114
                                 + 1.12804 w_k1_FDX_s114
                                 + 1.01394 w_k1_IBM_s114
                                 + 1.15416 w_k1_INTC_s114
                                 + 0.998957 w_k1_KO_s114
                                 + 1.25098 w_k1_MAT_s114
                                 + 0.978574 w_k1_MSFT_s114
                                 + 1.04176 w_k1_PG_s114 >= 1
 z_k1_s1143_constraint:          var_1 + z_k1_s1143 + 1.34074 w_k1_ADM_s114
                                 + 0.934129 w_k1_AXP_s114
                                 + 0.944939 w_k1_DIS_s114
                                 + 0.98402 w_k1_FDX_s114
                                 + 0.961375 w_k1_IBM_s114
                                 + 0.814174 w_k1_INTC_s114
                                 + 0.976648 w_k1_KO_s114
                                 + 1.03995 w_k1_MAT_s114
                                 + 0.838175 w_k1_MSFT_s114
                                 + 1.11727 w_k1_PG_s114 >= 1
 z_k1_s1144_constraint:          var_1 + z_k1_s1144 + 0.782522 w_k1_ADM_s114
                                 + 0.970377 w_k1_AXP_s114
                                 + 0.69334 w_k1_DIS_s114
                                 + 0.875978 w_k1_FDX_s114
                                 + 1.0992 w_k1_IBM_s114
                                 + 0.793276 w_k1_INTC_s114
                                 + 0.777791 w_k1_KO_s114
                                 + 1.70388 w_k1_MAT_s114
                                 + 0.834213 w_k1_MSFT_s114
                                 + 0.92708 w_k1_PG_s114 >= 1
 z_k1_s1145_constraint:          var_1 + z_k1_s1145 + 1.23294 w_k1_ADM_s114
                                 + 1.14276 w_k1_AXP_s114
                                 + 1.28861 w_k1_DIS_s114
                                 + 1.32264 w_k1_FDX_s114
                                 + 1.11664 w_k1_IBM_s114
                                 + 1.19234 w_k1_INTC_s114
                                 + 1.33234 w_k1_KO_s114 + 1.51327 w_k1_MAT_s114
                                 + 1.27538 w_k1_MSFT_s114
                                 + 1.42966 w_k1_PG_s114 >= 1
 z_k1_s1146_constraint:          var_1 + z_k1_s1146 + 0.959336 w_k1_ADM_s114
                                 + 1.03036 w_k1_AXP_s114
                                 + 0.748562 w_k1_DIS_s114
                                 + 1.0167 w_k1_FDX_s114
                                 + 0.775651 w_k1_IBM_s114
                                 + 0.86926 w_k1_INTC_s114
                                 + 0.910934 w_k1_KO_s114
                                 + 0.997647 w_k1_MAT_s114
                                 + 0.914517 w_k1_MSFT_s114
                                 + 1.17105 w_k1_PG_s114 >= 1
 z_k1_s1147_constraint:          var_1 + z_k1_s1147 + 0.981739 w_k1_ADM_s114
                                 + 0.711656 w_k1_AXP_s114
                                 + 0.895107 w_k1_DIS_s114
                                 + 0.960262 w_k1_FDX_s114
                                 + 0.675473 w_k1_IBM_s114
                                 + 0.826509 w_k1_INTC_s114
                                 + 0.823462 w_k1_KO_s114
                                 + 0.557749 w_k1_MAT_s114
                                 + 0.939278 w_k1_MSFT_s114
                                 + 0.897398 w_k1_PG_s114 >= 1
 z_k1_s1150_constraint:          var_1 + z_k1_s1150 + 1.05303 w_k1_ADM_s115
                                 + 0.870345 w_k1_AXP_s115
                                 + 0.717602 w_k1_DIS_s115
                                 + 0.908897 w_k1_FDX_s115
                                 + 0.560941 w_k1_IBM_s115
                                 + 1.13815 w_k1_INTC_s115
                                 + 0.796492 w_k1_KO_s115
                                 + 0.710868 w_k1_MAT_s115
                                 + 1.03379 w_k1_MSFT_s115
                                 + 0.963506 w_k1_PG_s115 >= 1
 z_k1_s1151_constraint:          var_1 + z_k1_s1151 + 1.15008 w_k1_ADM_s115
                                 + 1.27619 w_k1_AXP_s115
                                 + 1.02212 w_k1_DIS_s115
                                 + 1.02877 w_k1_FDX_s115
                                 + 1.07159 w_k1_IBM_s115
                                 + 1.18479 w_k1_INTC_s115
                                 + 1.01345 w_k1_KO_s115 + 1.23136 w_k1_MAT_s115
                                 + 1.08945 w_k1_MSFT_s115
                                 + 1.02369 w_k1_PG_s115 >= 1
 z_k1_s1152_constraint:          var_1 + z_k1_s1152 + 0.965322 w_k1_ADM_s115
                                 + 1.01157 w_k1_AXP_s115
                                 + 0.906112 w_k1_DIS_s115
                                 + 1.09406 w_k1_FDX_s115
                                 + 0.693395 w_k1_IBM_s115
                                 + 1.20206 w_k1_INTC_s115
                                 + 0.869518 w_k1_KO_s115
                                 + 0.782387 w_k1_MAT_s115
                                 + 1.19459 w_k1_MSFT_s115
                                 + 0.949181 w_k1_PG_s115 >= 1
 z_k1_s1153_constraint:          var_1 + z_k1_s1153 + 1.13276 w_k1_ADM_s115
                                 + 1.06141 w_k1_AXP_s115
                                 + 0.922069 w_k1_DIS_s115
                                 + 0.772573 w_k1_FDX_s115
                                 + 1.51442 w_k1_IBM_s115
                                 + 0.965592 w_k1_INTC_s115
                                 + 1.30259 w_k1_KO_s115 + 1.84369 w_k1_MAT_s115
                                 + 0.932936 w_k1_MSFT_s115
                                 + 1.10064 w_k1_PG_s115 >= 1
 z_k1_s1154_constraint:          var_1 + z_k1_s1154 + 0.748821 w_k1_ADM_s115
                                 + 0.762531 w_k1_AXP_s115
                                 + 0.656853 w_k1_DIS_s115
                                 + 0.766689 w_k1_FDX_s115
                                 + 0.771257 w_k1_IBM_s115
                                 + 1.16076 w_k1_INTC_s115
                                 + 0.973666 w_k1_KO_s115
                                 + 0.820894 w_k1_MAT_s115
                                 + 0.95914 w_k1_MSFT_s115
                                 + 0.871245 w_k1_PG_s115 >= 1
 z_k1_s1155_constraint:          var_1 + z_k1_s1155 + 1.10165 w_k1_ADM_s115
                                 + 0.811569 w_k1_AXP_s115
                                 + 1.22253 w_k1_DIS_s115
                                 + 1.00661 w_k1_FDX_s115
                                 + 0.872762 w_k1_IBM_s115
                                 + 1.36413 w_k1_INTC_s115
                                 + 1.03515 w_k1_KO_s115
                                 + 0.538252 w_k1_MAT_s115
                                 + 1.26664 w_k1_MSFT_s115
                                 + 0.99389 w_k1_PG_s115 >= 1
 z_k1_s1156_constraint:          var_1 + z_k1_s1156 + 1.03807 w_k1_ADM_s115
                                 + 1.57817 w_k1_AXP_s115
                                 + 1.48763 w_k1_DIS_s115
                                 + 1.28317 w_k1_FDX_s115
                                 + 1.32026 w_k1_IBM_s115
                                 + 1.43348 w_k1_INTC_s115
                                 + 1.42778 w_k1_KO_s115 + 1.48205 w_k1_MAT_s115
                                 + 1.43684 w_k1_MSFT_s115
                                 + 1.19878 w_k1_PG_s115 >= 1
 z_k1_s1157_constraint:          var_1 + z_k1_s1157 + 1.0735 w_k1_ADM_s115
                                 + 1.22924 w_k1_AXP_s115
                                 + 1.22621 w_k1_DIS_s115
                                 + 0.994776 w_k1_FDX_s115
                                 + 1.3317 w_k1_IBM_s115
                                 + 0.994329 w_k1_INTC_s115
                                 + 1.31004 w_k1_KO_s115 + 1.89688 w_k1_MAT_s115
                                 + 1.2674 w_k1_MSFT_s115 + 1.18137 w_k1_PG_s115
                                 >= 1
 z_k1_s1160_constraint:          var_1 + z_k1_s1160 + 1.16415 w_k1_ADM_s116
                                 + 1.30841 w_k1_AXP_s116
                                 + 1.65114 w_k1_DIS_s116
                                 + 1.25679 w_k1_FDX_s116
                                 + 1.27113 w_k1_IBM_s116
                                 + 1.15796 w_k1_INTC_s116
                                 + 1.49247 w_k1_KO_s116 + 1.21861 w_k1_MAT_s116
                                 + 1.27109 w_k1_MSFT_s116
                                 + 1.23133 w_k1_PG_s116 >= 1
 z_k1_s1161_constraint:          var_1 + z_k1_s1161 + 1.02083 w_k1_ADM_s116
                                 + 0.818981 w_k1_AXP_s116
                                 + 0.850927 w_k1_DIS_s116
                                 + 0.868886 w_k1_FDX_s116
                                 + 1.14824 w_k1_IBM_s116
                                 + 0.862796 w_k1_INTC_s116
                                 + 1.06555 w_k1_KO_s116
                                 + 0.934786 w_k1_MAT_s116
                                 + 0.921343 w_k1_MSFT_s116
                                 + 0.888804 w_k1_PG_s116 >= 1
 z_k1_s1162_constraint:          var_1 + z_k1_s1162 + 0.774882 w_k1_ADM_s116
                                 + 0.803995 w_k1_AXP_s116
                                 + 0.658058 w_k1_DIS_s116
                                 + 0.729956 w_k1_FDX_s116
                                 + 0.940559 w_k1_IBM_s116
                                 + 0.684578 w_k1_INTC_s116
                                 + 0.94232 w_k1_KO_s116 + 1.04674 w_k1_MAT_s116
                                 + 0.763514 w_k1_MSFT_s116
                                 + 0.95416 w_k1_PG_s116 >= 1
 z_k1_s1163_constraint:          var_1 + z_k1_s1163 + 1.1607 w_k1_ADM_s116
                                 + 1.15116 w_k1_AXP_s116 + 1.0066 w_k1_DIS_s116
                                 + 1.07791 w_k1_FDX_s116
                                 + 0.801571 w_k1_IBM_s116
                                 + 0.869183 w_k1_INTC_s116
                                 + 1.01127 w_k1_KO_s116
                                 + 0.885352 w_k1_MAT_s116
                                 + 1.05152 w_k1_MSFT_s116
                                 + 0.99791 w_k1_PG_s116 >= 1
 z_k1_s1164_constraint:          var_1 + z_k1_s1164 + 0.980418 w_k1_ADM_s116
                                 + 0.812029 w_k1_AXP_s116
                                 + 0.824453 w_k1_DIS_s116
                                 + 0.975385 w_k1_FDX_s116
                                 + 0.842627 w_k1_IBM_s116
                                 + 0.968671 w_k1_INTC_s116
                                 + 0.915197 w_k1_KO_s116
                                 + 0.679969 w_k1_MAT_s116
                                 + 1.0212 w_k1_MSFT_s116 + 1.0707 w_k1_PG_s116
                                 >= 1
 z_k1_s1165_constraint:          var_1 + z_k1_s1165 + 0.998089 w_k1_ADM_s116
                                 + 0.653785 w_k1_AXP_s116
                                 + 0.675779 w_k1_DIS_s116
                                 + 0.851068 w_k1_FDX_s116
                                 + 0.687837 w_k1_IBM_s116
                                 + 0.718603 w_k1_INTC_s116
                                 + 0.792012 w_k1_KO_s116
                                 + 0.428891 w_k1_MAT_s116
                                 + 0.892198 w_k1_MSFT_s116
                                 + 0.860751 w_k1_PG_s116 >= 1
 z_k1_s1166_constraint:          var_1 + z_k1_s1166 + 1.33057 w_k1_ADM_s116
                                 + 1.07488 w_k1_AXP_s116
                                 + 1.08434 w_k1_DIS_s116
                                 + 1.01774 w_k1_FDX_s116
                                 + 1.10204 w_k1_IBM_s116
                                 + 0.941301 w_k1_INTC_s116
                                 + 1.26155 w_k1_KO_s116 + 1.20432 w_k1_MAT_s116
                                 + 1.14335 w_k1_MSFT_s116 + 1.0916 w_k1_PG_s116
                                 >= 1
 z_k1_s1167_constraint:          var_1 + z_k1_s1167 + 1.01064 w_k1_ADM_s116
                                 + 1.27716 w_k1_AXP_s116
                                 + 0.871677 w_k1_DIS_s116
                                 + 1.12146 w_k1_FDX_s116
                                 + 1.13162 w_k1_IBM_s116
                                 + 1.08441 w_k1_INTC_s116
                                 + 1.03793 w_k1_KO_s116 + 1.14608 w_k1_MAT_s116
                                 + 1.06067 w_k1_MSFT_s116
                                 + 1.10381 w_k1_PG_s116 >= 1
 z_k1_s1170_constraint:          var_1 + z_k1_s1170 + 0.781178 w_k1_ADM_s117
                                 + 0.987131 w_k1_AXP_s117
                                 + 1.04036 w_k1_DIS_s117
                                 + 1.14749 w_k1_FDX_s117
                                 + 0.844709 w_k1_IBM_s117
                                 + 1.35194 w_k1_INTC_s117
                                 + 0.771473 w_k1_KO_s117
                                 + 0.956665 w_k1_MAT_s117
                                 + 1.15813 w_k1_MSFT_s117
                                 + 1.03775 w_k1_PG_s117 >= 1
 z_k1_s1171_constraint:          var_1 + z_k1_s1171 + 0.766434 w_k1_ADM_s117
                                 + 1.3262 w_k1_AXP_s117 + 1.162 w_k1_DIS_s117
                                 + 1.08209 w_k1_FDX_s117
                                 + 2.00809 w_k1_IBM_s117
                                 + 1.02469 w_k1_INTC_s117
                                 + 1.24582 w_k1_KO_s117 + 2.09183 w_k1_MAT_s117
                                 + 1.0977 w_k1_MSFT_s117 + 1.27206 w_k1_PG_s117
                                 >= 1
 z_k1_s1172_constraint:          var_1 + z_k1_s1172 + 1.01526 w_k1_ADM_s117
                                 + 1.34451 w_k1_AXP_s117
                                 + 1.57706 w_k1_DIS_s117
                                 + 1.31574 w_k1_FDX_s117
                                 + 1.37241 w_k1_IBM_s117
                                 + 1.43223 w_k1_INTC_s117
                                 + 1.20863 w_k1_KO_s117 + 1.63303 w_k1_MAT_s117
                                 + 1.55473 w_k1_MSFT_s117
                                 + 1.15622 w_k1_PG_s117 >= 1
 z_k1_s1173_constraint:          var_1 + z_k1_s1173 + 0.757146 w_k1_ADM_s117
                                 + 0.71701 w_k1_AXP_s117
                                 + 0.772677 w_k1_DIS_s117
                                 + 0.89953 w_k1_FDX_s117
                                 + 0.726987 w_k1_IBM_s117
                                 + 1.18396 w_k1_INTC_s117
                                 + 0.724958 w_k1_KO_s117
                                 + 0.554328 w_k1_MAT_s117
                                 + 0.96267 w_k1_MSFT_s117
                                 + 0.850437 w_k1_PG_s117 >= 1
 z_k1_s1174_constraint:          var_1 + z_k1_s1174 + 0.93828 w_k1_ADM_s117
                                 + 1.25086 w_k1_AXP_s117
                                 + 0.966133 w_k1_DIS_s117
                                 + 1.1732 w_k1_FDX_s117 + 1.1039 w_k1_IBM_s117
                                 + 1.28093 w_k1_INTC_s117
                                 + 0.868227 w_k1_KO_s117
                                 + 1.17588 w_k1_MAT_s117
                                 + 1.0844 w_k1_MSFT_s117 + 1.08867 w_k1_PG_s117
                                 >= 1
 z_k1_s1175_constraint:          var_1 + z_k1_s1175 + 0.854053 w_k1_ADM_s117
                                 + 0.719887 w_k1_AXP_s117
                                 + 0.831323 w_k1_DIS_s117
                                 + 0.977717 w_k1_FDX_s117
                                 + 0.608732 w_k1_IBM_s117
                                 + 1.12459 w_k1_INTC_s117
                                 + 0.699723 w_k1_KO_s117
                                 + 0.626938 w_k1_MAT_s117
                                 + 1.07842 w_k1_MSFT_s117
                                 + 0.91847 w_k1_PG_s117 >= 1
 z_k1_s1176_constraint:          var_1 + z_k1_s1176 + 1.01691 w_k1_ADM_s117
                                 + 1.35822 w_k1_AXP_s117
                                 + 1.24169 w_k1_DIS_s117
                                 + 1.37446 w_k1_FDX_s117
                                 + 0.854152 w_k1_IBM_s117
                                 + 1.50423 w_k1_INTC_s117
                                 + 1.02302 w_k1_KO_s117 + 1.07459 w_k1_MAT_s117
                                 + 1.39727 w_k1_MSFT_s117
                                 + 1.14191 w_k1_PG_s117 >= 1
 z_k1_s1177_constraint:          var_1 + z_k1_s1177 + 0.981712 w_k1_ADM_s117
                                 + 0.746612 w_k1_AXP_s117
                                 + 0.636562 w_k1_DIS_s117
                                 + 0.786236 w_k1_FDX_s117
                                 + 0.98432 w_k1_IBM_s117
                                 + 0.859492 w_k1_INTC_s117
                                 + 0.746438 w_k1_KO_s117
                                 + 0.931785 w_k1_MAT_s117
                                 + 0.896534 w_k1_MSFT_s117
                                 + 0.87789 w_k1_PG_s117 >= 1
 k1_return_constraint:           0.0268500981796701 w_k1_ADM_s000
                                 + 0.0112976753103245 w_k1_ADM_s001
                                 + 0.0576065893300059 w_k1_ADM_s002
                                 + 0.0145363576333413 w_k1_ADM_s003
                                 + 0.0296338527266304 w_k1_ADM_s004
                                 + 0.0232700586658286 w_k1_ADM_s005
                                 + 0.0209458175424566 w_k1_ADM_s006
                                 + 0.0473094123709464 w_k1_ADM_s007
                                 + 0.0301666793545205 w_k1_ADM_s010
                                 + 0.0366743013587972 w_k1_ADM_s011
                                 + 0.0211152130689815 w_k1_ADM_s012
                                 + 0.0223041173499432 w_k1_ADM_s013
                                 + 0.019397370724561 w_k1_ADM_s014
                                 + 0.0310608058322292 w_k1_ADM_s015
                                 + 0.03091740014278 w_k1_ADM_s016
                                 + 0.0283765390381709 w_k1_ADM_s017
                                 + 0.0956134411775079 w_k1_ADM_s100
                                 + 0.0308604797894573 w_k1_ADM_s101
                                 + 0.0917945870862861 w_k1_ADM_s102
                                 + 0.0381549212134365 w_k1_ADM_s103
                                 + 0.0444951693830579 w_k1_ADM_s104
                                 + 0.058958169977098 w_k1_ADM_s105
                                 + 0.0386422174712997 w_k1_ADM_s106
                                 + 0.0394463435651065 w_k1_ADM_s107
                                 + 0.0202518198356269 w_k1_ADM_s110
                                 + 0.0292573363911406 w_k1_ADM_s111
                                 + 0.00684743272132675 w_k1_ADM_s112
                                 + 0.0262912618728156 w_k1_ADM_s113
                                 + 0.00699947460332415 w_k1_ADM_s114
                                 + 0.0157585334671037 w_k1_ADM_s115
                                 + 0.0145639934392335 w_k1_ADM_s116
                                 + 0.0170373514459062 w_k1_ADM_s117
                                 + 0.025314962479757 w_k1_AXP_s000
                                 + 0.0130916817818015 w_k1_AXP_s001
                                 + 0.0601529076258559 w_k1_AXP_s002
                                 + 0.0127665545556833 w_k1_AXP_s003
                                 + 0.0304496721876142 w_k1_AXP_s004
                                 + 0.023640295504565 w_k1_AXP_s005
                                 + 0.017726897693504 w_k1_AXP_s006
                                 + 0.0486535649850325 w_k1_AXP_s007
                                 + 0.0269909655547775 w_k1_AXP_s010
                                 + 0.0395046511411315 w_k1_AXP_s011
                                 + 0.0212367674215858 w_k1_AXP_s012
                                 + 0.019206545100527 w_k1_AXP_s013
                                 + 0.0183783920950835 w_k1_AXP_s014
                                 + 0.0315012774229347 w_k1_AXP_s015
                                 + 0.0278770475734648 w_k1_AXP_s016
                                 + 0.0303176673503603 w_k1_AXP_s017
                                 + 0.0909319865495111 w_k1_AXP_s100
                                 + 0.0314679314622667 w_k1_AXP_s101
                                 + 0.104706565247335 w_k1_AXP_s102
                                 + 0.0416062339262379 w_k1_AXP_s103
                                 + 0.046857481819756 w_k1_AXP_s104
                                 + 0.0575034652528561 w_k1_AXP_s105
                                 + 0.0367473476568562 w_k1_AXP_s106
                                 + 0.0379500952725566 w_k1_AXP_s107
                                 + 0.0243115769010303 w_k1_AXP_s110
                                 + 0.0332238381536867 w_k1_AXP_s111
                                 + 0.00813522342232536 w_k1_AXP_s112
                                 + 0.0289648130512029 w_k1_AXP_s113
                                 + 0.00695320208824201 w_k1_AXP_s114
                                 + 0.0166069817813617 w_k1_AXP_s115
                                 + 0.0136445315173334 w_k1_AXP_s116
                                 + 0.0199026461244576 w_k1_AXP_s117
                                 + 0.0246204057911716 w_k1_DIS_s000
                                 + 0.011922826109655 w_k1_DIS_s001
                                 + 0.0560028723305457 w_k1_DIS_s002
                                 + 0.0127421563928458 w_k1_DIS_s003
                                 + 0.0320636415808403 w_k1_DIS_s004
                                 + 0.0239930802151653 w_k1_DIS_s005
                                 + 0.0200775016663631 w_k1_DIS_s006
                                 + 0.0476075347712624 w_k1_DIS_s007
                                 + 0.030475662634981 w_k1_DIS_s010
                                 + 0.0416608639374266 w_k1_DIS_s011
                                 + 0.0226067941928227 w_k1_DIS_s012
                                 + 0.0210588827386577 w_k1_DIS_s013
                                 + 0.020246358989364 w_k1_DIS_s014
                                 + 0.0336575685613399 w_k1_DIS_s015
                                 + 0.0299059207313658 w_k1_DIS_s016
                                 + 0.029893627864562 w_k1_DIS_s017
                                 + 0.0892515711032751 w_k1_DIS_s100
                                 + 0.0319563870961078 w_k1_DIS_s101
                                 + 0.0934506583437577 w_k1_DIS_s102
                                 + 0.0429863758736564 w_k1_DIS_s103
                                 + 0.0505255919330392 w_k1_DIS_s104
                                 + 0.0609782440420687 w_k1_DIS_s105
                                 + 0.0357729167401499 w_k1_DIS_s106
                                 + 0.0388613781733876 w_k1_DIS_s107
                                 + 0.0221356014705543 w_k1_DIS_s110
                                 + 0.0326955056094968 w_k1_DIS_s111
                                 + 0.00706036115917474 w_k1_DIS_s112
                                 + 0.0284970986985139 w_k1_DIS_s113
                                 + 0.00665804407761293 w_k1_DIS_s114
                                 + 0.0162166334406486 w_k1_DIS_s115
                                 + 0.0136115924167967 w_k1_DIS_s116
                                 + 0.0199546864382976 w_k1_DIS_s117
                                 + 0.0256109305727243 w_k1_FDX_s000
                                 + 0.0116787275203753 w_k1_FDX_s001
                                 + 0.0564898554264728 w_k1_FDX_s002
                                 + 0.0136809191521112 w_k1_FDX_s003
                                 + 0.0308294465486253 w_k1_FDX_s004
                                 + 0.0221778865547103 w_k1_FDX_s005
                                 + 0.0191173510198173 w_k1_FDX_s006
                                 + 0.0468329657127655 w_k1_FDX_s007
                                 + 0.0272977080486884 w_k1_FDX_s010
                                 + 0.0388146168229549 w_k1_FDX_s011
                                 + 0.0228258504776143 w_k1_FDX_s012
                                 + 0.0180800965000254 w_k1_FDX_s013
                                 + 0.0190766919439421 w_k1_FDX_s014
                                 + 0.0324462135419959 w_k1_FDX_s015
                                 + 0.0282065345909256 w_k1_FDX_s016
                                 + 0.0288436128985303 w_k1_FDX_s017
                                 + 0.093543512660849 w_k1_FDX_s100
                                 + 0.0297049544881134 w_k1_FDX_s101
                                 + 0.0944372325718526 w_k1_FDX_s102
                                 + 0.042185109108639 w_k1_FDX_s103
                                 + 0.04534823645508 w_k1_FDX_s104
                                 + 0.0539973764189196 w_k1_FDX_s105
                                 + 0.0333266434973669 w_k1_FDX_s106
                                 + 0.0362710466441668 w_k1_FDX_s107
                                 + 0.0215918605938904 w_k1_FDX_s110
                                 + 0.0318869692337513 w_k1_FDX_s111
                                 + 0.00732095798317433 w_k1_FDX_s112
                                 + 0.0296411615622361 w_k1_FDX_s113
                                 + 0.00733647940799621 w_k1_FDX_s114
                                 + 0.0152700319226355 w_k1_FDX_s115
                                 + 0.0134260867297022 w_k1_FDX_s116
                                 + 0.0209346534625028 w_k1_FDX_s117
                                 + 0.0260622981110338 w_k1_IBM_s000
                                 + 0.0115071001484053 w_k1_IBM_s001
                                 + 0.0556384920947586 w_k1_IBM_s002
                                 + 0.0135753327510721 w_k1_IBM_s003
                                 + 0.0293990413508017 w_k1_IBM_s004
                                 + 0.0240525610930102 w_k1_IBM_s005
                                 + 0.0193663979149743 w_k1_IBM_s006
                                 + 0.0480434842457769 w_k1_IBM_s007
                                 + 0.0277212468315002 w_k1_IBM_s010
                                 + 0.0392079505469875 w_k1_IBM_s011
                                 + 0.0222631839986283 w_k1_IBM_s012
                                 + 0.0188456373921974 w_k1_IBM_s013
                                 + 0.0188427147319807 w_k1_IBM_s014
                                 + 0.0319702480104011 w_k1_IBM_s015
                                 + 0.0288387248332166 w_k1_IBM_s016
                                 + 0.0280623308217314 w_k1_IBM_s017
                                 + 0.0980647049941557 w_k1_IBM_s100
                                 + 0.031515484477478 w_k1_IBM_s101
                                 + 0.0958025022708212 w_k1_IBM_s102
                                 + 0.0396157771847785 w_k1_IBM_s103
                                 + 0.0458465404285165 w_k1_IBM_s104
                                 + 0.0536890295138423 w_k1_IBM_s105
                                 + 0.0345315553342739 w_k1_IBM_s106
                                 + 0.036488473645686 w_k1_IBM_s107
                                 + 0.0218861506103942 w_k1_IBM_s110
                                 + 0.0322509786854981 w_k1_IBM_s111
                                 + 0.00783743157213504 w_k1_IBM_s112
                                 + 0.0282474751788954 w_k1_IBM_s113
                                 + 0.00727333255771264 w_k1_IBM_s114
                                 + 0.0161510708431714 w_k1_IBM_s115
                                 + 0.0137909512494091 w_k1_IBM_s116
                                 + 0.0196359045701476 w_k1_IBM_s117
                                 + 0.0267279897010812 w_k1_INTC_s000
                                 + 0.0144060440401619 w_k1_INTC_s001
                                 + 0.0629130884916879 w_k1_INTC_s002
                                 + 0.0124232650759526 w_k1_INTC_s003
                                 + 0.0314084637123478 w_k1_INTC_s004
                                 + 0.0234532648303471 w_k1_INTC_s005
                                 + 0.0149846244000828 w_k1_INTC_s006
                                 + 0.0408984260800216 w_k1_INTC_s007
                                 + 0.0257878579665481 w_k1_INTC_s010
                                 + 0.0387571667803922 w_k1_INTC_s011
                                 + 0.0233895763775287 w_k1_INTC_s012
                                 + 0.0175136277063511 w_k1_INTC_s013
                                 + 0.0187755245925579 w_k1_INTC_s014
                                 + 0.0280848352252155 w_k1_INTC_s015
                                 + 0.0305415528833324 w_k1_INTC_s016
                                 + 0.02578088312584 w_k1_INTC_s017
                                 + 0.0974934350365535 w_k1_INTC_s100
                                 + 0.0269457105897633 w_k1_INTC_s101
                                 + 0.103308187510658 w_k1_INTC_s102
                                 + 0.0452318832994207 w_k1_INTC_s103
                                 + 0.0435677384225711 w_k1_INTC_s104
                                 + 0.0565738539678814 w_k1_INTC_s105
                                 + 0.03354323166489 w_k1_INTC_s106
                                 + 0.039378466877943 w_k1_INTC_s107
                                 + 0.0244702057508588 w_k1_INTC_s110
                                 + 0.0345897946972578 w_k1_INTC_s111
                                 + 0.00663416518070634 w_k1_INTC_s112
                                 + 0.0251704055556853 w_k1_INTC_s113
                                 + 0.00706771374567458 w_k1_INTC_s114
                                 + 0.0187696651831211 w_k1_INTC_s115
                                 + 0.0123544190061013 w_k1_INTC_s116
                                 + 0.0236889700280725 w_k1_INTC_s117
                                 + 0.0264264865829363 w_k1_KO_s000
                                 + 0.012467347881846 w_k1_KO_s001
                                 + 0.060396137843946 w_k1_KO_s002
                                 + 0.0132638141228003 w_k1_KO_s003
                                 + 0.0320715740013752 w_k1_KO_s004
                                 + 0.0234029626365224 w_k1_KO_s005
                                 + 0.0194649033969577 w_k1_KO_s006
                                 + 0.0486583265657387 w_k1_KO_s007
                                 + 0.0269298586734179 w_k1_KO_s010
                                 + 0.037029355116355 w_k1_KO_s011
                                 + 0.0223964765901477 w_k1_KO_s012
                                 + 0.0188067608935912 w_k1_KO_s013
                                 + 0.0197264872765882 w_k1_KO_s014
                                 + 0.0319238920510776 w_k1_KO_s015
                                 + 0.0303732156061334 w_k1_KO_s016
                                 + 0.0276122159166788 w_k1_KO_s017
                                 + 0.0967271067142352 w_k1_KO_s100
                                 + 0.0295658819325757 w_k1_KO_s101
                                 + 0.0861035275903115 w_k1_KO_s102
                                 + 0.0403569838860661 w_k1_KO_s103
                                 + 0.0535786823741207 w_k1_KO_s104
                                 + 0.0582284426884984 w_k1_KO_s105
                                 + 0.0374050224234259 w_k1_KO_s106
                                 + 0.0405643526132977 w_k1_KO_s107
                                 + 0.0216228725612029 w_k1_KO_s110
                                 + 0.0297667632229641 w_k1_KO_s111
                                 + 0.00676636350990413 w_k1_KO_s112
                                 + 0.0273495938866169 w_k1_KO_s113
                                 + 0.00730578888239515 w_k1_KO_s114
                                 + 0.0175344704707935 w_k1_KO_s115
                                 + 0.0150061999029255 w_k1_KO_s116
                                 + 0.0171354721977246 w_k1_KO_s117
                                 + 0.0292656323067426 w_k1_MAT_s000
                                 + 0.0128056822031944 w_k1_MAT_s001
                                 + 0.0545576835333184 w_k1_MAT_s002
                                 + 0.0119474293579513 w_k1_MAT_s003
                                 + 0.0278568256884393 w_k1_MAT_s004
                                 + 0.0266046070210947 w_k1_MAT_s005
                                 + 0.0196433140779618 w_k1_MAT_s006
                                 + 0.0579618544061095 w_k1_MAT_s007
                                 + 0.0283104965315305 w_k1_MAT_s010
                                 + 0.0374143847254627 w_k1_MAT_s011
                                 + 0.0228972746191577 w_k1_MAT_s012
                                 + 0.0188925457258404 w_k1_MAT_s013
                                 + 0.014814872156219 w_k1_MAT_s014
                                 + 0.0312207413660384 w_k1_MAT_s015
                                 + 0.02641504468117 w_k1_MAT_s016
                                 + 0.0289790466626686 w_k1_MAT_s017
                                 + 0.0983660047199514 w_k1_MAT_s100
                                 + 0.0346538754808271 w_k1_MAT_s101
                                 + 0.0955169957964053 w_k1_MAT_s102
                                 + 0.0423602721393965 w_k1_MAT_s103
                                 + 0.0427177366185669 w_k1_MAT_s104
                                 + 0.0484263596148311 w_k1_MAT_s105
                                 + 0.0377592755564997 w_k1_MAT_s106
                                 + 0.0462715881287384 w_k1_MAT_s107
                                 + 0.0212520391572438 w_k1_MAT_s110
                                 + 0.0304352627295212 w_k1_MAT_s111
                                 + 0.00834874649823372 w_k1_MAT_s112
                                 + 0.0329980224959165 w_k1_MAT_s113
                                 + 0.00879406832631205 w_k1_MAT_s114
                                 + 0.0182205037646081 w_k1_MAT_s115
                                 + 0.0133950296979799 w_k1_MAT_s116
                                 + 0.0208967742239158 w_k1_MAT_s117
                                 + 0.0282908221700351 w_k1_MSFT_s000
                                 + 0.0122277499057355 w_k1_MSFT_s001
                                 + 0.0580702798933688 w_k1_MSFT_s002
                                 + 0.0127717202153276 w_k1_MSFT_s003
                                 + 0.0277303491482738 w_k1_MSFT_s004
                                 + 0.0244751896817299 w_k1_MSFT_s005
                                 + 0.0186498492094599 w_k1_MSFT_s006
                                 + 0.0462058505105704 w_k1_MSFT_s007
                                 + 0.0285609326001695 w_k1_MSFT_s010
                                 + 0.0362592260153677 w_k1_MSFT_s011
                                 + 0.0218156173713787 w_k1_MSFT_s012
                                 + 0.0180540895825492 w_k1_MSFT_s013
                                 + 0.0183060113800872 w_k1_MSFT_s014
                                 + 0.0300756375778784 w_k1_MSFT_s015
                                 + 0.0310398038370907 w_k1_MSFT_s016
                                 + 0.0275588014001607 w_k1_MSFT_s017
                                 + 0.0966789313355652 w_k1_MSFT_s100
                                 + 0.0276963654645267 w_k1_MSFT_s101
                                 + 0.0962989930743603 w_k1_MSFT_s102
                                 + 0.0394234301170507 w_k1_MSFT_s103
                                 + 0.0455016694236652 w_k1_MSFT_s104
                                 + 0.056172450863129 w_k1_MSFT_s105
                                 + 0.03328487667229 w_k1_MSFT_s106
                                 + 0.0396144091063913 w_k1_MSFT_s107
                                 + 0.021858930135886 w_k1_MSFT_s110
                                 + 0.0296205054081914 w_k1_MSFT_s111
                                 + 0.00677756377139217 w_k1_MSFT_s112
                                 + 0.0267162835580299 w_k1_MSFT_s113
                                 + 0.00681197806014797 w_k1_MSFT_s114
                                 + 0.0182035866629913 w_k1_MSFT_s115
                                 + 0.0139309130975551 w_k1_MSFT_s116
                                 + 0.0222930000955407 w_k1_MSFT_s117
                                 + 0.0272642038302463 w_k1_PG_s000
                                 + 0.0126852787287966 w_k1_PG_s001
                                 + 0.0554562236104795 w_k1_PG_s002
                                 + 0.0127537404644833 w_k1_PG_s003
                                 + 0.031502832492075 w_k1_PG_s004
                                 + 0.0256472374139823 w_k1_PG_s005
                                 + 0.0190510438715092 w_k1_PG_s006
                                 + 0.0433033092775545 w_k1_PG_s007
                                 + 0.0270345441444683 w_k1_PG_s010
                                 + 0.0375343900485633 w_k1_PG_s011
                                 + 0.0227229501063672 w_k1_PG_s012
                                 + 0.019173858064261 w_k1_PG_s013
                                 + 0.0187301848192192 w_k1_PG_s014
                                 + 0.0342324830617841 w_k1_PG_s015
                                 + 0.0292901974372446 w_k1_PG_s016
                                 + 0.0306261060531135 w_k1_PG_s017
                                 + 0.0929328797217272 w_k1_PG_s100
                                 + 0.0296988711512864 w_k1_PG_s101
                                 + 0.0908214529922369 w_k1_PG_s102
                                 + 0.0402189694085212 w_k1_PG_s103
                                 + 0.0456554856494352 w_k1_PG_s104
                                 + 0.0580150629663711 w_k1_PG_s105
                                 + 0.0349356660285571 w_k1_PG_s106
                                 + 0.0365022390021435 w_k1_PG_s107
                                 + 0.0223706923881144 w_k1_PG_s110
                                 + 0.0285913199302642 w_k1_PG_s111
                                 + 0.00668322798149015 w_k1_PG_s112
                                 + 0.0278104638762217 w_k1_PG_s113
                                 + 0.00784616687031122 w_k1_PG_s114
                                 + 0.0162355394624728 w_k1_PG_s115
                                 + 0.0139339359304028 w_k1_PG_s116
                                 + 0.019616149837876 w_k1_PG_s117 >= 1.13
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
