# -*- coding: utf-8 -*-
"""
Created on Sun May  1 23:17:43 2022

@author: Patrick Shen
"""
import pygame
from pygame.locals import *
from sys import exit
import random

SCREEN_WIDTH=500
SCREEN_HEIGHT=800

#bullets
class Bullet(pygame.sprite.Sprite):
    def __init__(self, bullet_img,init_pos):
        pygame.sprite.Sprite.__init__(self)#inherit_class
        self.image=bullet_img
        self.rect=self.image.get_rect()
        self.rect.midbottom=init_pos
        self.speed=10
    def move(self):
        self.rect.top-=self.spped
        
class Player(pygame.sprite.Sprite):
    def __init__(self, plane_img, player_rect,init_pos):
        pygame.sprite.Sprite.__init__(self)#inherit_class
        self.imager=[] #to_store_the_image
        for i in range(len(player_rect)):
            self.image.append(plane_img.subsurface(player_rect[i]).convert_alpha())
            self.rect=player_rect[0] #init_the rectangle of_images
            self.rect.topleft=init_pos #init_the Axis of the rectangle
            self.speed=8 #spped of the bullet
            self.bullets=pygame.sprite.Group() #set of all bullets that palyer has fired
            self.img_index=0 #index of image
            self.is_hit=False #judge whether is hit
    
    def shoot(self,bullet_img):
        bullet=Bullet(bullet_img,self.rect.midtop)#first position of the bullet is where the plane is
        self.bullets.add(bullet)
    
    def moveUp(self):
        #judge the range of airplane when moving up
        if self.rect.top<=0:
            self.rect.top=0 
        else:
            self.rect.top-=self.speed #image is moving back
    def moveDown(self):
        if self.rect.top>=SCREEN_HEIGHT-self.rect.height:
            self.rect.top=SCREEN_HEIGHT-self.rect.height
        else:
            self.rect.top+=self.speed
    def moveLeft(self):
        if self.rect.left<=0:
            self.rect.left=0
        else:
            self.rect.left-=self.speed
            
    def moveRight(self):
        if self.rect.left>=SCREEN_WIDTH-self.rect.width:
            self.rect.left=0
        else:
            self.rect.left+=self.speed

class Enemy(pygame.sprite.Sprite):
    def __init__(self, enemy_img,enemy_down_imgs,init_pos):
        pygame.sprite.Sprite.__init__(self)#inherit_class
        self.image=enemy_img
        self.rect=self.image.get_rect()
        self.rect.topleft=init_pos#position of the enemy image
        self.down_imgs=enemy_down_imgs #image when the enemy is down
        self.speed=2
        self.down_index=0
        
    def move(self):
        self.rect.top+=self.speed